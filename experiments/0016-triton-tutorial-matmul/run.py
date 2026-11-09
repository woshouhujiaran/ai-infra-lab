"""Triton tutorial matmul (autotuned, grouped tile order) vs cuBLAS on an RTX 4060 Laptop.

The tutorial (`python/tutorials/03-matrix-multiplication.py`) autotunes 15 tile
configs keyed on (M, N, K), maps program ids to C tiles in groups of
GROUP_SIZE_M rows "to promote L2 data reuse", accumulates in fp32, can fuse a
leaky_relu epilogue and also runs on fp8 inputs.

1. Which of the 15 CUDA configs fit an Ada consumer GPU (99 KiB shared memory per
   block): shared bytes, registers, spills for fp16 and fp8 inputs.
2. Square fp16 TFLOPS vs size for cuBLAS, the tutorial, a copy of the kernel with
   fp16 accumulation (GeForce tensor cores run fp16-accumulate MMA at twice the
   fp32-accumulate rate), a copy with int64 offsets, the tutorial on fp8
   e5m2 / e4m3 inputs and torch._scaled_mm e4m3; autotune time per new shape.
3. GROUP_SIZE_M sweep with the autotuned tile fixed.
4. LLM linear shapes x @ W.T (W stored [N, K] like nn.Linear) with B as the
   column-major W.T view vs a pre-transposed row-major copy vs F.linear; cost of
   re-autotuning for every new batch size M.
5. Fused leaky_relu epilogue vs matmul + F.leaky_relu.
6. Error vs fp64 as K grows for fp32 / fp16 accumulation, cuBLAS with and without
   reduced-precision reductions, and unscaled fp8 casts: the tutorial's in-MMA fp8
   accumulation vs a copy that adds each K block's dot into fp32 (`split_acc`) vs
   torch._scaled_mm with and without use_fast_accum.
7. Shapes that are not multiples of 16 (lost divisibility specialization).
8. int32 overflow of `offs_am * stride_am` once M * K >= 2^31 (child processes).

The tutorial is imported unmodified from the submodule; only its top-level unit
test and benchmark (everything from `torch.manual_seed(0)` on) are skipped. The
fp16-accumulate / int64 variants are copies of `matmul_kernel` in this file.
"""
import ast
import gc
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

# Fresh compile cache so the first autotune includes cold compilation.
os.environ.setdefault("TRITON_CACHE_DIR", tempfile.mkdtemp(prefix="triton-0016-"))

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRITON_DIR = ROOT / "upstream" / "triton"
TUTORIAL = TRITON_DIR / "python" / "tutorials" / "03-matrix-multiplication.py"
sys.path.insert(0, str(ROOT))

from lab import gpu_guard  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.runtime.errors import OutOfResources  # noqa: E402
from triton.testing import do_bench  # noqa: E402

SEED = 0
ROUNDS = 4
SQUARE = [256, 512, 1024, 2048, 3072, 4096, 6144, 8192]
GROUP_SHAPES = [(4096, 4096, 4096), (8192, 8192, 8192), (4096, 14336, 4096)]
GROUP_SIZES = [1, 2, 4, 8, 16, 32, 64]
LLM_WEIGHTS = [(4096, 4096), (14336, 4096)]  # (N, K) = nn.Linear weight shape
LLM_M = [1, 16, 128, 1024, 4096]
RETUNE_M = list(range(1, 33))
EPILOGUE_SHAPES = [(4096, 4096, 4096), (1024, 14336, 4096), (4096, 14336, 4096)]
NUMERICS_MN = 512
NUMERICS_K = [256, 1024, 4096, 16384, 65536]
ODD_SHAPES = [(4096, 4096, 4096), (4096, 4096, 4095), (4096, 4096, 4097), (4096, 4096, 4104),
              (4096, 4097, 4096), (4097, 4096, 4096), (4095, 4095, 4095)]
OVERFLOW_K = 4096
OVERFLOW_N = 64
OVERFLOW_CFG = dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8, num_warps=4, num_stages=4)
FP8 = {"e5m2": torch.float8_e5m2, "e4m3": torch.float8_e4m3fn}


def load_tutorial() -> types.ModuleType:
    tree = ast.parse(TUTORIAL.read_text(encoding="utf-8"), filename=str(TUTORIAL))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and ast.unparse(node).startswith("torch.manual_seed"):
            break
        body.append(node)
    tree.body = body
    mod = types.ModuleType("matmul_tutorial")
    mod.__file__ = str(TUTORIAL)
    sys.modules[mod.__name__] = mod
    exec(compile(tree, str(TUTORIAL), "exec"), mod.__dict__)
    return mod


# The tutorial queries the active device at import time.
GPU = gpu_guard.acquire()
tut = load_tutorial()


@triton.jit
def variant_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   scale, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                   GROUP_SIZE_M: tl.constexpr, ACC: tl.constexpr, INT64_OFFSETS: tl.constexpr):
    """tutorial matmul_kernel with another accumulation mode and / or int64 row / column offsets.

    ACC: "fp32" (tutorial), "fp16" (fp16 MMA accumulator), "split" (acc += dot(a, b)),
    "imprecise_limit" (dot(..., max_num_imprecise_acc=BLOCK_SIZE_K)), "scaled" (acc += dot(a, b) * scale).
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if INT64_OFFSETS:
        offs_am = offs_am.to(tl.int64)
        offs_bn = offs_bn.to(tl.int64)
        offs_cm = offs_cm.to(tl.int64)
        offs_cn = offs_cn.to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    if ACC == "fp16":
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        if ACC == "fp16":
            accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float16)
        elif ACC == "split":
            accumulator += tl.dot(a, b)
        elif ACC == "imprecise_limit":
            accumulator = tl.dot(a, b, accumulator, max_num_imprecise_acc=BLOCK_SIZE_K)
        elif ACC == "scaled":
            accumulator += tl.dot(a, b) * scale
        else:
            accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def autotuned(fn):
    """A separate autotuner (own cache) over the tutorial's config list."""
    return triton.autotune(configs=tut.get_autotune_config(), key=["M", "N", "K"])(fn)


TUNERS = {
    "tutorial": tut.matmul_kernel,
    "tutorial_rowmajor_b": autotuned(tut.matmul_kernel.fn),
    "acc16": autotuned(variant_kernel),
    "int64": autotuned(variant_kernel),
    "split_acc": autotuned(variant_kernel),
    "imprecise_limit": autotuned(variant_kernel),
    "promote": autotuned(variant_kernel),
}
VARIANT_META = {"acc16": dict(ACC="fp16", INT64_OFFSETS=False),
                "int64": dict(ACC="fp32", INT64_OFFSETS=True),
                "split_acc": dict(ACC="split", INT64_OFFSETS=False),
                "imprecise_limit": dict(ACC="imprecise_limit", INT64_OFFSETS=False),
                "promote": dict(ACC="scaled", INT64_OFFSETS=False)}


def launch(kernel, a, b, **meta):
    """The tutorial's matmul() wrapper for an arbitrary (autotuned or plain) kernel."""
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                 **meta)
    return c


def variant(name):
    return lambda a, b: launch(TUNERS[name], a, b, scale=1.0, **VARIANT_META[name])


def tutorial_rowmajor(a, b):
    return launch(TUNERS["tutorial_rowmajor_b"], a, b, ACTIVATION="")


ONE = None


def scaled_mm(a8, b8, fast_accum=False):
    return torch._scaled_mm(a8, b8, scale_a=ONE, scale_b=ONE, out_dtype=torch.float16, use_fast_accum=fast_accum)


def cfg_str(cfg):
    k = cfg.kwargs
    return (f"{k['BLOCK_SIZE_M']}x{k['BLOCK_SIZE_N']}x{k['BLOCK_SIZE_K']} g{k['GROUP_SIZE_M']} "
            f"w{cfg.num_warps} s{cfg.num_stages}")


def tflops(M, N, K, ms):
    return round(2 * M * N * K * 1e-12 / (ms * 1e-3), 2)


def bench(fn):
    return do_bench(fn, return_mode="median")


def bench_rounds(fns: dict) -> dict:
    """{name: {"best_ms", "median_ms", "rounds_ms"}}; order alternates per round so clock drift cancels."""
    warm_gpu(2.0)
    names = list(fns)
    times = {n: [] for n in names}
    for r in range(ROUNDS):
        for n in (names if r % 2 == 0 else names[::-1]):
            times[n].append(bench(fns[n]))
    return {n: {"best_ms": round(min(t), 4), "median_ms": round(statistics.median(t), 4),
                "rounds_ms": [round(x, 4) for x in t]} for n, t in times.items()}


def probe_tflops():
    """cuBLAS fp16 4096^3 after a short warm-up: a per-section reference for the GPU's clock state."""
    a = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
    warm_gpu(2.0)
    ms = min(bench(lambda: a @ a) for _ in range(3))
    return tflops(4096, 4096, 4096, ms)


def first_call(fn, tuner=None):
    """Wall time of the first call (autotune + compile when the key is new) and the tuner's choice."""
    before = dict(tuner.cache) if tuner is not None else None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    info = {"first_call_s": round(time.perf_counter() - t0, 3)}
    if tuner is not None:
        new_keys = [k for k in tuner.cache if k not in before]
        info["retuned"] = bool(new_keys)
        info["config"] = cfg_str(tuner.best_config)
        if new_keys:
            info["bench_s"] = round(tuner.bench_time, 3)
            info["config_ms"] = {cfg_str(c): (round(t[0], 4) if t[0] != float("inf") else None)
                                 for c, t in tuner.configs_timings.items()}
    return out, info


def rel_err(c, ref):
    d = c.double() - ref
    rms = ref.pow(2).mean().sqrt()
    return {"rel_rms": float(f"{(d.pow(2).mean().sqrt() / rms).item():.3g}"),
            "max_abs_rel": float(f"{(d.abs().max() / rms).item():.3g}"),
            "nonfinite": int((~torch.isfinite(c)).sum().item())}


def free():
    gc.collect()
    torch.cuda.empty_cache()


def warm_gpu(seconds=3.0):
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(10):
            a @ a
        torch.cuda.synchronize()


# ---------------------------------------------------------------- 1. config fit

def config_table():
    rows = []
    M = N = K = 4096
    for dt_name, dt in [("fp16", torch.float16), ("fp8e4m3", torch.float8_e4m3fn)]:
        a = torch.empty((16, 16), device="cuda", dtype=torch.float16).to(dt)
        c = torch.empty((16, 16), device="cuda", dtype=torch.float16)
        # fp16: B row-major (stride K, 1); fp8: B column-major as in the tutorial (stride 1, K)
        sbk, sbn = (N, 1) if dt == torch.float16 else (1, K)
        for cfg in tut.get_autotune_config():
            row = {"dtype": dt_name, "config": cfg_str(cfg)}
            try:
                kern = tut.matmul_kernel.fn.warmup(a, a, c, M, N, K, K, 1, sbk, sbn, N, 1, ACTIVATION="",
                                                   num_warps=cfg.num_warps, num_stages=cfg.num_stages, grid=(1, ),
                                                   **cfg.kwargs)
                row["shared_bytes"] = kern.metadata.shared
                try:
                    kern._init_handles()
                    row.update(fits=True, n_regs=kern.n_regs, n_spills=kern.n_spills)
                except OutOfResources as e:
                    row.update(fits=False, error=str(e)[:160])
            except Exception as e:  # noqa: BLE001
                row.update(fits=False, error=f"{type(e).__name__}: {str(e)[:160]}")
            rows.append(row)
            print(f"config {row}", flush=True)
    return rows


# ---------------------------------------------------------------- 2. square sizes

SQUARE_VARIANTS = ["cublas", "tutorial", "acc16", "int64", "tutorial_e5m2", "tutorial_e4m3", "promote_e4m3",
                   "scaled_mm_e4m3", "scaled_mm_e4m3_fast"]
SQUARE_TUNER = {"tutorial": "tutorial", "acc16": "acc16", "int64": "int64", "tutorial_e5m2": "tutorial",
                "tutorial_e4m3": "tutorial", "promote_e4m3": "promote"}


def square():
    rows = []
    for S in SQUARE:
        a = torch.randn((S, S), device="cuda", dtype=torch.float16)
        b = torch.randn((S, S), device="cuda", dtype=torch.float16)
        a8 = {n: a.to(dt) for n, dt in FP8.items()}
        b8 = {n: b.T.to(dt) for n, dt in FP8.items()}  # column-major, as in the tutorial
        fns = {
            "cublas": lambda: torch.matmul(a, b),
            "tutorial": lambda: tut.matmul(a, b),
            "acc16": lambda: variant("acc16")(a, b),
            "int64": lambda: variant("int64")(a, b),
            "tutorial_e5m2": lambda: tut.matmul(a8["e5m2"], b8["e5m2"]),
            "tutorial_e4m3": lambda: tut.matmul(a8["e4m3"], b8["e4m3"]),
            "promote_e4m3": lambda: variant("promote")(a8["e4m3"], b8["e4m3"]),
            "scaled_mm_e4m3": lambda: scaled_mm(a8["e4m3"], b8["e4m3"]),
            "scaled_mm_e4m3_fast": lambda: scaled_mm(a8["e4m3"], b8["e4m3"], fast_accum=True),
        }
        row = {"S": S}
        ref = None
        for name in SQUARE_VARIANTS:
            tuner = TUNERS[SQUARE_TUNER[name]] if name in SQUARE_TUNER else None
            out, info = first_call(fns[name], tuner)
            row[name] = info
            if name == "cublas":
                ref = out.float()
            elif name in ("tutorial", "acc16", "int64"):
                info["max_abs_diff_vs_cublas"] = round((out.float() - ref).abs().max().item(), 4)
        for name, t in bench_rounds(fns).items():
            row[name].update(t, tflops=tflops(S, S, S, t["best_ms"]), tflops_median=tflops(S, S, S, t["median_ms"]))
        rows.append(row)
        print(f"square S={S}: " + ", ".join(f"{n}={row[n]['tflops']}" for n in SQUARE_VARIANTS) + " TFLOPS; "
              + ", ".join(f"{n}:{row[n].get('config')} ({row[n]['first_call_s']}s)" for n in SQUARE_TUNER), flush=True)
        del a, b, a8, b8, fns, ref
        free()
    return rows


# ---------------------------------------------------------------- 3. GROUP_SIZE_M

def group_sweep():
    rows = []
    for M, N, K in GROUP_SHAPES:
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        _, info = first_call(lambda: tut.matmul(a, b), tut.matmul_kernel)
        cfg = tut.matmul_kernel.best_config
        base = {k: v for k, v in cfg.kwargs.items() if k != "GROUP_SIZE_M"}
        tiles = (triton.cdiv(M, base["BLOCK_SIZE_M"]), triton.cdiv(N, base["BLOCK_SIZE_N"]))
        ref = tut.matmul(a, b)
        fns = {}
        for g in GROUP_SIZES:
            fns[g] = (lambda g=g: launch(tut.matmul_kernel.fn, a, b, GROUP_SIZE_M=g, ACTIVATION="",
                                         num_warps=cfg.num_warps, num_stages=cfg.num_stages, **base))
            assert torch.equal(fns[g](), ref)
        for g, t in bench_rounds(fns).items():
            rows.append({"M": M, "N": N, "K": K, "tile_config": info["config"], "tiles_mn": tiles,
                         "group_size_m": g, **t, "tflops": tflops(M, N, K, t["best_ms"]),
                         "tflops_median": tflops(M, N, K, t["median_ms"])})
        print(f"group {M}x{N}x{K} ({info['config']}, tiles {tiles}): "
              + ", ".join(f"g{r['group_size_m']}={r['tflops']}" for r in rows[-len(GROUP_SIZES):]), flush=True)
        del a, b, ref, fns
        free()
    return rows


# ---------------------------------------------------------------- 4. LLM linear shapes + re-autotuning

def llm_linear():
    rows = []
    for N, K in LLM_WEIGHTS:
        w = torch.randn((N, K), device="cuda", dtype=torch.float16) / K**0.5
        w_t = w.T.contiguous()
        for M in LLM_M:
            x = torch.randn((M, K), device="cuda", dtype=torch.float16)
            fns = {
                "F.linear": (lambda: F.linear(x, w), None),
                "tutorial_wT_view": (lambda: tut.matmul(x, w.T), tut.matmul_kernel),
                "tutorial_rowmajor_b": (lambda: tutorial_rowmajor(x, w_t), TUNERS["tutorial_rowmajor_b"]),
            }
            row = {"M": M, "N": N, "K": K}
            ref = None
            for name, (fn, tuner) in fns.items():
                out, info = first_call(fn, tuner)
                if ref is None:
                    ref = out.float()
                else:
                    info["max_abs_diff_vs_cublas"] = round((out.float() - ref).abs().max().item(), 4)
                row[name] = info
            for name, t in bench_rounds({n: f for n, (f, _) in fns.items()}).items():
                ms = t["best_ms"]
                row[name].update(t, us=round(ms * 1e3, 2), tflops=tflops(M, N, K, ms),
                                 weight_gbps=round(N * K * 2 * 1e-9 / (ms * 1e-3), 1))
            rows.append(row)
            print(f"llm M={M} N={N} K={K}: " + ", ".join(
                f"{n}={row[n]['us']}us/{row[n].get('config', '')}" for n in fns), flush=True)
            del x, fns, ref
        del w, w_t
        free()
    return rows


def retune_cost():
    """Fresh autotuner, N = K = 4096, column-major W.T: first-call time for every new M = 1..32."""
    tuner = autotuned(tut.matmul_kernel.fn)
    w = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
    rows = []
    for M in RETUNE_M:
        x = torch.randn((M, 4096), device="cuda", dtype=torch.float16)
        _, info = first_call(lambda: launch(tuner, x, w.T, ACTIVATION=""), tuner)
        t0 = time.perf_counter()
        launch(tuner, x, w.T, ACTIVATION="")
        torch.cuda.synchronize()
        info["second_call_ms"] = round((time.perf_counter() - t0) * 1e3, 3)
        info.pop("config_ms", None)
        rows.append({"M": M, **info})
    print("retune: " + ", ".join(f"M={r['M']}:{r['first_call_s']}s" for r in rows), flush=True)
    del w
    free()
    return rows


# ---------------------------------------------------------------- 5. fused epilogue

def epilogue():
    rows = []
    for M, N, K in EPILOGUE_SHAPES:
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16) / K**0.5
        fns = {
            "tutorial_fused": lambda: tut.matmul(a, b, activation="leaky_relu"),
            "tutorial_then_leaky_relu": lambda: F.leaky_relu(tut.matmul(a, b), 0.01),
            "cublas_then_leaky_relu": lambda: F.leaky_relu(torch.matmul(a, b), 0.01),
            "tutorial_no_activation": lambda: tut.matmul(a, b),
        }
        outs = {n: fn() for n, fn in fns.items()}
        ref = F.leaky_relu(a.double() @ b.double(), 0.01)
        row = {"M": M, "N": N, "K": K, "config": cfg_str(tut.matmul_kernel.best_config)}
        row["fused_vs_separate_max_abs_diff"] = round(
            (outs["tutorial_fused"].float() - outs["tutorial_then_leaky_relu"].float()).abs().max().item(), 5)
        row["fused_vs_separate_differing_elems"] = int(
            (outs["tutorial_fused"] != outs["tutorial_then_leaky_relu"]).sum().item())
        for n in ("tutorial_fused", "tutorial_then_leaky_relu"):
            row[n + "_err"] = rel_err(outs[n], ref)
        del outs, ref
        for n, t in bench_rounds(fns).items():
            row[n + "_ms"] = t["best_ms"]
            row[n + "_median_ms"] = t["median_ms"]
        rows.append(row)
        print(f"epilogue {row}", flush=True)
        del a, b, fns
        free()
    return rows


# ---------------------------------------------------------------- 6. numerics

def numerics():
    rows = []
    M = N = NUMERICS_MN
    for K in NUMERICS_K:
        g = torch.Generator(device="cuda").manual_seed(SEED + K)
        a = torch.randn((M, K), device="cuda", dtype=torch.float16, generator=g)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16, generator=g)
        ref = a.double() @ b.double()
        row = {"K": K, "ref_rms": round(ref.pow(2).mean().sqrt().item(), 2)}
        row["fp16_output_rounding"] = rel_err(ref.half(), ref)
        row["tutorial"] = rel_err(tut.matmul(a, b), ref)
        row["acc16"] = rel_err(variant("acc16")(a, b), ref)
        row["cublas"] = rel_err(torch.matmul(a, b), ref)
        prev = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        row["cublas_no_reduced_reduction"] = rel_err(torch.matmul(a, b), ref)
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = prev
        for n, dt in FP8.items():
            a8, b8 = a.to(dt), b.T.contiguous().T.to(dt)
            ref8 = a8.to(torch.float64) @ b8.to(torch.float64)
            outs = {f"tutorial_{n}": tut.matmul(a8, b8)}
            for v in ("split_acc", "imprecise_limit", "promote"):
                outs[f"{v}_{n}"] = variant(v)(a8, b8)
            if n == "e4m3":
                outs["scaled_mm_e4m3"] = scaled_mm(a8, b8)
                outs["scaled_mm_e4m3_fast"] = scaled_mm(a8, b8, fast_accum=True)
            for name, c in outs.items():
                row[name] = dict(rel_err(c, ref), accumulation_only=rel_err(c, ref8)["rel_rms"])
            del a8, b8, ref8, outs
        rows.append(row)
        print(f"numerics K={K}: " + ", ".join(
            f"{k}={v['rel_rms']}" for k, v in row.items() if isinstance(v, dict)), flush=True)
        del a, b, ref
        free()
    return rows


# ---------------------------------------------------------------- 7. shapes not divisible by 16

def odd_shapes():
    rows = []
    for M, N, K in ODD_SHAPES:
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        ref = a.double() @ b.double()
        fns = {"cublas": (lambda: torch.matmul(a, b), None), "tutorial": (lambda: tut.matmul(a, b), tut.matmul_kernel)}
        row = {"M": M, "N": N, "K": K}
        for n, (fn, tuner) in fns.items():
            out, info = first_call(fn, tuner)
            info.pop("config_ms", None)
            info["err"] = rel_err(out, ref)
            row[n] = info
        for n, t in bench_rounds({n: f for n, (f, _) in fns.items()}).items():
            row[n].update(t, tflops=tflops(M, N, K, t["best_ms"]), tflops_median=tflops(M, N, K, t["median_ms"]))
        rows.append(row)
        print(f"odd {M}x{N}x{K}: cublas={row['cublas']['tflops']} tutorial={row['tutorial']['tflops']} "
              f"({row['tutorial']['config']}) err={row['tutorial']['err']['rel_rms']}", flush=True)
        del a, b, ref, fns
        free()
    return rows


# ---------------------------------------------------------------- 8. int32 overflow (child process)

def overflow_child(case: str) -> dict:
    """A [M, 4096] fp16 with M*K just below / above 2^31 elements, B [4096, 64]; compare every row with cuBLAS."""
    kernel_name, side = case.split(":")
    K, N = OVERFLOW_K, OVERFLOW_N
    M = 2**31 // K + (0 if side == "below" else 256)
    g = torch.Generator(device="cuda").manual_seed(SEED)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16, generator=g)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16, generator=g) / K**0.5
    ref = torch.matmul(a, b)
    torch.cuda.synchronize()
    if kernel_name == "tutorial":
        c = launch(tut.matmul_kernel.fn, a, b, ACTIVATION="", **OVERFLOW_CFG)
    else:
        c = launch(variant_kernel, a, b, scale=1.0, ACC="fp32", INT64_OFFSETS=True, **OVERFLOW_CFG)
    torch.cuda.synchronize()
    bad = torch.cat([((c[i:i + 65536].float() - ref[i:i + 65536].float()).abs().amax(1) > 0.05).nonzero().flatten() + i
                     for i in range(0, M, 65536)])
    return {"case": case, "M": M, "K": K, "N": N, "max_a_offset": (M - 1) * K + K - 1,
            "a_gib": round(a.numel() * 2 / 2**30, 3), "wrong_rows": int(bad.numel()),
            "first_wrong_row": int(bad[0]) if bad.numel() else None,
            "first_overflowing_row": 2**31 // K, "peak_mib": gpu_guard.peak_memory_mib()}


def overflow_tests():
    rows = []
    for case in ["tutorial:below", "tutorial:above", "int64:above"]:
        proc = subprocess.run([sys.executable, __file__, "--overflow", case], capture_output=True, text=True,
                              timeout=900)
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("OVERFLOW ")]
        if proc.returncode == 0 and line:
            rows.append(json.loads(line[-1][len("OVERFLOW "):]))
        else:
            err = [ln.strip() for ln in proc.stderr.splitlines() if "Error" in ln or "error:" in ln]
            rows.append({"case": case, "returncode": proc.returncode,
                         "error": (err or ["?"])[-1][:240]})
        print(f"overflow {rows[-1]}", flush=True)
    return rows


# ---------------------------------------------------------------- report

def upstream_commit() -> str:
    return subprocess.run(["git", "-C", str(TRITON_DIR), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()


def fmt(v, digits=2):
    return "" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))


def write_report(result):
    gpu = result["gpu"]
    lines = [
        f"triton {result['upstream_commit'][:10]} `03-matrix-multiplication.py`, triton-windows {triton.__version__}, "
        f"torch {torch.__version__}, {gpu.get('name', 'cuda')} (memory cap {gpu.get('memory_fraction_cap')}). "
        f"Times are `triton.testing.do_bench` medians (L2 flushed before each rep), best of {ROUNDS} interleaved "
        "rounds whose order alternates (medians of rounds are in results.json). Each timed block starts after "
        "2 s of GPU warm-up.",
        "",
        "Clock-state reference, cuBLAS fp16 4096^3 TFLOPS at the start of each section: "
        + ", ".join(f"{k} {v}" for k, v in result["probe_cublas_4096_tflops"].items()),
        "",
        "## 1. Tutorial CUDA autotune configs at 4096^3 (99 KiB = 101,376 B shared memory per block)",
        "",
        "| config (BMxBNxBK group warps stages) | dtype | shared B | fits | regs | spills | 4096^3 fp16 autotune ms |",
        "|---|---|---|---|---|---|---|",
    ]
    tune4096 = next(r for r in result["square"] if r["S"] == 4096)["tutorial"].get("config_ms", {})
    for r in result["configs"]:
        ms = tune4096.get(r["config"]) if r["dtype"] == "fp16" else None
        lines.append(f"| {r['config']} | {r['dtype']} | {r.get('shared_bytes', '')} | {r['fits']} | "
                     f"{r.get('n_regs', '')} | {r.get('n_spills', '')} | {fmt(ms, 3)} |")
    names = SQUARE_VARIANTS
    lines += [
        "",
        "## 2a. Square M = N = K, TFLOPS",
        "",
        "| S | " + " | ".join(names) + " | tutorial / cuBLAS | acc16 / cuBLAS |",
        "|---|" + "---|" * (len(names) + 2),
    ]
    for r in result["square"]:
        c = r["cublas"]["tflops"]
        lines.append(f"| {r['S']} | " + " | ".join(fmt(r[n]["tflops"]) for n in names)
                     + f" | {r['tutorial']['tflops'] / c:.2f} | {r['acc16']['tflops'] / c:.2f} |")
    lines += [
        "",
        "## 2b. Autotune: chosen config and first-call wall time (s, includes compilation on a cold cache)",
        "",
        "| S | " + " | ".join(SQUARE_TUNER) + " |",
        "|---|" + "---|" * len(SQUARE_TUNER),
    ]
    for r in result["square"]:
        lines.append(f"| {r['S']} | " + " | ".join(
            f"{r[n]['config']} ({r[n]['first_call_s']:.1f} s)" for n in SQUARE_TUNER) + " |")
    lines += [
        "",
        "## 3. GROUP_SIZE_M sweep, TFLOPS (tile fixed to the autotuned winner; outputs bit-identical)",
        "",
        "| M x N x K | tile | tiles (m, n) | " + " | ".join(f"g={g}" for g in GROUP_SIZES) + " | best / g=1 |",
        "|---|---|---|" + "---|" * (len(GROUP_SIZES) + 1),
    ]
    for M, N, K in GROUP_SHAPES:
        rs = [r for r in result["group"] if (r["M"], r["N"], r["K"]) == (M, N, K)]
        best = max(r["tflops"] for r in rs)
        lines.append(f"| {M}x{N}x{K} | {rs[0]['tile_config']} | {tuple(rs[0]['tiles_mn'])} | "
                     + " | ".join(fmt(r["tflops"]) for r in rs) + f" | {best / rs[0]['tflops']:.2f} |")
    llm_names = ["F.linear", "tutorial_wT_view", "tutorial_rowmajor_b"]
    lines += [
        "",
        "## 4a. LLM linear y = x @ W.T, W [N, K] fp16: time in us (weight GB/s for M <= 16, else TFLOPS)",
        "",
        "`tutorial_wT_view` passes the column-major `W.T` view as B; `tutorial_rowmajor_b` uses a pre-transposed "
        "contiguous copy (separate autotuner).",
        "",
        "| N x K | M | " + " | ".join(llm_names) + " | config (view / row-major) |",
        "|---|---|" + "---|" * (len(llm_names) + 1),
    ]
    for r in result["llm"]:
        metric = "weight_gbps" if r["M"] <= 16 else "tflops"
        unit = "GB/s" if metric == "weight_gbps" else "TF"
        lines.append(f"| {r['N']}x{r['K']} | {r['M']} | " + " | ".join(
            f"{fmt(r[n]['us'], 1)} ({fmt(r[n][metric], 1)} {unit})" for n in llm_names)
            + f" | {r['tutorial_wT_view']['config']} / {r['tutorial_rowmajor_b']['config']} |")
    rt = result["retune"]
    lines += [
        "",
        "## 4b. Re-autotuning per new batch size (fresh autotuner, x [M, 4096] @ W.T [4096, 4096])",
        "",
        f"Total first-call time for M = 1..{len(rt)}: {sum(r['first_call_s'] for r in rt):.1f} s; median "
        f"{statistics.median(r['first_call_s'] for r in rt):.2f} s per new M; second call median "
        f"{statistics.median(r['second_call_ms'] for r in rt):.3f} ms.",
        "",
        "| M | first call s | bench s | second call ms | config |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {r['M']} | {r['first_call_s']:.2f} | {fmt(r.get('bench_s'))} | {r['second_call_ms']:.3f} | "
              f"{r['config']} |" for r in rt]
    lines += [
        "",
        "## 5. Fused leaky_relu epilogue, ms",
        "",
        "| M x N x K | fused | tutorial + F.leaky_relu | cuBLAS + F.leaky_relu | tutorial, no activation | "
        "fused vs separate: differing elems / max abs diff | rel err vs fp64 (fused / separate) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in result["epilogue"]:
        lines.append(f"| {r['M']}x{r['N']}x{r['K']} | {r['tutorial_fused_ms']:.3f} | "
                     f"{r['tutorial_then_leaky_relu_ms']:.3f} | {r['cublas_then_leaky_relu_ms']:.3f} | "
                     f"{r['tutorial_no_activation_ms']:.3f} | {r['fused_vs_separate_differing_elems']} / "
                     f"{r['fused_vs_separate_max_abs_diff']} | {r['tutorial_fused_err']['rel_rms']:.3g} / "
                     f"{r['tutorial_then_leaky_relu_err']['rel_rms']:.3g} |")
    fp16_names = ["fp16_output_rounding", "tutorial", "acc16", "cublas", "cublas_no_reduced_reduction"]
    fp8_names = [f"{v}_{n}" for n in FP8 for v in ("tutorial", "split_acc", "imprecise_limit", "promote")]
    fp8_names += ["scaled_mm_e4m3", "scaled_mm_e4m3_fast"]
    lines += [
        "",
        f"## 6a. fp16 inputs: relative RMS error vs fp64, {NUMERICS_MN}x{NUMERICS_MN} output, N(0, 1) inputs",
        "",
        "| K | " + " | ".join(fp16_names) + " |",
        "|---|" + "---|" * len(fp16_names),
    ]
    for r in result["numerics"]:
        lines.append(f"| {r['K']} | " + " | ".join(
            f"{r[n]['rel_rms']:.3g}" + (f", {r[n]['nonfinite']} non-finite" if r[n]["nonfinite"] else "")
            for n in fp16_names) + " |")
    lines += [
        "",
        "## 6b. fp8 inputs: relative RMS error vs fp64 of the quantized fp8 inputs (accumulation + fp16 output "
        "rounding only)",
        "",
        "fp8 inputs are plain casts of the same fp16 inputs (no scaling, as in the tutorial). Total error vs the "
        "original fp16 inputs is dominated by quantization and given in the last two columns.",
        "",
        "| K | " + " | ".join(fp8_names) + " | total err tutorial e5m2 | total err tutorial e4m3 |",
        "|---|" + "---|" * (len(fp8_names) + 2),
    ]
    for r in result["numerics"]:
        lines.append(f"| {r['K']} | " + " | ".join(f"{r[n]['accumulation_only']:.3g}" for n in fp8_names)
                     + f" | {r['tutorial_e5m2']['rel_rms']:.3g} | {r['tutorial_e4m3']['rel_rms']:.3g} |")
    lines += [
        "",
        "## 7. Shapes not divisible by 16, TFLOPS",
        "",
        "| M x N x K | cuBLAS | tutorial | tutorial / cuBLAS | tutorial config | rel err (tutorial / cuBLAS) |",
        "|---|---|---|---|---|---|",
    ]
    for r in result["odd"]:
        lines.append(f"| {r['M']}x{r['N']}x{r['K']} | {fmt(r['cublas']['tflops'])} | {fmt(r['tutorial']['tflops'])} | "
                     f"{r['tutorial']['tflops'] / r['cublas']['tflops']:.2f} | {r['tutorial']['config']} | "
                     f"{r['tutorial']['err']['rel_rms']:.3g} / {r['cublas']['err']['rel_rms']:.3g} |")
    cfg = OVERFLOW_CFG
    lines += [
        "",
        f"## 8. int32 offsets: A [M, {OVERFLOW_K}] fp16 with M*K around 2^31, B [{OVERFLOW_K}, {OVERFLOW_N}] "
        f"(fixed tile {cfg['BLOCK_SIZE_M']}x{cfg['BLOCK_SIZE_N']}x{cfg['BLOCK_SIZE_K']}; each case in a fresh "
        "process; rows compared with cuBLAS)",
        "",
        "| case | M | max A offset | A GiB | wrong rows | first wrong row | result |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in result["overflow"]:
        if "error" in r:
            lines.append(f"| {r['case']} | | | | | | exit {r['returncode']}: {r['error']} |")
        else:
            lines.append(f"| {r['case']} | {r['M']} | {r['max_a_offset']} | {r['a_gib']} | {r['wrong_rows']} | "
                         f"{fmt(r['first_wrong_row'])} | {'correct' if r['wrong_rows'] == 0 else 'WRONG'} |")
    lines += ["", f"Runtime: {result['runtime_s']:.0f} s; section timings (s): {result['timings']}"]
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    global ONE
    t_start = time.time()
    torch.manual_seed(SEED)
    ONE = torch.ones((), device="cuda")
    timings = {}
    result = {
        "experiment": HERE.name,
        "upstream": "triton-lang/triton",
        "upstream_commit": upstream_commit(),
        "component": "python/tutorials/03-matrix-multiplication.py (matmul_kernel + autotune configs + matmul)",
        "seed": SEED,
        "rounds": ROUNDS,
        "triton": triton.__version__,
        "torch": torch.__version__,
        "gpu": GPU,
        "triton_cache_dir": os.environ["TRITON_CACHE_DIR"],
    }
    sections = [("overflow", overflow_tests), ("configs", config_table), ("warm", warm_gpu), ("square", square),
                ("group", group_sweep), ("llm", llm_linear), ("retune", retune_cost), ("epilogue", epilogue),
                ("numerics", numerics), ("odd", odd_shapes)]
    probes = {}
    for name, fn in sections:
        if name in ("square", "group", "llm", "epilogue", "odd"):
            probes[name] = probe_tflops()
            print(f"probe before {name}: cuBLAS 4096^3 {probes[name]} TFLOPS", flush=True)
            free()
        t = time.time()
        out = fn()
        timings[name + "_s"] = round(time.time() - t, 1)
        if out is not None:
            result[name] = out
        free()
    result["probe_cublas_4096_tflops"] = probes
    result["peak_memory_mib"] = gpu_guard.peak_memory_mib()
    result["timings"] = timings
    result["runtime_s"] = round(time.time() - t_start, 1)
    (HERE / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--overflow":
        print("OVERFLOW " + json.dumps(overflow_child(sys.argv[2])), flush=True)
    else:
        main()
