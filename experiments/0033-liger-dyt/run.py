"""Liger-Kernel Dynamic Tanh (`LigerDyTFunction`).

DyT (arxiv 2503.10622 / Liger) replaces LayerNorm with
    y = gamma * tanh(alpha * x) + beta
(beta optional). Forward is a per-row autotuned Triton kernel
(BLOCK_N in {1024,2048,4096}); backward partitions rows across SMs,
accumulates fp32 partial dgamma/dbeta/dalpha into `empty`/`zeros`
buffers, then `.sum(0)`. DA is sized `(NUM_SMS, cdiv(N, 512))` while
the grid's program_id(0) is `cdiv(N, BLOCK_N)` with BLOCK_N >= 1024,
so most DA slots stay unused (autotune uses `reset_to_zero=["DA"]`).
`@ensure_contiguous` on both forward and backward. Distinct from
0031 (LayerNorm) / 0022 (RMSNorm).

1. Numerics vs fp64 textbook + eager DyT / torch.compile for
   fp32/fp16/bf16, with and without beta, alpha in {0.5, 1, 2}.
2. Edges: large |alpha*x|, NaN/Inf, alpha=0, huge alpha, all-equal.
3. Autotune: cold vs warm latency per N; DA buffer shape vs winning
   BLOCK_N; bitwise dalpha/dgamma determinism over repeats.
4. Layouts: row-strided / transposed / 3D; Module vs Function.
5. Memory / latency at LLM hidden sizes vs eager DyT / F.layer_norm /
   torch.compile(eager DyT).
6. Elementwise bandwidth vs H (bf16).

Liger-Kernel is imported unmodified from the submodule.
"""
import gc
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LIGER_DIR = ROOT / "upstream" / "Liger-Kernel"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LIGER_DIR / "src"))

from lab import gpu_guard  # noqa: E402

GPU = gpu_guard.acquire()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from triton.testing import do_bench  # noqa: E402
from liger_kernel.ops.dyt import (  # noqa: E402
    LigerDyTFunction,
    liger_dyt_bwd,
    liger_dyt_fwd,
    _dyt_bwd_kernel,
)
from liger_kernel.transformers.dyt import LigerDyT  # noqa: E402

SEED = 0
DEV = "cuda"
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
NUM_T, NUM_H = 2048, 4096
MEM_SHAPES = [
    ("GPT-2-124M", 768),
    ("BERT-base", 768),
    ("Qwen3-0.6B", 1024),
    ("Llama-3.2-1B", 2048),
    ("Llama-3-8B", 4096),
]
MEM_TOKENS = [2048, 8192, 16384]
BW_T = 4096
BW_HS = [512, 768, 1024, 2048, 2049, 4096, 8192, 16384]
DET_REPEATS = 8
AUTOTUNE_NS = [512, 768, 1024, 2048, 4096, 8192]

torch._dynamo.config.recompile_limit = 64
torch._functorch.config.donated_buffer = False


def liger(x, alpha, gamma, beta):
    return LigerDyTFunction.apply(x, alpha, gamma, beta)


def eager_dyt(x, alpha, gamma, beta):
    y = torch.tanh(alpha * x) * gamma
    if beta is not None:
        y = y + beta
    return y


def fp64_dyt(x, alpha, gamma, beta):
    x64 = x.to(torch.float64)
    a64 = alpha.to(torch.float64)
    g64 = gamma.to(torch.float64)
    y = torch.tanh(a64 * x64) * g64
    if beta is not None:
        y = y + beta.to(torch.float64)
    return y


def rel(a, b):
    a, b = a.detach().double(), b.detach().double()
    den = b.norm().clamp_min(1e-300)
    return float((a - b).norm() / den)


def fmt(x, digits=4):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, str):
        return x.splitlines()[0][:100]
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")
    if isinstance(x, float) and x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e5):
        return f"{x:.{digits - 1}e}"
    if isinstance(x, float):
        return f"{x:.{digits}g}"
    return str(x)


def table(headers, rows):
    cells = [[fmt(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in cells:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    out = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |", sep]
    for row in cells:
        out.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    return "\n".join(out)


_PEAK_MIB = 0.0


def cleanup():
    global _PEAK_MIB
    _PEAK_MIB = max(_PEAK_MIB, gpu_guard.peak_memory_mib())
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def make_params(H, dtype, seed=SEED, alpha0=0.5, with_beta=True):
    g = torch.Generator(device=DEV).manual_seed(seed)
    alpha = torch.full((1,), alpha0, device=DEV, dtype=dtype)
    gamma = torch.randn(H, device=DEV, dtype=dtype, generator=g)
    beta = torch.randn(H, device=DEV, dtype=dtype, generator=g) if with_beta else None
    return alpha, gamma, beta


def make_inputs(T, H, dtype, seed=SEED, scale=1.0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = scale * torch.randn(T, H, device=DEV, dtype=dtype, generator=g)
    dy = 0.1 * torch.randn(T, H, device=DEV, dtype=dtype, generator=g)
    return x, dy


def run_variant(fn, x, alpha, gamma, beta, dy):
    xx = x.detach().clone().requires_grad_(True)
    aa = alpha.detach().clone().requires_grad_(True)
    gg = gamma.detach().clone().requires_grad_(True)
    bb = None
    if beta is not None:
        bb = beta.detach().clone().requires_grad_(True)
    y = fn(xx, aa, gg, bb)
    torch.autograd.backward(y, dy)
    out = {
        "y": y.detach(),
        "dx": xx.grad.detach(),
        "dalpha": aa.grad.detach(),
        "dgamma": gg.grad.detach(),
        "dbeta": None if bb is None else bb.grad.detach(),
    }
    return out


# ---------------------------------------------------------------- 1. numerics
def sec_numerics():
    rows = []
    for dname, dt in DTYPES.items():
        for with_beta in (True, False):
            for alpha0 in (0.5, 1.0, 2.0):
                x, dy = make_inputs(NUM_T, NUM_H, dt)
                alpha, gamma, beta = make_params(NUM_H, dt, alpha0=alpha0, with_beta=with_beta)

                y_ref = fp64_dyt(x, alpha, gamma, beta)
                xd = x.double().requires_grad_(True)
                ad = alpha.double().requires_grad_(True)
                gd = gamma.double().requires_grad_(True)
                bd = None if beta is None else beta.double().requires_grad_(True)
                yd = fp64_dyt(xd, ad, gd, bd)
                torch.autograd.backward(yd, dy.double())
                floor_y = y_ref.to(dt)
                floor_dx = xd.grad.to(dt)
                floor_da = ad.grad.to(dt)
                floor_dg = gd.grad.to(dt)
                floor_db = None if bd is None else bd.grad.to(dt)

                def pack(r):
                    p = {
                        "y_rel": rel(r["y"], y_ref),
                        "dx_rel": rel(r["dx"], xd.grad),
                        "dalpha_rel": rel(r["dalpha"], ad.grad),
                        "dgamma_rel": rel(r["dgamma"], gd.grad),
                        "y_vs_floor": rel(r["y"], floor_y),
                        "dx_vs_floor": rel(r["dx"], floor_dx),
                        "dalpha_vs_floor": rel(r["dalpha"], floor_da),
                        "dgamma_vs_floor": rel(r["dgamma"], floor_dg),
                    }
                    if floor_db is not None and r["dbeta"] is not None:
                        p["dbeta_rel"] = rel(r["dbeta"], bd.grad)
                        p["dbeta_vs_floor"] = rel(r["dbeta"], floor_db)
                    return p

                rl = run_variant(liger, x, alpha, gamma, beta, dy)
                re = run_variant(eager_dyt, x, alpha, gamma, beta, dy)
                variants = {"liger": pack(rl), "eager": pack(re)}
                variants["liger_vs_eager"] = {
                    "y": rel(rl["y"], re["y"]),
                    "dx": rel(rl["dx"], re["dx"]),
                    "dalpha": rel(rl["dalpha"], re["dalpha"]),
                    "dgamma": rel(rl["dgamma"], re["dgamma"]),
                }
                if with_beta:
                    variants["liger_vs_eager"]["dbeta"] = rel(rl["dbeta"], re["dbeta"])

                try:
                    compiled = torch.compile(
                        lambda a, al, g, b: eager_dyt(a, al, g, b), fullgraph=False
                    )
                    rc = run_variant(compiled, x, alpha, gamma, beta, dy)
                    variants["torch_compile"] = pack(rc)
                except Exception as e:
                    variants["torch_compile"] = {"error": f"{type(e).__name__}: {e}"}

                # LayerNorm floor comparison (same x/weights shape, not same math)
                try:
                    w = torch.ones(NUM_H, device=DEV, dtype=dt)
                    b = torch.zeros(NUM_H, device=DEV, dtype=dt)
                    y_ln = F.layer_norm(x, (NUM_H,), w, b, 1e-6)
                    variants["ln_y_rel_vs_dyt_fp64"] = rel(y_ln, y_ref)
                except Exception as e:
                    variants["ln_y_rel_vs_dyt_fp64"] = f"{type(e).__name__}: {e}"

                row = {
                    "dtype": dname,
                    "with_beta": with_beta,
                    "alpha0": alpha0,
                    "floor": {
                        "y": rel(floor_y, y_ref),
                        "dx": rel(floor_dx, xd.grad),
                        "dalpha": rel(floor_da, ad.grad),
                        "dgamma": rel(floor_dg, gd.grad),
                    },
                    **variants,
                }
                rows.append(row)
                print(
                    "numerics", dname, f"beta={with_beta}", f"a={alpha0}",
                    "liger_y", variants["liger"]["y_rel"],
                    "eager_y", variants["eager"]["y_rel"],
                    "vs_eager", variants["liger_vs_eager"]["y"],
                    flush=True,
                )
                cleanup()
    return {"rows": rows}


# ---------------------------------------------------------------- 2. edges
def sec_edges():
    rows = []
    T, H = 64, 2048
    dtype = torch.bfloat16
    g = torch.Generator(device=DEV).manual_seed(SEED + 20)
    alpha, gamma, beta = make_params(H, dtype, seed=SEED + 21, alpha0=0.5, with_beta=True)
    base = torch.randn(T, H, device=DEV, dtype=dtype, generator=g)

    cases = {
        "normal": base.clone(),
        "large_x_1e3": 1000.0 * base,
        "large_x_1e6": 1e6 * base,
        "all_zero": torch.zeros(T, H, device=DEV, dtype=dtype),
        "all_one": torch.ones(T, H, device=DEV, dtype=dtype),
        "nan_cell": base.clone(),
        "inf_cell": base.clone(),
        "neg_inf_cell": base.clone(),
    }
    cases["nan_cell"][0, 0] = float("nan")
    cases["inf_cell"][1, 1] = float("inf")
    cases["neg_inf_cell"][2, 2] = float("-inf")

    for name, x in cases.items():
        cell = {"case": name, "alpha0": 0.5}
        try:
            y_ref = fp64_dyt(x, alpha, gamma, beta)
            ye = eager_dyt(x, alpha, gamma, beta)
            yl = liger(x, alpha, gamma, beta)
            cell["y_liger_vs_fp64"] = rel(yl, y_ref)
            cell["y_eager_vs_fp64"] = rel(ye, y_ref)
            cell["y_liger_vs_eager"] = rel(yl, ye)
            cell["y_has_nan"] = bool(torch.isnan(yl).any())
            cell["eager_has_nan"] = bool(torch.isnan(ye).any())
            dy = 0.1 * torch.randn_like(x)
            if torch.isnan(yl).any() or torch.isnan(ye).any():
                cell["bwd"] = "skipped_nan_y"
            else:
                rl = run_variant(liger, x, alpha, gamma, beta, dy)
                re = run_variant(eager_dyt, x, alpha, gamma, beta, dy)
                cell["dx_vs_eager"] = rel(rl["dx"], re["dx"])
                cell["dalpha_vs_eager"] = rel(rl["dalpha"], re["dalpha"])
                cell["dgamma_vs_eager"] = rel(rl["dgamma"], re["dgamma"])
                cell["dbeta_vs_eager"] = rel(rl["dbeta"], re["dbeta"])
        except Exception as e:
            cell["error"] = f"{type(e).__name__}: {e}"
        rows.append(cell)
        print("edges", name, {k: cell[k] for k in cell if k != "case"}, flush=True)
        cleanup()

    # alpha extremes on normal x
    for alpha0, label in [(0.0, "alpha_0"), (1e-8, "alpha_tiny"), (100.0, "alpha_100"), (1e6, "alpha_1e6")]:
        x = base.clone()
        a, gmm, b = make_params(H, dtype, seed=SEED + 22, alpha0=alpha0, with_beta=True)
        cell = {"case": label, "alpha0": alpha0}
        try:
            y_ref = fp64_dyt(x, a, gmm, b)
            ye = eager_dyt(x, a, gmm, b)
            yl = liger(x, a, gmm, b)
            cell["y_liger_vs_fp64"] = rel(yl, y_ref)
            cell["y_eager_vs_fp64"] = rel(ye, y_ref)
            cell["y_liger_vs_eager"] = rel(yl, ye)
            cell["y_has_nan"] = bool(torch.isnan(yl).any())
            dy = 0.1 * torch.randn_like(x)
            rl = run_variant(liger, x, a, gmm, b, dy)
            re = run_variant(eager_dyt, x, a, gmm, b, dy)
            cell["dx_vs_eager"] = rel(rl["dx"], re["dx"])
            cell["dalpha_vs_eager"] = rel(rl["dalpha"], re["dalpha"])
        except Exception as e:
            cell["error"] = f"{type(e).__name__}: {e}"
        rows.append(cell)
        print("edges", label, cell, flush=True)
        cleanup()
    return {"rows": rows}


# ---------------------------------------------------------------- 3. autotune / DA / determinism
def sec_autotune():
    out = {"cold_vs_warm": [], "da_shape": [], "determinism": [], "no_beta": []}
    dtype = torch.bfloat16
    T = 4096

    for H in AUTOTUNE_NS:
        alpha, gamma, beta = make_params(H, dtype, seed=SEED + H, with_beta=True)
        x, dy = make_inputs(T, H, dtype, seed=SEED + H + 1)
        # cold: first call after empty_cache may still hit cached autotune for same N
        # measure by timing a fresh N if possible; report both
        cleanup()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = liger(x, alpha, gamma, beta)
        torch.cuda.synchronize()
        cold_ms = (time.perf_counter() - t0) * 1e3
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            y = liger(x, alpha, gamma, beta)
        torch.cuda.synchronize()
        warm_ms = (time.perf_counter() - t0) * 1e3 / 10
        # also do_bench
        try:
            bench_ms = do_bench(lambda: liger(x, alpha, gamma, beta), warmup=5, rep=30)
        except Exception as e:
            bench_ms = f"{type(e).__name__}: {e}"
        cell = {
            "H": H,
            "cold_ms": cold_ms,
            "warm_ms": warm_ms,
            "bench_fwd_ms": bench_ms,
            "cold_over_warm": (cold_ms / warm_ms) if warm_ms > 0 else None,
        }
        out["cold_vs_warm"].append(cell)
        print("autotune", cell, flush=True)
        cleanup()

    # DA buffer shape vs grid
    for H in [512, 1024, 2048, 4096, 8192]:
        alpha, gamma, beta = make_params(H, dtype, with_beta=True)
        x, dy = make_inputs(256, H, dtype)
        # warm kernels so we can inspect best config
        _ = liger(x, alpha, gamma, beta)
        rl = run_variant(liger, x, alpha, gamma, beta, dy)
        re = run_variant(eager_dyt, x, alpha, gamma, beta, dy)
        # probe low-level bwd buffer sizing
        NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
        da_cols = (H + 511) // 512
        # winning config from autotune cache if available
        conf = None
        try:
            conf = _dyt_bwd_kernel.best_config
            conf = {
                "BLOCK_N": conf.kwargs.get("BLOCK_N"),
                "num_warps": conf.num_warps,
                "num_stages": conf.num_stages,
            }
        except Exception as e:
            conf = {"error": f"{type(e).__name__}: {e}"}
        cell = {
            "H": H,
            "NUM_SMS": NUM_SMS,
            "da_shape": [NUM_SMS, da_cols],
            "da_slots": NUM_SMS * da_cols,
            "cdiv_H_512": da_cols,
            "best_config": conf,
            "dalpha_vs_eager": rel(rl["dalpha"], re["dalpha"]),
            "dgamma_vs_eager": rel(rl["dgamma"], re["dgamma"]),
            "dbeta_vs_eager": rel(rl["dbeta"], re["dbeta"]),
            "dx_vs_eager": rel(rl["dx"], re["dx"]),
        }
        if isinstance(conf, dict) and conf.get("BLOCK_N"):
            bn = conf["BLOCK_N"]
            cell["programs_x"] = (H + bn - 1) // bn
            cell["da_slots_used_per_sm"] = cell["programs_x"]
            cell["da_slot_waste_frac"] = 1.0 - (cell["programs_x"] / da_cols)
        out["da_shape"].append(cell)
        print("da_shape", cell, flush=True)
        cleanup()

    # bitwise determinism of parameter grads
    H = 4096
    alpha, gamma, beta = make_params(H, dtype, with_beta=True)
    x, dy = make_inputs(NUM_T, H, dtype)
    refs = []
    for i in range(DET_REPEATS):
        r = run_variant(liger, x, alpha, gamma, beta, dy)
        refs.append(r)
        cleanup()
    cell = {
        "repeats": DET_REPEATS,
        "dx_all_equal": all(torch.equal(refs[0]["dx"], refs[i]["dx"]) for i in range(1, DET_REPEATS)),
        "dalpha_all_equal": all(
            torch.equal(refs[0]["dalpha"], refs[i]["dalpha"]) for i in range(1, DET_REPEATS)
        ),
        "dgamma_all_equal": all(
            torch.equal(refs[0]["dgamma"], refs[i]["dgamma"]) for i in range(1, DET_REPEATS)
        ),
        "dbeta_all_equal": all(
            torch.equal(refs[0]["dbeta"], refs[i]["dbeta"]) for i in range(1, DET_REPEATS)
        ),
        "dalpha_max_pairwise_rel": max(
            rel(refs[i]["dalpha"], refs[j]["dalpha"])
            for i in range(DET_REPEATS) for j in range(i + 1, DET_REPEATS)
        ) if DET_REPEATS > 1 else 0.0,
    }
    out["determinism"].append(cell)
    print("determinism", cell, flush=True)

    # beta=None path
    alpha, gamma, _ = make_params(H, dtype, with_beta=False)
    x, dy = make_inputs(512, H, dtype)
    try:
        rl = run_variant(liger, x, alpha, gamma, None, dy)
        re = run_variant(eager_dyt, x, alpha, gamma, None, dy)
        out["no_beta"].append({
            "y_vs_eager": rel(rl["y"], re["y"]),
            "dx_vs_eager": rel(rl["dx"], re["dx"]),
            "dalpha_vs_eager": rel(rl["dalpha"], re["dalpha"]),
            "dgamma_vs_eager": rel(rl["dgamma"], re["dgamma"]),
            "dbeta_is_none": rl["dbeta"] is None,
        })
    except Exception as e:
        out["no_beta"].append({"error": f"{type(e).__name__}: {e}"})
    print("no_beta", out["no_beta"][-1], flush=True)
    cleanup()
    return out


# ---------------------------------------------------------------- 4. layouts
def sec_layouts():
    rows = []
    T, H = 1024, 2048
    dtype = torch.bfloat16
    g = torch.Generator(device=DEV).manual_seed(SEED + 30)
    alpha, gamma, beta = make_params(H, dtype, seed=SEED + 31, with_beta=True)
    base = torch.randn(T, H + 64, device=DEV, dtype=dtype, generator=g)
    dy_c = torch.randn(T, H, device=DEV, dtype=dtype, generator=g)

    cases = []
    x0 = base[:, :H].contiguous()
    cases.append(("contiguous", x0, dy_c))
    x_row = base[:, :H]
    assert not x_row.is_contiguous()
    cases.append(("row_strided_x", x_row, dy_c))
    x_t = torch.randn(H, T, device=DEV, dtype=dtype, generator=g).t()
    cases.append(("transposed_x", x_t, dy_c))
    B, S = 4, 256
    x3 = torch.randn(B, S, H, device=DEV, dtype=dtype, generator=g)
    dy3 = torch.randn(B, S, H, device=DEV, dtype=dtype, generator=g)
    cases.append(("input_3d", x3, dy3))

    for name, x, dy in cases:
        cell = {
            "case": name,
            "x_contiguous": bool(x.is_contiguous()),
            "x_stride": list(x.stride()),
            "shape": list(x.shape),
        }
        try:
            canary = None
            if name == "row_strided_x":
                canary = base[:, H:H + 8].clone()
            xl = x.detach().clone().requires_grad_(True)
            # eager needs same contiguous semantics for fair compare
            xe = x.contiguous().detach().clone().requires_grad_(True)
            aa = alpha.detach().clone().requires_grad_(True)
            gg = gamma.detach().clone().requires_grad_(True)
            bb = beta.detach().clone().requires_grad_(True)
            yl = liger(xl, aa, gg, bb)
            ye = eager_dyt(xe, aa.detach(), gg.detach(), bb.detach())
            # re-run eager with grads
            ae = alpha.detach().clone().requires_grad_(True)
            ge = gamma.detach().clone().requires_grad_(True)
            be = beta.detach().clone().requires_grad_(True)
            ye = eager_dyt(xe, ae, ge, be)
            torch.autograd.backward(yl, dy)
            torch.autograd.backward(ye, dy.contiguous())
            cell["y_vs_eager"] = rel(yl, ye)
            cell["dx_vs_eager"] = rel(xl.grad, xe.grad)
            if canary is not None:
                cell["canary_unchanged"] = bool(torch.equal(base[:, H:H + 8], canary))
            if name == "contiguous":
                mod = LigerDyT(H, beta=True, init_alpha=0.5).to(DEV)
                with torch.no_grad():
                    mod.alpha.copy_(alpha)
                    mod.gamma.copy_(gamma)
                    mod.beta.copy_(beta)
                xm = x.detach().clone().requires_grad_(True)
                ym = mod(xm)
                torch.autograd.backward(ym, dy)
                cell["module_vs_fn_y"] = rel(ym, yl)
                cell["module_vs_fn_dx"] = rel(xm.grad, xl.grad)
        except Exception as e:
            cell["error"] = f"{type(e).__name__}: {e}"
        rows.append(cell)
        print("layouts", cell, flush=True)
        cleanup()
    return {"rows": rows}


# ---------------------------------------------------------------- 5. memory / latency
def sec_memory():
    rows = []
    dtype = torch.bfloat16
    for label, H in MEM_SHAPES:
        for T in MEM_TOKENS:
            cell = {"label": label, "T": T, "H": H}
            x, dy = make_inputs(T, H, dtype, seed=SEED + T + H)
            alpha, gamma, beta = make_params(H, dtype, seed=SEED + H, with_beta=True)
            w = torch.ones(H, device=DEV, dtype=dtype)
            b = torch.zeros(H, device=DEV, dtype=dtype)

            def measure(fn, name, use_ln=False):
                cleanup()
                torch.cuda.reset_peak_memory_stats()
                xx = x.detach().clone().requires_grad_(True)
                if use_ln:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    y = fn(xx)
                    torch.cuda.synchronize()
                    fwd_ms = (time.perf_counter() - t0) * 1e3
                    alloc_fwd = torch.cuda.memory_allocated() / 2**20
                    torch.autograd.backward(y, dy)
                    torch.cuda.synchronize()
                    peak = gpu_guard.peak_memory_mib()

                    def fwd_only():
                        with torch.no_grad():
                            fn(x)

                    def fwd_bwd():
                        xx2 = x.detach().clone().requires_grad_(True)
                        yy = fn(xx2)
                        torch.autograd.backward(yy, dy)
                else:
                    aa = alpha.detach().clone().requires_grad_(True)
                    gg = gamma.detach().clone().requires_grad_(True)
                    bb = beta.detach().clone().requires_grad_(True)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    y = fn(xx, aa, gg, bb)
                    torch.cuda.synchronize()
                    fwd_ms = (time.perf_counter() - t0) * 1e3
                    alloc_fwd = torch.cuda.memory_allocated() / 2**20
                    torch.autograd.backward(y, dy)
                    torch.cuda.synchronize()
                    peak = gpu_guard.peak_memory_mib()

                    def fwd_only():
                        with torch.no_grad():
                            fn(x, alpha, gamma, beta)

                    def fwd_bwd():
                        xx2 = x.detach().clone().requires_grad_(True)
                        aa2 = alpha.detach().clone().requires_grad_(True)
                        gg2 = gamma.detach().clone().requires_grad_(True)
                        bb2 = beta.detach().clone().requires_grad_(True)
                        yy = fn(xx2, aa2, gg2, bb2)
                        torch.autograd.backward(yy, dy)

                try:
                    ms_fwd = do_bench(fwd_only, warmup=5, rep=20)
                except Exception as e:
                    ms_fwd = f"{type(e).__name__}: {e}"
                try:
                    ms_fb = do_bench(fwd_bwd, warmup=3, rep=10)
                except Exception as e:
                    ms_fb = f"{type(e).__name__}: {e}"
                return {
                    "fwd_ms_once": fwd_ms,
                    "held_after_fwd_mib": round(alloc_fwd, 2),
                    "peak_mib": peak,
                    "bench_fwd_ms": ms_fwd,
                    "bench_fwd_bwd_ms": ms_fb,
                }

            cell["liger"] = measure(liger, "liger")
            cell["eager_dyt"] = measure(eager_dyt, "eager")
            cell["F_layer_norm"] = measure(
                lambda a: F.layer_norm(a, (H,), w, b, 1e-6), "ln", use_ln=True
            )
            try:
                compiled = torch.compile(
                    lambda a, al, g, bt: eager_dyt(a, al, g, bt), fullgraph=False
                )
                _ = compiled(
                    x.detach().clone().requires_grad_(True),
                    alpha.detach().clone().requires_grad_(True),
                    gamma.detach().clone().requires_grad_(True),
                    beta.detach().clone().requires_grad_(True),
                )
                torch.cuda.synchronize()
                cell["torch_compile"] = measure(compiled, "compile")
            except Exception as e:
                cell["torch_compile"] = {"error": f"{type(e).__name__}: {e}"}

            cell["input_mib"] = round(T * H * 2 / 2**20, 3)
            rows.append(cell)
            print(
                "memory", label, T, H,
                "liger", cell["liger"].get("bench_fwd_ms"),
                "eager", cell["eager_dyt"].get("bench_fwd_ms"),
                "ln", cell["F_layer_norm"].get("bench_fwd_ms"),
                flush=True,
            )
            cleanup()
    return {"rows": rows}


# ---------------------------------------------------------------- 6. bandwidth
def sec_bandwidth():
    rows = []
    dtype = torch.bfloat16
    for H in BW_HS:
        cell = {"H": H, "T": BW_T}
        x = torch.randn(BW_T, H, device=DEV, dtype=dtype)
        dy = torch.randn(BW_T, H, device=DEV, dtype=dtype)
        alpha, gamma, beta = make_params(H, dtype, with_beta=True)
        # fwd: read x,alpha,gamma,beta write y ≈ 2*T*H*2 (+ tiny params)
        fwd_bytes = 2 * BW_T * H * 2
        # bwd: read dy,x,gamma write dx + param grads ≈ 3*T*H*2
        bwd_bytes = 3 * BW_T * H * 2

        def gbs(ms, nbytes):
            if not isinstance(ms, (int, float)) or ms <= 0:
                return None
            return (nbytes / 1e9) / (ms / 1e3)

        def bench_fwd(fn, use_ln=False):
            if use_ln:
                return do_bench(lambda: fn(x), warmup=5, rep=30)
            return do_bench(lambda: fn(x, alpha, gamma, beta), warmup=5, rep=30)

        def bench_bwd(fn, use_ln=False):
            if use_ln:
                def go():
                    xx = x.detach().clone().requires_grad_(True)
                    y = fn(xx)
                    torch.autograd.backward(y, dy)
                return do_bench(go, warmup=3, rep=15)

            def go():
                xx = x.detach().clone().requires_grad_(True)
                aa = alpha.detach().clone().requires_grad_(True)
                gg = gamma.detach().clone().requires_grad_(True)
                bb = beta.detach().clone().requires_grad_(True)
                y = fn(xx, aa, gg, bb)
                torch.autograd.backward(y, dy)
            return do_bench(go, warmup=3, rep=15)

        w = torch.ones(H, device=DEV, dtype=dtype)
        b = torch.zeros(H, device=DEV, dtype=dtype)
        for name, fn, use_ln in [
            ("liger", liger, False),
            ("eager_dyt", eager_dyt, False),
            ("F_layer_norm", lambda a: F.layer_norm(a, (H,), w, b, 1e-6), True),
        ]:
            try:
                ms_f = bench_fwd(fn, use_ln=use_ln)
                ms_b = bench_bwd(fn, use_ln=use_ln)
                cell[name] = {
                    "fwd_ms": ms_f,
                    "fwd_gbs": gbs(ms_f, fwd_bytes),
                    "fwd_bwd_ms": ms_b,
                    "bwd_approx_gbs": gbs(ms_b, fwd_bytes + bwd_bytes),
                }
            except Exception as e:
                cell[name] = {"error": f"{type(e).__name__}: {e}"}

        try:
            compiled = torch.compile(
                lambda a, al, g, bt: eager_dyt(a, al, g, bt), fullgraph=False
            )
            _ = compiled(x, alpha, gamma, beta)
            torch.cuda.synchronize()
            ms_f = bench_fwd(compiled)
            ms_b = bench_bwd(compiled)
            cell["torch_compile"] = {
                "fwd_ms": ms_f,
                "fwd_gbs": gbs(ms_f, fwd_bytes),
                "fwd_bwd_ms": ms_b,
                "bwd_approx_gbs": gbs(ms_b, fwd_bytes + bwd_bytes),
            }
        except Exception as e:
            cell["torch_compile"] = {"error": f"{type(e).__name__}: {e}"}

        rows.append(cell)
        print(
            "bw", H,
            "liger", cell.get("liger", {}).get("fwd_gbs"),
            "eager", cell.get("eager_dyt", {}).get("fwd_gbs"),
            "ln", cell.get("F_layer_norm", {}).get("fwd_gbs"),
            flush=True,
        )
        cleanup()
    return {"rows": rows}


def write_results(data):
    results_json = HERE / "results.json"
    results_md = HERE / "results.md"
    results_json.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    lines = [
        "# 0033 — Liger DyT results",
        "",
        f"GPU: {data['gpu']}",
        f"Peak allocated (tracker): {data['peak_mib']} MiB",
        f"Wall time: {data['wall_s']:.1f}s",
        "",
        "## 1. Numerics",
        "",
    ]
    num_rows = []
    for r in data["numerics"]["rows"]:
        lig = r["liger"]
        num_rows.append([
            r["dtype"], r["with_beta"], r["alpha0"],
            lig["y_rel"], lig["dx_rel"], lig["dalpha_rel"], lig["dgamma_rel"],
            lig.get("dbeta_rel"),
            lig["y_vs_floor"], lig["dx_vs_floor"],
            r["liger_vs_eager"]["y"], r["liger_vs_eager"]["dx"],
            r["liger_vs_eager"]["dalpha"], r["liger_vs_eager"]["dgamma"],
        ])
    lines.append(table(
        ["dtype", "beta", "alpha", "y_rel", "dx_rel", "da_rel", "dg_rel", "db_rel",
         "y_floor", "dx_floor", "y_vs_e", "dx_vs_e", "da_vs_e", "dg_vs_e"],
        num_rows,
    ))

    lines += ["", "## 2. Edges", ""]
    erows = []
    for r in data["edges"]["rows"]:
        erows.append([
            r["case"], r.get("alpha0"),
            r.get("y_liger_vs_fp64"), r.get("y_eager_vs_fp64"), r.get("y_liger_vs_eager"),
            r.get("y_has_nan"), r.get("eager_has_nan"),
            r.get("dx_vs_eager"), r.get("dalpha_vs_eager"),
            r.get("error") or r.get("bwd"),
        ])
    lines.append(table(
        ["case", "alpha", "liger/fp64", "eager/fp64", "liger/eager",
         "nan_l", "nan_e", "dx_vs_e", "da_vs_e", "note"],
        erows,
    ))

    lines += ["", "## 3. Autotune cold vs warm", ""]
    arows = []
    for r in data["autotune"]["cold_vs_warm"]:
        arows.append([
            r["H"], r["cold_ms"], r["warm_ms"], r["bench_fwd_ms"], r["cold_over_warm"],
        ])
    lines.append(table(["H", "cold_ms", "warm_ms", "bench_ms", "cold/warm"], arows))

    lines += ["", "### DA buffer vs BLOCK_N", ""]
    drows = []
    for r in data["autotune"]["da_shape"]:
        bc = r.get("best_config") or {}
        drows.append([
            r["H"], r["NUM_SMS"], r["cdiv_H_512"], r.get("programs_x"),
            bc.get("BLOCK_N") if isinstance(bc, dict) else None,
            r.get("da_slot_waste_frac"),
            r.get("dalpha_vs_eager"), r.get("dgamma_vs_eager"), r.get("dx_vs_eager"),
        ])
    lines.append(table(
        ["H", "SMs", "da_cols", "prog_x", "BLOCK_N", "waste", "da_vs_e", "dg_vs_e", "dx_vs_e"],
        drows,
    ))

    lines += ["", "### Determinism / no-beta", ""]
    for r in data["autotune"]["determinism"]:
        lines.append(
            f"- repeats={r['repeats']}: dx_eq={r['dx_all_equal']} "
            f"dalpha_eq={r['dalpha_all_equal']} dgamma_eq={r['dgamma_all_equal']} "
            f"dbeta_eq={r['dbeta_all_equal']} "
            f"dalpha_max_pair_rel={fmt(r['dalpha_max_pairwise_rel'])}"
        )
    for r in data["autotune"]["no_beta"]:
        lines.append(f"- no_beta: {r}")

    lines += ["", "## 4. Layouts", ""]
    lyrows = []
    for r in data["layouts"]["rows"]:
        lyrows.append([
            r["case"], r["x_contiguous"], r.get("y_vs_eager"), r.get("dx_vs_eager"),
            r.get("canary_unchanged"), r.get("module_vs_fn_y"), r.get("error"),
        ])
    lines.append(table(
        ["case", "contig", "y_vs_e", "dx_vs_e", "canary_ok", "mod_y", "err"],
        lyrows,
    ))

    lines += ["", "## 5. Memory / latency", ""]
    mrows = []
    for r in data["memory"]["rows"]:
        lig = r["liger"]
        eg = r["eager_dyt"]
        ln = r["F_layer_norm"]
        tc = r.get("torch_compile", {})
        mrows.append([
            r["label"], r["T"], r["H"],
            lig.get("bench_fwd_ms"), eg.get("bench_fwd_ms"), ln.get("bench_fwd_ms"),
            tc.get("bench_fwd_ms") if isinstance(tc, dict) else None,
            lig.get("bench_fwd_bwd_ms"), eg.get("bench_fwd_bwd_ms"), ln.get("bench_fwd_bwd_ms"),
            lig.get("held_after_fwd_mib"), lig.get("peak_mib"),
        ])
    lines.append(table(
        ["label", "T", "H", "liger_fwd", "eager_fwd", "ln_fwd", "cmp_fwd",
         "liger_fb", "eager_fb", "ln_fb", "held_MiB", "peak_MiB"],
        mrows,
    ))

    lines += ["", "## 6. Bandwidth (bf16, T=4096)", ""]
    brows = []
    for r in data["bandwidth"]["rows"]:
        lig = r.get("liger", {})
        eg = r.get("eager_dyt", {})
        ln = r.get("F_layer_norm", {})
        tc = r.get("torch_compile", {})
        brows.append([
            r["H"],
            lig.get("fwd_gbs"), eg.get("fwd_gbs"), ln.get("fwd_gbs"),
            tc.get("fwd_gbs") if isinstance(tc, dict) else None,
            lig.get("fwd_ms"), eg.get("fwd_ms"), ln.get("fwd_ms"),
            lig.get("fwd_bwd_ms"), eg.get("fwd_bwd_ms"),
        ])
    lines.append(table(
        ["H", "liger_GB/s", "eager_GB/s", "ln_GB/s", "cmp_GB/s",
         "liger_ms", "eager_ms", "ln_ms", "liger_fb", "eager_fb"],
        brows,
    ))
    lines.append("")
    results_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    global _PEAK_MIB
    t0 = time.perf_counter()
    print("GPU", GPU, flush=True)
    data = {
        "gpu": GPU,
        "seed": SEED,
        "numerics": sec_numerics(),
        "edges": sec_edges(),
        "autotune": sec_autotune(),
        "layouts": sec_layouts(),
        "memory": sec_memory(),
        "bandwidth": sec_bandwidth(),
    }
    data["wall_s"] = time.perf_counter() - t0
    data["peak_mib"] = _PEAK_MIB
    write_results(data)
    print(f"done in {data['wall_s']:.1f}s peak={_PEAK_MIB} MiB", flush=True)


if __name__ == "__main__":
    main()
