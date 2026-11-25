"""Liger-Kernel Softmax (`LigerSoftmaxFunction`).

`_softmax_forward` picks single-block when `n_cols <= BLOCK_SIZE` else
multi-block online max/sum, with `BLOCK_SIZE, num_warps = calculate_settings(n)`.
`calculate_settings` sets `BLOCK_SIZE = next_power_of_2(n)` and raises above
65536, so the public API never reaches multi-block: either single-block or
RuntimeError. Backward saves `y` and applies `y * (dy - sum(dy*y))`. Distinct
from 0010 (triton tutorial fused softmax forward bandwidth only).

1. Numerics vs fp64 + F.softmax / torch.compile for fp32 / fp16 / bf16.
2. Edges: causal -inf mask, all -inf, NaN / Inf, large logits, all-equal rows.
3. Limits: V = 65536 / 65537; prove multi_block_launch is always False through
   the public path; force multi-block with a capped BLOCK_SIZE and check
   correctness vs single-block / F.softmax.
4. Layouts: row-strided / transposed / 3D; Module wrapper.
5. Memory / latency / bandwidth at attention (S) and vocab-ish row widths.

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
import torch.nn.functional as F  # noqa: E402
from triton.testing import do_bench  # noqa: E402
from liger_kernel.ops.softmax import (  # noqa: E402
    LigerSoftmaxFunction,
    _softmax_backward,
    _softmax_forward,
    _softmax_multi_block_backward_kernel,
    _softmax_multi_block_forward_kernel,
    _softmax_single_block_backward_kernel,
    _softmax_single_block_forward_kernel,
)
from liger_kernel.ops.utils import calculate_settings  # noqa: E402
from liger_kernel.transformers.softmax import LigerSoftmax  # noqa: E402

SEED = 0
DEV = "cuda"
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
NUM_T, NUM_V = 2048, 4096
ATTN_SHAPES = [
    ("attn S=512", 512, 512),
    ("attn S=2048", 256, 2048),
    ("attn S=8192", 128, 8192),
    ("attn S=16384", 64, 16384),
]
VOCAB_SHAPES = [
    ("vocab 32k", 256, 32000),
    ("vocab 64k-1", 64, 65535),
    ("vocab 64k", 64, 65536),
]
BW_T = 4096
BW_VS = [512, 1024, 2048, 2049, 4096, 8192, 16384, 32768, 65536]
BW_ROUNDS = 3
FORCE_BLOCK = 1024  # force multi-block when V > FORCE_BLOCK

torch._dynamo.config.recompile_limit = 64
torch._functorch.config.donated_buffer = False


def liger(x):
    return LigerSoftmaxFunction.apply(x)


def fp64_softmax(x):
    x64 = x.to(torch.float64)
    m = x64.max(dim=-1, keepdim=True).values
    e = torch.exp(x64 - m)
    return e / e.sum(dim=-1, keepdim=True)


def eager_softmax(x):
    return F.softmax(x, dim=-1)


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
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(c) for c in r) + " |")
    return "\n".join(out)


_PEAK_MIB = 0.0


def cleanup():
    global _PEAK_MIB
    _PEAK_MIB = max(_PEAK_MIB, gpu_guard.peak_memory_mib())
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def make_inputs(T, V, dtype, seed=SEED, scale=1.0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = scale * torch.randn(T, V, device=DEV, dtype=dtype, generator=g)
    dy = 0.1 * torch.randn(T, V, device=DEV, dtype=dtype, generator=g)
    return x, dy


def run_variant(fn, x, dy):
    xx = x.detach().clone().requires_grad_(True)
    y = fn(xx)
    torch.autograd.backward(y, dy)
    return y.detach(), xx.grad.detach()


def force_multi_block_forward(x, block_size=FORCE_BLOCK, num_warps=8):
    """Bypass calculate_settings and launch the multi-block kernel."""
    *batch, n_cols = x.shape
    x2d = x.contiguous().view(-1, n_cols)
    n_rows = x2d.shape[0]
    y2d = torch.empty_like(x2d)
    _softmax_multi_block_forward_kernel[(n_rows,)](
        y2d, y2d.stride(0), x2d, x2d.stride(0), n_cols,
        BLOCK_SIZE=block_size, num_warps=num_warps,
    )
    return y2d.view(*batch, n_cols)


def force_multi_block_backward(dy, y, block_size=FORCE_BLOCK, num_warps=8):
    *batch, n_cols = dy.shape
    dy2d = dy.contiguous().view(-1, n_cols)
    y2d = y.contiguous().view(-1, n_cols)
    n_rows = dy2d.shape[0]
    dx2d = torch.empty_like(dy2d)
    _softmax_multi_block_backward_kernel[(n_rows,)](
        dy2d, dy2d.stride(0), y2d, y2d.stride(0), dx2d, dx2d.stride(0), n_cols,
        BLOCK_SIZE=block_size, num_warps=num_warps,
    )
    return dx2d.view(*batch, n_cols)


# ---------------------------------------------------------------- 1. numerics
def sec_numerics():
    rows = []
    for dname, dt in DTYPES.items():
        for scale in (1.0, 10.0):
            x, dy = make_inputs(NUM_T, NUM_V, dt, scale=scale)
            y_ref = fp64_softmax(x)
            xd = x.double().requires_grad_(True)
            yd = fp64_softmax(xd)
            torch.autograd.backward(yd, dy.double())
            floor_y = y_ref.to(dt)
            floor_dx = xd.grad.to(dt)

            def pack(y, dx):
                return {
                    "y_rel": rel(y, y_ref),
                    "dx_rel": rel(dx, xd.grad),
                    "y_vs_floor": rel(y, floor_y),
                    "dx_vs_floor": rel(dx, floor_dx),
                    "y_sum_err": float((y.double().sum(-1) - 1.0).abs().max()),
                }

            variants = {}
            yl, dxl = run_variant(liger, x, dy)
            variants["liger"] = pack(yl, dxl)
            yf, dxf = run_variant(eager_softmax, x, dy)
            variants["F_softmax"] = pack(yf, dxf)

            compiled = torch.compile(lambda a: F.softmax(a, dim=-1), fullgraph=False)
            try:
                yc, dxc = run_variant(compiled, x, dy)
                variants["torch_compile"] = pack(yc, dxc)
            except Exception as e:
                variants["torch_compile"] = {"error": f"{type(e).__name__}: {e}"}

            # low-level forward path flags
            y_ll, bs, nw, multi = _softmax_forward(x.contiguous())
            variants["path"] = {
                "BLOCK_SIZE": bs,
                "num_warps": nw,
                "multi_block_launch": multi,
                "y_vs_liger": rel(y_ll, yl),
            }

            row = {
                "dtype": dname,
                "scale": scale,
                "floor": {"y": rel(floor_y, y_ref), "dx": rel(floor_dx, xd.grad)},
                "liger_vs_F": {"y": rel(yl, yf), "dx": rel(dxl, dxf)},
                **variants,
            }
            rows.append(row)
            print(
                "numerics", dname, f"scale={scale}",
                {k: (variants[k].get("y_rel") if isinstance(variants[k], dict) else variants[k])
                 for k in ("liger", "F_softmax", "torch_compile")},
                "multi=", multi, flush=True,
            )
            cleanup()
    return {"rows": rows}


# ---------------------------------------------------------------- 2. edges
def sec_edges():
    rows = []
    T, V = 64, 2048
    dtype = torch.bfloat16
    g = torch.Generator(device=DEV).manual_seed(SEED + 20)

    cases = {}
    base = torch.randn(T, V, device=DEV, dtype=dtype, generator=g)
    cases["normal"] = base.clone()

    causal = base.clone()
    # upper triangle -inf (mask future), last dim is key position
    idx = torch.arange(V, device=DEV)
    # For a square-ish mask on first min(T,V) — use a per-row prefix keep
    keep = V // 2
    causal[:, keep:] = float("-inf")
    cases["half_row_neg_inf"] = causal

    all_inf = base.clone()
    all_inf[:] = float("-inf")
    cases["all_neg_inf"] = all_inf

    with_nan = base.clone()
    with_nan[0, 0] = float("nan")
    with_nan[1, 1] = float("inf")
    cases["nan_inf_cells"] = with_nan

    large = 1000.0 * base
    cases["large_logits_1e3"] = large

    equal = torch.zeros(T, V, device=DEV, dtype=dtype)
    cases["all_equal_zero"] = equal

    one_hotish = torch.full((T, V), -50.0, device=DEV, dtype=dtype)
    one_hotish[:, 0] = 50.0
    cases["one_hotish"] = one_hotish

    for name, x in cases.items():
        cell = {"case": name}
        try:
            y_ref = fp64_softmax(x)
            yf = eager_softmax(x)
            yl = liger(x)
            cell["y_liger_vs_fp64"] = rel(yl, y_ref)
            cell["y_F_vs_fp64"] = rel(yf, y_ref)
            cell["y_liger_vs_F"] = rel(yl, yf)
            cell["y_has_nan"] = bool(torch.isnan(yl).any())
            cell["F_has_nan"] = bool(torch.isnan(yf).any())
            cell["y_sum_max_abs_err"] = float((yl.double().sum(-1) - 1.0).abs().max())
            dy = 0.1 * torch.randn_like(x)
            xl = x.detach().clone().requires_grad_(True)
            xf = x.detach().clone().requires_grad_(True)
            yl2 = liger(xl)
            yf2 = eager_softmax(xf)
            if torch.isnan(yl2).any() or torch.isnan(yf2).any():
                cell["bwd"] = "skipped_nan_y"
            else:
                torch.autograd.backward(yl2, dy)
                torch.autograd.backward(yf2, dy)
                cell["dx_liger_vs_F"] = rel(xl.grad, xf.grad)
                if name == "half_row_neg_inf":
                    cell["dx_masked_abs_max"] = float(xl.grad[:, keep:].double().abs().max())
                    cell["dx_F_masked_abs_max"] = float(xf.grad[:, keep:].double().abs().max())
        except Exception as e:
            cell["error"] = f"{type(e).__name__}: {e}"
        rows.append(cell)
        print("edges", name, {k: cell[k] for k in cell if k != "case"}, flush=True)
        cleanup()
    return {"rows": rows}


# ---------------------------------------------------------------- 3. limits + dead multi-block
def sec_limits():
    out = {"public_path": [], "forced_multi": [], "limit": []}

    # Public path: multi_block_launch must be False for every legal V
    for V in [1, 2, 3, 63, 64, 127, 128, 1023, 1024, 2049, 4096, 8191, 8192,
              16384, 32768, 65535, 65536]:
        try:
            bs, nw = calculate_settings(V)
            x = torch.randn(4, V, device=DEV, dtype=torch.bfloat16)
            y, bsr, nwr, multi = _softmax_forward(x)
            cell = {
                "V": V,
                "BLOCK_SIZE": bs,
                "num_warps": nw,
                "multi_block_launch": multi,
                "next_pow2_ge_V": bs >= V,
                "y_sum_err": float((y.double().sum(-1) - 1.0).abs().max()),
            }
        except Exception as e:
            cell = {"V": V, "error": f"{type(e).__name__}: {e}"}
        out["public_path"].append(cell)
        print("public_path", cell, flush=True)
        cleanup()

    # 65537 must raise
    for V in [65537, 65536 + 1, 128256]:
        cell = {"V": V}
        try:
            calculate_settings(V)
            cell["calculate_settings"] = "no_raise"
        except Exception as e:
            cell["calculate_settings"] = f"{type(e).__name__}: {e}"
        try:
            x = torch.randn(2, V, device=DEV, dtype=torch.bfloat16)
            liger(x)
            cell["liger"] = "no_raise"
        except Exception as e:
            cell["liger"] = f"{type(e).__name__}: {e}"
        out["limit"].append(cell)
        print("limit", cell, flush=True)
        cleanup()

    # Force multi-block vs public single-block vs F.softmax.
    # On current triton-windows the multi-block kernels may fail to compile
    # (tl.float32(-inf) TypeError) — record that as a finding.
    for V in [2048, 4096, 8192, 16384]:
        assert V > FORCE_BLOCK
        x, dy = make_inputs(128, V, torch.bfloat16, seed=SEED + V)
        y_pub, bs, nw, multi = _softmax_forward(x)
        dx_pub = _softmax_backward(dy, y_pub, bs, nw, multi)
        y_torch = eager_softmax(x)
        xt = x.detach().clone().requires_grad_(True)
        yt = eager_softmax(xt)
        torch.autograd.backward(yt, dy)

        cell = {
            "V": V,
            "public_multi": multi,
            "public_BLOCK": bs,
            "force_BLOCK": FORCE_BLOCK,
        }
        try:
            y_f = force_multi_block_forward(x, FORCE_BLOCK, 8)
            dx_f = force_multi_block_backward(dy, y_f, FORCE_BLOCK, 8)

            def bench_single():
                _softmax_single_block_forward_kernel[(x.shape[0],)](
                    torch.empty_like(x), x.stride(0), x, x.stride(0), V,
                    BLOCK_SIZE=bs, num_warps=nw,
                )

            def bench_multi():
                force_multi_block_forward(x, FORCE_BLOCK, 8)

            t_single = do_bench(bench_single, warmup=5, rep=20)
            t_multi = do_bench(bench_multi, warmup=5, rep=20)
            cell.update({
                "y_force_vs_public": rel(y_f, y_pub),
                "y_force_vs_F": rel(y_f, y_torch),
                "dx_force_vs_public": rel(dx_f, dx_pub),
                "dx_force_vs_F": rel(dx_f, xt.grad),
                "fwd_ms_single": t_single,
                "fwd_ms_forced_multi": t_multi,
                "multi_vs_single": t_multi / t_single if t_single > 0 else None,
            })
        except Exception as e:
            # Capture first line of triton CompilationError body
            msg = f"{type(e).__name__}: {e}"
            cell["force_error"] = msg.splitlines()[0][:200]
            cell["force_error_detail"] = "\n".join(msg.splitlines()[:12])[:800]
        out["forced_multi"].append(cell)
        print("forced_multi", cell, flush=True)
        cleanup()
    return out


# ---------------------------------------------------------------- 4. layouts
def sec_layouts():
    rows = []
    T, V = 1024, 2048
    dtype = torch.bfloat16
    g = torch.Generator(device=DEV).manual_seed(SEED + 30)
    base = torch.randn(T, V + 64, device=DEV, dtype=dtype, generator=g)
    dy_c = torch.randn(T, V, device=DEV, dtype=dtype, generator=g)

    cases = []
    x0 = base[:, :V].contiguous()
    cases.append(("contiguous", x0, dy_c))
    x_row = base[:, :V]
    assert not x_row.is_contiguous()
    cases.append(("row_strided_x", x_row, dy_c))
    x_t = torch.randn(V, T, device=DEV, dtype=dtype, generator=g).t()
    cases.append(("transposed_x", x_t, dy_c))
    B, S = 4, 256
    x3 = torch.randn(B, S, V, device=DEV, dtype=dtype, generator=g)
    dy3 = torch.randn(B, S, V, device=DEV, dtype=dtype, generator=g)
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
                canary = base[:, V:V + 8].clone()
            xl = x.detach().clone().requires_grad_(True)
            xf = x.contiguous().detach().clone().requires_grad_(True)
            yl = liger(xl)
            yf = eager_softmax(xf)
            torch.autograd.backward(yl, dy)
            torch.autograd.backward(yf, dy.contiguous())
            cell["y_vs_F"] = rel(yl, yf)
            cell["dx_vs_F"] = rel(xl.grad, xf.grad)
            if canary is not None:
                cell["canary_unchanged"] = bool(torch.equal(base[:, V:V + 8], canary))
            # Module
            if name == "contiguous":
                mod = LigerSoftmax().to(DEV)
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
    for label, T, V in ATTN_SHAPES + VOCAB_SHAPES:
        cell = {"label": label, "T": T, "V": V}
        try:
            # skip illegal
            calculate_settings(V)
        except Exception as e:
            cell["error"] = f"{type(e).__name__}: {e}"
            rows.append(cell)
            print("memory", cell, flush=True)
            cleanup()
            continue

        x, dy = make_inputs(T, V, dtype, seed=SEED + T + V)
        bytes_in = T * V * 2

        def measure(fn, name):
            cleanup()
            torch.cuda.reset_peak_memory_stats()
            # held after forward
            xx = x.detach().clone().requires_grad_(True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            y = fn(xx)
            torch.cuda.synchronize()
            fwd_ms = (time.perf_counter() - t0) * 1e3
            held = gpu_guard.peak_memory_mib()
            # also track allocated after fwd
            alloc_fwd = torch.cuda.memory_allocated() / 2**20
            torch.autograd.backward(y, dy)
            torch.cuda.synchronize()
            peak = gpu_guard.peak_memory_mib()
            # latency with do_bench (fwd only, no_grad)
            def fwd_only():
                with torch.no_grad():
                    fn(x)

            def fwd_bwd():
                xx2 = x.detach().clone().requires_grad_(True)
                yy = fn(xx2)
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
        cell["F_softmax"] = measure(eager_softmax, "F")
        try:
            compiled = torch.compile(lambda a: F.softmax(a, dim=-1), fullgraph=False)
            # warm compile
            _ = compiled(x.detach().clone().requires_grad_(True))
            torch.cuda.synchronize()
            cell["torch_compile"] = measure(compiled, "compile")
        except Exception as e:
            cell["torch_compile"] = {"error": f"{type(e).__name__}: {e}"}

        # path info
        _, bs, nw, multi = _softmax_forward(x.contiguous())
        cell["path"] = {"BLOCK_SIZE": bs, "num_warps": nw, "multi_block_launch": multi,
                        "input_mib": round(bytes_in / 2**20, 3)}
        rows.append(cell)
        print(
            "memory", label,
            "liger_fwd", cell["liger"].get("bench_fwd_ms"),
            "F_fwd", cell["F_softmax"].get("bench_fwd_ms"),
            "multi", multi, flush=True,
        )
        cleanup()
    return {"rows": rows}


# ---------------------------------------------------------------- 6. bandwidth
def sec_bandwidth():
    rows = []
    dtype = torch.bfloat16
    for V in BW_VS:
        cell = {"V": V, "T": BW_T}
        try:
            bs, nw = calculate_settings(V)
        except Exception as e:
            cell["error"] = f"{type(e).__name__}: {e}"
            rows.append(cell)
            print("bw", cell, flush=True)
            cleanup()
            continue

        x = torch.randn(BW_T, V, device=DEV, dtype=dtype)
        dy = torch.randn(BW_T, V, device=DEV, dtype=dtype)
        # bytes: fwd reads x writes y = 2 * T * V * 2; bwd reads dy+y writes dx = 3 * T*V*2
        fwd_bytes = 2 * BW_T * V * 2
        bwd_bytes = 3 * BW_T * V * 2

        def gbs(ms, nbytes):
            if not isinstance(ms, (int, float)) or ms <= 0:
                return None
            return (nbytes / 1e9) / (ms / 1e3)

        def bench_fwd(fn):
            return do_bench(lambda: fn(x), warmup=5, rep=30)

        def bench_bwd(fn):
            def go():
                xx = x.detach().clone().requires_grad_(True)
                y = fn(xx)
                torch.autograd.backward(y, dy)
            return do_bench(go, warmup=3, rep=15)

        for name, fn in [("liger", liger), ("F_softmax", eager_softmax)]:
            try:
                ms_f = bench_fwd(fn)
                ms_b = bench_bwd(fn)
                cell[name] = {
                    "fwd_ms": ms_f,
                    "fwd_gbs": gbs(ms_f, fwd_bytes),
                    "fwd_bwd_ms": ms_b,
                    "bwd_approx_gbs": gbs(ms_b, fwd_bytes + bwd_bytes),
                }
            except Exception as e:
                cell[name] = {"error": f"{type(e).__name__}: {e}"}

        try:
            compiled = torch.compile(lambda a: F.softmax(a, dim=-1), fullgraph=False)
            _ = compiled(x)
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

        _, _, _, multi = _softmax_forward(x)
        cell["multi_block_launch"] = multi
        cell["BLOCK_SIZE"] = bs
        cell["lane_util"] = V / bs
        rows.append(cell)
        print(
            "bw", V,
            "liger", cell.get("liger", {}).get("fwd_gbs"),
            "F", cell.get("F_softmax", {}).get("fwd_gbs"),
            flush=True,
        )
        cleanup()
    return {"rows": rows}


def write_results(data):
    results_json = HERE / "results.json"
    results_md = HERE / "results.md"
    results_json.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    lines = [
        "# 0032 — Liger Softmax results",
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
            r["dtype"], r["scale"],
            lig["y_rel"], lig["dx_rel"], lig["y_vs_floor"], lig["dx_vs_floor"],
            r["liger_vs_F"]["y"], r["liger_vs_F"]["dx"],
            r["path"]["multi_block_launch"], r["path"]["BLOCK_SIZE"],
            lig["y_sum_err"],
        ])
    lines.append(table(
        ["dtype", "scale", "y_rel", "dx_rel", "y_floor", "dx_floor",
         "y_vs_F", "dx_vs_F", "multi", "BLOCK", "sum_err"],
        num_rows,
    ))

    lines += ["", "## 2. Edges", ""]
    erows = []
    for r in data["edges"]["rows"]:
        erows.append([
            r["case"],
            r.get("y_liger_vs_fp64"), r.get("y_F_vs_fp64"), r.get("y_liger_vs_F"),
            r.get("y_has_nan"), r.get("F_has_nan"),
            r.get("dx_liger_vs_F"), r.get("dx_masked_abs_max"),
            r.get("y_sum_max_abs_err"), r.get("error") or r.get("bwd"),
        ])
    lines.append(table(
        ["case", "liger/fp64", "F/fp64", "liger/F", "nan_l", "nan_F",
         "dx_vs_F", "dx_mask", "sum_err", "note"],
        erows,
    ))

    lines += ["", "## 3. Public path (multi_block_launch)", ""]
    prows = []
    for r in data["limits"]["public_path"]:
        prows.append([
            r.get("V"), r.get("BLOCK_SIZE"), r.get("num_warps"),
            r.get("multi_block_launch"), r.get("next_pow2_ge_V"),
            r.get("y_sum_err"), r.get("error"),
        ])
    lines.append(table(
        ["V", "BLOCK", "warps", "multi", "pow2>=V", "sum_err", "err"],
        prows,
    ))

    lines += ["", "### Over-limit", ""]
    lrows = [[r["V"], r.get("calculate_settings"), r.get("liger")] for r in data["limits"]["limit"]]
    lines.append(table(["V", "calculate_settings", "liger"], lrows))

    lines += ["", "### Forced multi-block vs public single-block", ""]
    frows = []
    for r in data["limits"]["forced_multi"]:
        frows.append([
            r["V"], r["public_BLOCK"], r["force_BLOCK"],
            r.get("y_force_vs_public"), r.get("y_force_vs_F"),
            r.get("dx_force_vs_public"), r.get("dx_force_vs_F"),
            r.get("fwd_ms_single"), r.get("fwd_ms_forced_multi"),
            r.get("multi_vs_single"), r.get("force_error"),
        ])
    lines.append(table(
        ["V", "pub_B", "force_B", "y_vs_pub", "y_vs_F", "dx_vs_pub", "dx_vs_F",
         "ms_single", "ms_multi", "multi/single", "err"],
        frows,
    ))

    lines += ["", "## 4. Layouts", ""]
    lyrows = []
    for r in data["layouts"]["rows"]:
        lyrows.append([
            r["case"], r["x_contiguous"], r.get("y_vs_F"), r.get("dx_vs_F"),
            r.get("canary_unchanged"), r.get("module_vs_fn_y"), r.get("error"),
        ])
    lines.append(table(
        ["case", "contig", "y_vs_F", "dx_vs_F", "canary_ok", "mod_y", "err"],
        lyrows,
    ))

    lines += ["", "## 5. Memory / latency", ""]
    mrows = []
    for r in data["memory"]["rows"]:
        if "error" in r and "liger" not in r:
            mrows.append([r["label"], r["T"], r["V"], r["error"], "-", "-", "-", "-", "-"])
            continue
        lig, fs = r["liger"], r["F_softmax"]
        tc = r.get("torch_compile", {})
        mrows.append([
            r["label"], r["T"], r["V"],
            lig.get("bench_fwd_ms"), fs.get("bench_fwd_ms"),
            tc.get("bench_fwd_ms") if isinstance(tc, dict) else None,
            lig.get("bench_fwd_bwd_ms"), fs.get("bench_fwd_bwd_ms"),
            lig.get("held_after_fwd_mib"), lig.get("peak_mib"),
            r.get("path", {}).get("multi_block_launch"),
        ])
    lines.append(table(
        ["label", "T", "V", "liger_fwd", "F_fwd", "cmp_fwd",
         "liger_fb", "F_fb", "held_MiB", "peak_MiB", "multi"],
        mrows,
    ))

    lines += ["", "## 6. Bandwidth (bf16, T=4096)", ""]
    brows = []
    for r in data["bandwidth"]["rows"]:
        if "error" in r and "liger" not in r:
            brows.append([r["V"], r["error"], "-", "-", "-", "-", "-"])
            continue
        lig = r.get("liger", {})
        fs = r.get("F_softmax", {})
        tc = r.get("torch_compile", {})
        brows.append([
            r["V"], r.get("BLOCK_SIZE"), r.get("lane_util"),
            lig.get("fwd_gbs"), fs.get("fwd_gbs"),
            tc.get("fwd_gbs") if isinstance(tc, dict) else None,
            lig.get("fwd_ms"), fs.get("fwd_ms"),
            r.get("multi_block_launch"),
        ])
    lines.append(table(
        ["V", "BLOCK", "util", "liger_GB/s", "F_GB/s", "cmp_GB/s",
         "liger_ms", "F_ms", "multi"],
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
        "limits": sec_limits(),
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
