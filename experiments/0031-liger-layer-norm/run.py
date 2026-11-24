"""Liger-Kernel LayerNorm (`LigerLayerNormFunction`).

`layer_norm_forward` launches one Triton program per row with
BLOCK_SIZE = next_power_of_2(H) (hard-fails above 65536), computes mean /
rstd in fp32, fuses the affine, and saves X / W / B / Mean / RSTD.
Backward partitions rows across SMs (`rows_per_program = ceil(T / sm)`),
accumulates fp32 partial dW / db per SM into `torch.empty` buffers, then
`.sum(0)`. `@ensure_contiguous` copies every tensor arg on both forward and
backward. Unlike Liger RMSNorm there is no in-place dY→dX path and no casting
modes; bias is required by the Triton kernel (functional invents zeros when
None). Distinct from 0020 (tutorial layernorm stride / lock bugs) and 0022
(RMSNorm casting / in-place).

1. Numerics vs fp64 textbook + F.layer_norm / nn.LayerNorm / torch.compile
   for fp32 / fp16 / bf16, with and without a real bias.
2. Layout: row-strided / transposed / non-contiguous weight+bias / stride-0
   dy from y.sum() — does ensure_contiguous save us from the tutorial bugs?
3. Limits: H = 65536 compiles, H = 65537 raises; needs_dw / needs_db skips;
   NaN / Inf rows; Module bias=False still carries a zero Parameter.
4. SM-partitioned dW/db: bitwise determinism over repeated backward; wrong
   grads when T << SM count (empty partial slots).
5. Memory / latency / bandwidth at GPT-2 / BERT / Llama-ish H vs eager /
   F.layer_norm / torch.compile.

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
from liger_kernel.ops.layer_norm import (  # noqa: E402
    LigerLayerNormFunction,
    layer_norm_backward,
    layer_norm_forward,
)
from liger_kernel.ops.utils import calculate_settings  # noqa: E402
from liger_kernel.transformers.layer_norm import LigerLayerNorm  # noqa: E402

SEED = 0
DEV = "cuda"
EPS = 1e-6
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
NUM_T, NUM_H = 2048, 4096
MEM_SHAPES = [
    ("GPT-2-124M", 768),
    ("BERT-base", 768),
    ("Llama-3.2-1B", 2048),
    ("Qwen3-0.6B", 1024),
]
MEM_TOKENS = [2048, 8192, 16384]
BW_T = 4096
BW_HS = [512, 768, 1024, 2048, 2049, 4096, 8192, 11008, 16384]
BW_ROUNDS = 3
DET_REPEATS = 8

torch._dynamo.config.recompile_limit = 64
torch._functorch.config.donated_buffer = False


def liger(x, w, b, eps=EPS):
    return LigerLayerNormFunction.apply(x, w, b, eps)


def fp64_ln(x, w, b, eps=EPS):
    x64 = x.to(torch.float64)
    w64 = w.to(torch.float64)
    mean = x64.mean(-1, keepdim=True)
    var = ((x64 - mean) ** 2).mean(-1, keepdim=True)
    y = (x64 - mean) * torch.rsqrt(var + eps)
    y = y * w64
    if b is not None:
        y = y + b.to(torch.float64)
    return y


def eager_ln(x, w, b, eps=EPS):
    """Match F.layer_norm elementwise affine in the input dtype path torch uses."""
    return F.layer_norm(x, (x.shape[-1],), w, b, eps)


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


def make_inputs(T, H, dtype, with_bias=True, seed=SEED):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(T, H, device=DEV, dtype=dtype, generator=g)
    w = torch.randn(H, device=DEV, dtype=dtype, generator=g)
    b = torch.randn(H, device=DEV, dtype=dtype, generator=g) if with_bias else torch.zeros(
        H, device=DEV, dtype=dtype
    )
    dy = 0.1 * torch.randn(T, H, device=DEV, dtype=dtype, generator=g)
    return x, w, b, dy


def run_variant(fn, x, w, b, dy):
    xx = x.detach().clone().requires_grad_(True)
    ww = w.detach().clone().requires_grad_(True)
    bb = b.detach().clone().requires_grad_(True)
    y = fn(xx, ww, bb)
    torch.autograd.backward(y, dy)
    return y.detach(), xx.grad.detach(), ww.grad.detach(), bb.grad.detach()


# ---------------------------------------------------------------- 1. numerics
def sec_numerics():
    rows = []
    for dname, dt in DTYPES.items():
        for with_bias in (True, False):
            x, w, b, dy = make_inputs(NUM_T, NUM_H, dt, with_bias=with_bias)
            y_ref = fp64_ln(x, w, b if with_bias else None)
            xd = x.double().requires_grad_(True)
            wd = w.double().requires_grad_(True)
            bd = b.double().requires_grad_(True) if with_bias else None
            yd = fp64_ln(xd, wd, bd)
            torch.autograd.backward(yd, dy.double())
            floor_y = y_ref.to(dt)
            floor_dx = xd.grad.to(dt)
            floor_dw = wd.grad.to(dt)
            floor_db = bd.grad.to(dt) if with_bias else None

            def pack(y, dx, dw, db):
                out = {
                    "y_rel": rel(y, y_ref),
                    "dx_rel": rel(dx, xd.grad),
                    "dw_rel": rel(dw, wd.grad),
                    "y_vs_floor": rel(y, floor_y),
                    "dx_vs_floor": rel(dx, floor_dx),
                    "dw_vs_floor": rel(dw, floor_dw),
                }
                if with_bias:
                    out["db_rel"] = rel(db, bd.grad)
                    out["db_vs_floor"] = rel(db, floor_db)
                else:
                    out["db_rel"] = None
                    out["db_vs_floor"] = None
                    out["db_abs_max"] = float(db.double().abs().max())
                return out

            variants = {}
            # Liger
            yl, dxl, dwl, dbl = run_variant(liger, x, w, b, dy)
            variants["liger"] = pack(yl, dxl, dwl, dbl)

            # F.layer_norm
            yf, dxf, dwf, dbf = run_variant(eager_ln, x, w, b, dy)
            variants["F_layer_norm"] = pack(yf, dxf, dwf, dbf)

            # nn.LayerNorm module (elementwise_affine=True always here)
            mod = nn.LayerNorm(NUM_H, eps=EPS, elementwise_affine=True, bias=with_bias).to(DEV, dt)
            with torch.no_grad():
                mod.weight.copy_(w)
                if with_bias:
                    mod.bias.copy_(b)
            xm = x.clone().requires_grad_(True)
            ym = mod(xm)
            torch.autograd.backward(ym, dy)
            variants["nn_LayerNorm"] = pack(
                ym.detach(),
                xm.grad,
                mod.weight.grad,
                mod.bias.grad if with_bias else torch.zeros_like(w),
            )

            # torch.compile of F.layer_norm
            def _closed(a, ww, bb):
                return F.layer_norm(a, (a.shape[-1],), ww, bb, EPS)

            compiled = torch.compile(_closed, fullgraph=False)
            try:
                yc, dxc, dwc, dbc = run_variant(compiled, x, w, b, dy)
                variants["torch_compile"] = pack(yc, dxc, dwc, dbc)
            except Exception as e:
                variants["torch_compile"] = {"error": f"{type(e).__name__}: {e}"}

            row = {
                "dtype": dname,
                "with_bias": with_bias,
                "floor": {
                    "y": rel(floor_y, y_ref),
                    "dx": rel(floor_dx, xd.grad),
                    "dw": rel(floor_dw, wd.grad),
                    "db": rel(floor_db, bd.grad) if with_bias else None,
                },
                "liger_vs_F": {
                    "y": rel(yl, yf),
                    "dx": rel(dxl, dxf),
                    "dw": rel(dwl, dwf),
                    "db": rel(dbl, dbf) if with_bias else float(dbl.double().abs().max()),
                },
                **{k: v for k, v in variants.items()},
            }
            rows.append(row)
            print(
                "numerics",
                dname,
                "bias" if with_bias else "no_bias",
                {k: variants[k].get("y_rel") if isinstance(variants[k], dict) else variants[k]
                 for k in variants},
                flush=True,
            )
            cleanup()
    return {"rows": rows}


# ---------------------------------------------------------------- 2. layouts
def sec_layouts():
    """Strided / transposed inputs that broke the Triton tutorial layernorm."""
    rows = []
    T, H = 1024, 2048
    dtype = torch.bfloat16
    g = torch.Generator(device=DEV).manual_seed(SEED + 10)
    base = torch.randn(T, H + 64, device=DEV, dtype=dtype, generator=g)
    w = torch.randn(H, device=DEV, dtype=dtype, generator=g)
    b = torch.randn(H, device=DEV, dtype=dtype, generator=g)
    dy_c = torch.randn(T, H, device=DEV, dtype=dtype, generator=g)

    cases = []

    # Contiguous baseline
    x0 = base[:, :H].contiguous()
    cases.append(("contiguous", x0, w, b, dy_c))

    # Row-strided: view into wider storage (stride(-1)=1 still, but row stride > H)
    x_row = base[:, :H]
    assert x_row.stride(0) == H + 64 and x_row.is_contiguous() is False
    cases.append(("row_strided_x", x_row, w, b, dy_c))

    # Transposed then view as (T, H) via .t() of (H, T)
    x_t = torch.randn(H, T, device=DEV, dtype=dtype, generator=g).t()
    assert x_t.shape == (T, H) and not x_t.is_contiguous()
    cases.append(("transposed_x", x_t, w.contiguous(), b.contiguous(), dy_c))

    # Non-contiguous weight / bias
    w_wide = torch.randn(H + 8, device=DEV, dtype=dtype, generator=g)
    b_wide = torch.randn(H + 8, device=DEV, dtype=dtype, generator=g)
    cases.append(("strided_weight_bias", x0, w_wide[:H], b_wide[:H], dy_c))

    # 3D input (B, S, H)
    B, S = 4, 256
    x3 = torch.randn(B, S, H, device=DEV, dtype=dtype, generator=g)
    dy3 = torch.randn(B, S, H, device=DEV, dtype=dtype, generator=g)
    cases.append(("input_3d", x3, w, b, dy3))

    for name, x, ww, bb, dy in cases:
        cell = {"case": name, "x_contiguous": bool(x.is_contiguous()),
                "x_stride": list(x.stride()), "shape": list(x.shape)}
        try:
            # Reference on contiguous clones
            xref = x.contiguous()
            y_ref = fp64_ln(xref, ww.contiguous(), bb.contiguous())
            xd = xref.double().requires_grad_(True)
            wd = ww.contiguous().double().requires_grad_(True)
            bd = bb.contiguous().double().requires_grad_(True)
            yd = fp64_ln(xd, wd, bd)
            torch.autograd.backward(yd, dy.contiguous().double())

            xl = x.detach().clone().requires_grad_(True)
            wl = ww.detach().clone().requires_grad_(True)
            bl = bb.detach().clone().requires_grad_(True)
            # Canary after a contiguous clone of x storage neighbour for row_strided
            canary = None
            if name == "row_strided_x":
                canary = base[:, H:H + 8].clone()

            yl = liger(xl, wl, bl)
            torch.autograd.backward(yl, dy)

            cell["y_rel"] = rel(yl, y_ref)
            cell["dx_rel"] = rel(xl.grad, xd.grad)
            cell["dw_rel"] = rel(wl.grad, wd.grad)
            cell["db_rel"] = rel(bl.grad, bd.grad)
            cell["y_finite"] = bool(torch.isfinite(yl).all())
            cell["dx_finite"] = bool(torch.isfinite(xl.grad).all())
            if canary is not None:
                cell["canary_unchanged"] = bool(torch.equal(base[:, H:H + 8], canary))
            # F.layer_norm on same layout
            xf = x.detach().clone().requires_grad_(True)
            wf = ww.detach().clone().requires_grad_(True)
            bf = bb.detach().clone().requires_grad_(True)
            yf = eager_ln(xf, wf, bf)
            torch.autograd.backward(yf, dy)
            cell["liger_vs_F_y"] = rel(yl, yf)
            cell["liger_vs_F_dx"] = rel(xl.grad, xf.grad)
        except Exception as e:
            cell["error"] = f"{type(e).__name__}: {e}"
        rows.append(cell)
        print("layout", cell, flush=True)
        cleanup()

    # stride-0 dy from y.sum() — tutorial wrote NaN grads
    x, w, b, _ = make_inputs(512, 1024, torch.bfloat16, seed=SEED + 11)
    cell = {"case": "stride0_dy_from_ysum"}
    try:
        xl = x.clone().requires_grad_(True)
        wl = w.clone().requires_grad_(True)
        bl = b.clone().requires_grad_(True)
        yl = liger(xl, wl, bl)
        yl.sum().backward()
        cell["dx_finite"] = bool(torch.isfinite(xl.grad).all())
        cell["dw_finite"] = bool(torch.isfinite(wl.grad).all())
        cell["db_finite"] = bool(torch.isfinite(bl.grad).all())
        cell["dx_abs_mean"] = float(xl.grad.double().abs().mean())
        xf = x.clone().requires_grad_(True)
        wf = w.clone().requires_grad_(True)
        bf = b.clone().requires_grad_(True)
        eager_ln(xf, wf, bf).sum().backward()
        cell["dx_rel_vs_F"] = rel(xl.grad, xf.grad)
        cell["dw_rel_vs_F"] = rel(wl.grad, wf.grad)
    except Exception as e:
        cell["error"] = f"{type(e).__name__}: {e}"
    rows.append(cell)
    print("layout", cell, flush=True)
    cleanup()
    return {"rows": rows}


# ---------------------------------------------------------------- 3. limits / edges
def sec_edges():
    rows = []

    # Hidden-dim hard limit
    for H, label in [(65536, "H_65536"), (65537, "H_65537"), (65535, "H_65535")]:
        cell = {"case": label, "H": H}
        try:
            block, warps = calculate_settings(H)
            cell["BLOCK_SIZE"] = block
            cell["num_warps"] = warps
        except Exception as e:
            cell["calculate_settings"] = f"{type(e).__name__}: {e}"
            rows.append(cell)
            print("edge", cell, flush=True)
            continue
        # Tiny T to stay in memory for H=65536 (bf16 row = 128 KiB)
        T = 4 if H >= 32768 else 64
        try:
            x = torch.randn(T, H, device=DEV, dtype=torch.bfloat16)
            w = torch.randn(H, device=DEV, dtype=torch.bfloat16)
            b = torch.zeros(H, device=DEV, dtype=torch.bfloat16)
            y = liger(x, w, b)
            cell["fwd_ok"] = True
            cell["y_finite"] = bool(torch.isfinite(y).all())
            xx = x.clone().requires_grad_(True)
            ww = w.clone().requires_grad_(True)
            bb = b.clone().requires_grad_(True)
            yy = liger(xx, ww, bb)
            yy.sum().backward()
            cell["bwd_ok"] = True
            cell["dx_finite"] = bool(torch.isfinite(xx.grad).all())
            cell["dw_finite"] = bool(torch.isfinite(ww.grad).all())
        except Exception as e:
            cell["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        rows.append(cell)
        print("edge", cell, flush=True)
        cleanup()

    # needs_dw / needs_db via frozen weight or bias
    x, w, b, dy = make_inputs(256, 1024, torch.bfloat16, seed=SEED + 20)
    for freeze_w, freeze_b, label in [
        (False, False, "grad_w_and_b"),
        (True, False, "freeze_w"),
        (False, True, "freeze_b"),
        (True, True, "freeze_both"),
    ]:
        cell = {"case": label}
        try:
            xl = x.clone().requires_grad_(True)
            wl = w.clone().requires_grad_(not freeze_w)
            bl = b.clone().requires_grad_(not freeze_b)
            yl = liger(xl, wl, bl)
            torch.autograd.backward(yl, dy)
            cell["dx_finite"] = bool(torch.isfinite(xl.grad).all())
            cell["w_grad_is_none"] = wl.grad is None
            cell["b_grad_is_none"] = bl.grad is None
            if wl.grad is not None:
                cell["dw_abs_mean"] = float(wl.grad.double().abs().mean())
            if bl.grad is not None:
                cell["db_abs_mean"] = float(bl.grad.double().abs().mean())
        except Exception as e:
            cell["error"] = f"{type(e).__name__}: {e}"
        rows.append(cell)
        print("edge", cell, flush=True)
        cleanup()

    # Non-finite inputs
    for kind in ["nan_row", "inf_row", "all_nan"]:
        x, w, b, dy = make_inputs(64, 512, torch.float32, seed=SEED + 21)
        if kind == "nan_row":
            x[3, :] = float("nan")
        elif kind == "inf_row":
            x[3, 0] = float("inf")
        else:
            x[:] = float("nan")
        cell = {"case": kind}
        try:
            y = liger(x, w, b)
            cell["y_nan"] = bool(torch.isnan(y).any())
            cell["y_inf"] = bool(torch.isinf(y).any())
            yf = eager_ln(x, w, b)
            cell["F_y_nan"] = bool(torch.isnan(yf).any())
            cell["F_y_inf"] = bool(torch.isinf(yf).any())
        except Exception as e:
            cell["error"] = f"{type(e).__name__}: {e}"
        rows.append(cell)
        print("edge", cell, flush=True)
        cleanup()

    # Module bias=False still has a bias Parameter (zeros)
    cell = {"case": "module_bias_false"}
    mod = LigerLayerNorm(512, eps=EPS, bias=False).cuda().to(torch.bfloat16)
    cell["has_bias_param"] = hasattr(mod, "bias") and mod.bias is not None
    cell["bias_abs_max"] = float(mod.bias.detach().float().abs().max())
    cell["bias_requires_grad"] = bool(mod.bias.requires_grad)
    x = torch.randn(32, 512, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    y = mod(x)
    y.sum().backward()
    cell["fwd_bwd_ok"] = True
    cell["dx_finite"] = bool(torch.isfinite(x.grad).all())
    rows.append(cell)
    print("edge", cell, flush=True)
    cleanup()

    # functional.layer_norm(bias=None) invents zeros_like(weight) each call
    from liger_kernel.functional import layer_norm as functional_layer_norm

    x, w, _, dy = make_inputs(64, 512, torch.bfloat16, seed=SEED + 22)
    cell = {"case": "functional_bias_none"}
    try:
        x1 = x.clone().requires_grad_(True)
        w1 = w.clone().requires_grad_(True)
        y1 = functional_layer_norm(x1, w1, None, EPS)
        torch.autograd.backward(y1, dy)
        x2 = x.clone().requires_grad_(True)
        w2 = w.clone().requires_grad_(True)
        b0 = torch.zeros_like(w)
        y2 = liger(x2, w2, b0)
        torch.autograd.backward(y2, dy)
        cell["y_rel_vs_zero_bias"] = rel(y1, y2)
        cell["dx_rel"] = rel(x1.grad, x2.grad)
        cell["dw_rel"] = rel(w1.grad, w2.grad)
    except Exception as e:
        cell["error"] = f"{type(e).__name__}: {e}"
    rows.append(cell)
    print("edge", cell, flush=True)
    cleanup()
    return {"rows": rows}


# ---------------------------------------------------------------- 4. SM reduction determinism
def sec_determinism():
    rows = []
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    dtype = torch.bfloat16
    # T covering: << SM, == SM, >> SM
    for T in [1, 3, sm - 1, sm, sm + 1, 256, 4096]:
        H = 2048
        x, w, b, dy = make_inputs(T, H, dtype, seed=SEED + 30 + T)
        # Low-level: capture dW across repeats
        Y, X, Mean, RSTD, BS, NW = layer_norm_forward(x, w, b, EPS)
        dws = []
        dxs = []
        dbs = []
        for i in range(DET_REPEATS):
            dx, dw, db = layer_norm_backward(dy, X, w, b, Mean, RSTD, needs_dw=True, needs_db=True)
            dws.append(dw.detach().clone())
            dxs.append(dx.detach().clone())
            dbs.append(db.detach().clone())
        bit_dw = all(torch.equal(dws[0], d) for d in dws[1:])
        bit_dx = all(torch.equal(dxs[0], d) for d in dxs[1:])
        bit_db = all(torch.equal(dbs[0], d) for d in dbs[1:])
        # vs F.layer_norm
        xf = x.clone().requires_grad_(True)
        wf = w.clone().requires_grad_(True)
        bf = b.clone().requires_grad_(True)
        eager_ln(xf, wf, bf).backward(dy)
        cell = {
            "T": T,
            "H": H,
            "sm_count": sm,
            "rows_per_program": math.ceil(T / sm),
            "bit_identical_dx": bit_dx,
            "bit_identical_dw": bit_dw,
            "bit_identical_db": bit_db,
            "dx_rel_vs_F": rel(dxs[0], xf.grad),
            "dw_rel_vs_F": rel(dws[0], wf.grad),
            "db_rel_vs_F": rel(dbs[0], bf.grad),
            "max_dw_spread_rel": 0.0 if bit_dw else max(rel(dws[0], d) for d in dws[1:]),
        }
        rows.append(cell)
        print("det", cell, flush=True)
        cleanup()
    return {"sm_count": sm, "repeats": DET_REPEATS, "rows": rows}


# ---------------------------------------------------------------- 5. memory / latency
def sec_memory():
    rows = []
    dtype = torch.bfloat16

    def _closed(a, ww, bb):
        return F.layer_norm(a, (a.shape[-1],), ww, bb, EPS)

    compiled = torch.compile(_closed, fullgraph=False)

    for name, H in MEM_SHAPES:
        for T in MEM_TOKENS:
            # Skip cells that would blow the 60% cap (~4.9 GB)
            est = T * H * 2 * 8 / 2**20  # rough: several activations
            if est > 3500:
                rows.append({"shape": name, "T": T, "H": H, "skipped": "est_too_big",
                             "est_mib": round(est, 1)})
                continue
            x, w, b, dy = make_inputs(T, H, dtype, seed=SEED + H + T)
            act = round(x.numel() * x.element_size() / 2**20, 2)
            cell = {"shape": name, "T": T, "H": H, "act_MiB": act}

            def one(label, fwd):
                cleanup()
                torch.cuda.synchronize()
                base = torch.cuda.memory_allocated()
                xx = x.clone().requires_grad_(True)
                ww = w.clone().requires_grad_(True)
                bb = b.clone().requires_grad_(True)
                y = fwd(xx, ww, bb)
                torch.cuda.synchronize()
                held = torch.cuda.memory_allocated() - base
                torch.autograd.backward(y, dy)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() - base

                def bench():
                    x2 = x.clone().requires_grad_(True)
                    w2 = w.clone().requires_grad_(True)
                    b2 = b.clone().requires_grad_(True)
                    y2 = fwd(x2, w2, b2)
                    torch.autograd.backward(y2, dy)

                ms = do_bench(bench, return_mode="median")
                cell[label] = {
                    "held_after_fwd_MiB": round(held / 2**20, 1),
                    "extra_peak_MiB": round(peak / 2**20, 1),
                    "fwd_bwd_ms": round(float(ms), 3),
                }
                del xx, ww, bb, y
                cleanup()

            one("liger", liger)
            one("F_layer_norm", eager_ln)

            def eager_manual(a, ww, bb):
                mean = a.float().mean(-1, keepdim=True)
                var = ((a.float() - mean) ** 2).mean(-1, keepdim=True)
                y = (a.float() - mean) * torch.rsqrt(var + EPS)
                return (y * ww.float() + bb.float()).to(a.dtype)

            one("eager_fp32_manual", eager_manual)
            try:
                # warmup compile
                _ = compiled(x[:2].clone(), w.clone(), b.clone())
                torch.cuda.synchronize()
                one("torch_compile", compiled)
            except Exception as e:
                cell["torch_compile"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            rows.append(cell)
            print("mem", name, T, {k: cell[k] for k in cell if k not in ("act_MiB",)}, flush=True)
            cleanup()
    return {"rows": rows}


# ---------------------------------------------------------------- 6. bandwidth
def sec_bandwidth():
    rows = []
    dtype = torch.bfloat16
    sm = torch.cuda.get_device_properties(0).multi_processor_count

    def _closed(a, ww, bb):
        return F.layer_norm(a, (a.shape[-1],), ww, bb, EPS)

    for H in BW_HS:
        try:
            block, warps = calculate_settings(H)
        except Exception as e:
            rows.append({"H": H, "error": f"{type(e).__name__}: {e}"})
            continue
        x, w, b, dy = make_inputs(BW_T, H, dtype, seed=SEED + H)
        nbytes = x.numel() * x.element_size()
        Y, X, Mean, RSTD, BS, NW = layer_norm_forward(x, w, b, EPS)

        compiled = torch.compile(_closed, fullgraph=False)
        compile_ok = True
        compile_err = None
        try:
            xc = x.clone().requires_grad_(True)
            yc = compiled(xc, w.clone(), b.clone())
            torch.autograd.grad(yc, (xc,), dy, retain_graph=True)
            del xc, yc
        except Exception as e:
            compile_ok = False
            compile_err = f"{type(e).__name__}: {str(e)[:120]}"

        xf = x.clone().requires_grad_(True)
        wf = w.clone().requires_grad_(True)
        bf = b.clone().requires_grad_(True)
        yf = eager_ln(xf, wf, bf)

        fns = {
            "copy": (lambda: x.clone(), 1),
            "fwd_liger": (lambda: layer_norm_forward(x, w, b, EPS)[0], 2),
            "fwd_F": (lambda: F.layer_norm(x, (H,), w, b, EPS), 2),
            "bwd_liger": (
                lambda: layer_norm_backward(dy, X, w, b, Mean, RSTD, needs_dw=True, needs_db=True),
                3,
            ),
            "bwd_F": (lambda: torch.autograd.grad(yf, (xf, wf, bf), dy, retain_graph=True), 3),
        }
        if compile_ok:
            xcc = x.clone().requires_grad_(True)
            wcc = w.clone().requires_grad_(True)
            bcc = b.clone().requires_grad_(True)
            ycc = compiled(xcc, wcc, bcc)
            fns["fwd_compile"] = (lambda: compiled(x, w, b), 2)
            fns["bwd_compile"] = (
                lambda: torch.autograd.grad(ycc, (xcc, wcc, bcc), dy, retain_graph=True),
                3,
            )

        do_bench(fns["copy"][0], rep=100)
        times = {k: [] for k in fns}
        for _ in range(BW_ROUNDS):
            for k, (fn, _) in fns.items():
                times[k].append(float(do_bench(fn, return_mode="median")))
        row = {
            "T": BW_T,
            "H": H,
            "BLOCK_SIZE": block,
            "num_warps": warps,
            "sm_count": sm,
            "lane_util": round(H / block, 3),
            "MiB_per_tensor": round(nbytes / 2**20, 1),
            "compile_ok": compile_ok,
        }
        if compile_err:
            row["compile_error"] = compile_err
        for k, (_, mult) in fns.items():
            ms = sorted(times[k])[len(times[k]) // 2]
            row[k + "_ms"] = round(ms, 4)
            row[k + "_gbps"] = round(mult * nbytes * 1e-9 / (ms * 1e-3), 1)
        row["fwd_liger_vs_F"] = round(row["fwd_F_ms"] / max(row["fwd_liger_ms"], 1e-9), 2)
        row["bwd_liger_vs_F"] = round(row["bwd_F_ms"] / max(row["bwd_liger_ms"], 1e-9), 2)
        if compile_ok:
            row["fwd_compile_vs_F"] = round(row["fwd_F_ms"] / max(row["fwd_compile_ms"], 1e-9), 2)
            row["bwd_compile_vs_F"] = round(row["bwd_F_ms"] / max(row["bwd_compile_ms"], 1e-9), 2)
        rows.append(row)
        print("bw", row, flush=True)
        cleanup()
    return {"rows": rows}


def main():
    sections = sys.argv[1:] or ["numerics", "layouts", "edges", "determinism", "memory", "bandwidth"]
    results = {
        "experiment": "0031-liger-layer-norm",
        "date": "2026-11-24",
        "seed": SEED,
        "gpu": GPU,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "eps": EPS,
    }
    md = [
        "# 0031 — Liger-Kernel LayerNorm results",
        "",
        f"GPU: {GPU.get('name')} | torch {torch.__version__} | seed {SEED}",
        "",
    ]
    t0 = time.time()

    if "numerics" in sections:
        print("== numerics ==")
        data = sec_numerics()
        results["numerics"] = data
        md.append("## Numerics (vs fp64 + F.layer_norm)")
        md.append("")
        md.append(table(
            ["dtype", "bias", "floor_y", "liger_y", "F_y", "liger_dx", "F_dx",
             "liger_dw", "F_dw", "vs_F_y", "compile_y"],
            [[
                r["dtype"], r["with_bias"], r["floor"]["y"],
                r["liger"]["y_rel"], r["F_layer_norm"]["y_rel"],
                r["liger"]["dx_rel"], r["F_layer_norm"]["dx_rel"],
                r["liger"]["dw_rel"], r["F_layer_norm"]["dw_rel"],
                r["liger_vs_F"]["y"],
                r["torch_compile"].get("y_rel", r["torch_compile"].get("error", "-")),
            ] for r in data["rows"]],
        ))
        md.append("")

    if "layouts" in sections:
        print("== layouts ==")
        data = sec_layouts()
        results["layouts"] = data
        md.append("## Layouts (strided / transposed / stride-0 dy)")
        md.append("")
        md.append(table(
            ["case", "contig", "y_rel", "dx_rel", "dw_rel", "vs_F_dx", "extra"],
            [[
                r["case"], r.get("x_contiguous", "-"),
                r.get("y_rel", r.get("error")),
                r.get("dx_rel", r.get("dx_finite")),
                r.get("dw_rel", r.get("dw_finite")),
                r.get("liger_vs_F_dx", r.get("dx_rel_vs_F")),
                {k: r[k] for k in ("canary_unchanged", "dx_abs_mean", "error") if k in r},
            ] for r in data["rows"]],
        ))
        md.append("")

    if "edges" in sections:
        print("== edges ==")
        data = sec_edges()
        results["edges"] = data
        md.append("## Edges / limits")
        md.append("")
        md.append(table(
            ["case", "detail"],
            [[r["case"], {k: r[k] for k in r if k != "case"}] for r in data["rows"]],
        ))
        md.append("")

    if "determinism" in sections:
        print("== determinism ==")
        data = sec_determinism()
        results["determinism"] = data
        md.append(f"## SM-partitioned dW/db determinism (sm={data['sm_count']}, repeats={data['repeats']})")
        md.append("")
        md.append(table(
            ["T", "rows/prog", "bit_dx", "bit_dw", "bit_db", "dx_vs_F", "dw_vs_F", "db_vs_F"],
            [[
                r["T"], r["rows_per_program"], r["bit_identical_dx"], r["bit_identical_dw"],
                r["bit_identical_db"], r["dx_rel_vs_F"], r["dw_rel_vs_F"], r["db_rel_vs_F"],
            ] for r in data["rows"]],
        ))
        md.append("")

    if "memory" in sections:
        print("== memory ==")
        data = sec_memory()
        results["memory"] = data
        md.append("## Memory / latency (bf16)")
        md.append("")

        def mem_row(r):
            if "skipped" in r:
                return [r.get("shape"), r.get("T"), r.get("H"), r["skipped"], "-", "-", "-"]
            lig = r.get("liger", {})
            fl = r.get("F_layer_norm", {})
            return [
                r.get("shape"), r.get("T"), r.get("H"),
                lig.get("held_after_fwd_MiB"), lig.get("extra_peak_MiB"),
                lig.get("fwd_bwd_ms"), fl.get("fwd_bwd_ms"),
                (r.get("torch_compile") or {}).get("fwd_bwd_ms",
                                                    (r.get("torch_compile") or {}).get("error", "-")),
            ]

        md.append(table(
            ["shape", "T", "H", "liger_held", "liger_peak", "liger_ms", "F_ms", "compile_ms"],
            [mem_row(r) for r in data["rows"]],
        ))
        md.append("")

    if "bandwidth" in sections:
        print("== bandwidth ==")
        data = sec_bandwidth()
        results["bandwidth"] = data
        md.append("## Bandwidth vs H (bf16, T=4096)")
        md.append("")
        md.append(table(
            ["H", "block", "copy", "fwd_L", "fwd_F", "bwd_L", "bwd_F", "L/F_fwd", "L/F_bwd"],
            [[
                r.get("H"), r.get("BLOCK_SIZE"),
                r.get("copy_gbps"), r.get("fwd_liger_gbps"), r.get("fwd_F_gbps"),
                r.get("bwd_liger_gbps"), r.get("bwd_F_gbps"),
                r.get("fwd_liger_vs_F"), r.get("bwd_liger_vs_F"),
            ] for r in data["rows"]],
        ))
        md.append("")

    cleanup()
    results["wall_s"] = round(time.time() - t0, 1)
    results["peak_memory_mib"] = round(max(_PEAK_MIB, gpu_guard.peak_memory_mib()), 1)
    (HERE / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    (HERE / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"done in {results['wall_s']}s, peak {results['peak_memory_mib']} MiB")
    print(f"wrote {HERE / 'results.json'}")


if __name__ == "__main__":
    main()
