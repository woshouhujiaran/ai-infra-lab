"""Liger-Kernel Triton sparsemax (`LigerSparsemaxFunction`) on an RTX 4060 Laptop.

Forward: `torch.sort(x.float())` (values + int64 indices, both materialized),
then `_sparsemax_forward_kernel` (one program per row, the whole row in one
`next_power_of_2(n_cols)` block, fp32 `cumsum` of the sorted row, support
`z_k > (cumsum_k - 1) / k`, `tau = (sum_support - 1) / k`, `y = max(x - tau, 0)`).
Backward: `_sparsemax_backward_kernel` recomputes the support from `y > 0` in the
input dtype and subtracts the support mean of grad_output (cast to the dtype).

1. Numerics vs an fp64 exact sparsemax (and analytic backward) for fp32 / fp16 /
   bf16, V in 128..65536, logit scale 0.01..10 (flat rows have huge supports).
2. Shift invariance: sparsemax(x + C) = sparsemax(x), but the fp32 cumsum is not.
   Exact ties of an element with tau (frequent in bf16 / fp16): support membership flips.
3. Edge cases: ties, n_cols 1 / 2, -inf padding, +inf, NaN, all -inf, V = 65536 /
   65537 / 151936.
4. API / autograd: dim != -1, strided input, in-place edits of the output,
   grad_output with a large common offset in half precision.
5. Kernel registers / spills, memory and latency vs shape; sort share.

`RowsSparsemax` (in this file) is a sort-free exact rewrite: row max shift, then
Michelot's fixed-point iteration over the support in 4096-column chunks, fp32 mean
in backward, support recomputed from the saved input. Liger-Kernel is imported
unmodified from the submodule.
"""
import gc
import json
import math
import subprocess
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
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.testing import do_bench  # noqa: E402
from liger_kernel.ops.sparsemax import (  # noqa: E402
    LigerSparsemaxFunction, _sparsemax_backward_kernel, _sparsemax_forward_kernel)
from liger_kernel.ops.utils import calculate_settings  # noqa: E402

SEED = 0
DEV = "cuda"
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
NUM_BT = 64
NUM_VS = [128, 4096, 32768, 65536]
NUM_SCALES = [0.01, 0.1, 1.0, 10.0]
SHIFTS = [0.0, 1e2, 1e3, 1e4, 1e5, 1e6]
SHIFT_BT, SHIFT_V = 64, 32768
MAX_IT = 1024
ELEMS = 2 ** 22
PERF_VS = [128, 512, 2048, 8192, 32768, 65536]
COMPILE_VS = {2048, 32768}
MEM_CELLS = [(65536, [256, 1024, 2048, 3072, 4096]), (151936, [256, 1024, 2048])]

torch._dynamo.config.recompile_limit = 64


# ---------------------------------------------------------------- rewrite
@triton.jit
def _michelot_fwd_kernel(X, Y, ST, IT, stride, n_cols, MAX_IT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    xr = X + row * stride
    yr = Y + row * stride
    offs0 = tl.arange(0, BLOCK)
    m = tl.zeros((), tl.float32) + float("-inf")
    for i in range(0, n_cols, BLOCK):
        offs = i + offs0
        x = tl.load(xr + offs, mask=offs < n_cols, other=float("-inf")).to(tl.float32)
        m = tl.maximum(m, tl.max(x, axis=0))
    m = tl.where(m == float("-inf"), 0.0, m)
    t = tl.zeros((), tl.float32) - 1.0
    c_prev = tl.zeros((), tl.int32) - 1
    c_cur = tl.zeros((), tl.int32)
    it = tl.zeros((), tl.int32)
    while (c_cur != c_prev) & (it < MAX_IT):
        c_prev = c_cur
        s = tl.zeros((), tl.float32)
        c = tl.zeros((), tl.int32)
        for i in range(0, n_cols, BLOCK):
            offs = i + offs0
            x = tl.load(xr + offs, mask=offs < n_cols, other=float("-inf")).to(tl.float32)
            z = x - m
            sel = z > t
            s += tl.sum(tl.where(sel, z, 0.0), axis=0)
            c += tl.sum(sel.to(tl.int32), axis=0)
        t = (s - 1.0) / tl.maximum(c, 1).to(tl.float32)
        c_cur = c
        it += 1
    for i in range(0, n_cols, BLOCK):
        offs = i + offs0
        mask = offs < n_cols
        x = tl.load(xr + offs, mask=mask, other=float("-inf")).to(tl.float32)
        tl.store(yr + offs, tl.maximum(x - m - t, 0.0).to(Y.dtype.element_ty), mask=mask)
    tl.store(ST + row * 2, m)
    tl.store(ST + row * 2 + 1, t)
    tl.store(IT + row, it)


@triton.jit
def _michelot_bwd_kernel(X, G, DX, ST, stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    xr = X + row * stride
    gr = G + row * stride
    dr = DX + row * stride
    m = tl.load(ST + row * 2)
    t = tl.load(ST + row * 2 + 1)
    offs0 = tl.arange(0, BLOCK)
    gs = tl.zeros((), tl.float32)
    cnt = tl.zeros((), tl.float32)
    for i in range(0, n_cols, BLOCK):
        offs = i + offs0
        mask = offs < n_cols
        x = tl.load(xr + offs, mask=mask, other=float("-inf")).to(tl.float32)
        g = tl.load(gr + offs, mask=mask, other=0.0).to(tl.float32)
        sel = (x - m) > t
        gs += tl.sum(tl.where(sel, g, 0.0), axis=0)
        cnt += tl.sum(sel.to(tl.float32), axis=0)
    mean = gs / tl.maximum(cnt, 1.0)
    for i in range(0, n_cols, BLOCK):
        offs = i + offs0
        mask = offs < n_cols
        x = tl.load(xr + offs, mask=mask, other=float("-inf")).to(tl.float32)
        g = tl.load(gr + offs, mask=mask, other=0.0).to(tl.float32)
        sel = (x - m) > t
        tl.store(dr + offs, tl.where(sel, g - mean, 0.0).to(DX.dtype.element_ty), mask=mask)


def _rows_cfg(n):
    block = min(4096, triton.next_power_of_2(n))
    return block, (4 if block <= 1024 else 8)


class RowsSparsemax(torch.autograd.Function):
    last_iters = None

    @staticmethod
    def forward(ctx, x):
        V = x.shape[-1]
        x2 = x.contiguous().view(-1, V)
        y = torch.empty_like(x2)
        st = torch.empty(x2.shape[0], 2, device=x.device, dtype=torch.float32)
        it = torch.empty(x2.shape[0], device=x.device, dtype=torch.int32)
        block, nw = _rows_cfg(V)
        _michelot_fwd_kernel[(x2.shape[0],)](x2, y, st, it, x2.stride(0), V,
                                             MAX_IT=MAX_IT, BLOCK=block, num_warps=nw)
        RowsSparsemax.last_iters = it
        ctx.save_for_backward(x2, st)
        return y.view_as(x)

    @staticmethod
    def backward(ctx, g):
        x2, st = ctx.saved_tensors
        V = x2.shape[-1]
        g2 = g.contiguous().view(-1, V)
        dx = torch.empty_like(g2)
        block, nw = _rows_cfg(V)
        _michelot_bwd_kernel[(x2.shape[0],)](x2, g2, dx, st, x2.stride(0), V, BLOCK=block, num_warps=nw)
        return dx.view_as(g)


def liger_sm(x):
    return LigerSparsemaxFunction.apply(x, -1)


def rows_sm(x):
    return RowsSparsemax.apply(x)


def torch_sm(x):
    """Eager sort + cumsum sparsemax in fp32 (upstream's own test reference), autograd through it."""
    xf = x.float()
    s, _ = torch.sort(xf, dim=-1, descending=True)
    cs = torch.cumsum(s, dim=-1)
    k = torch.arange(1, xf.shape[-1] + 1, device=x.device, dtype=torch.float32)
    sup = (1 + k * s) > cs
    kk = sup.sum(-1, keepdim=True).clamp(min=1)
    tau = ((s * sup).sum(-1, keepdim=True) - 1) / kk
    return (xf - tau).clamp(min=0).to(x.dtype)


IMPLS = {"liger": liger_sm, "rows": rows_sm, "torch": torch_sm}


# ---------------------------------------------------------------- references / helpers
def ref_sm64(x, dim=-1):
    x = x.double().movedim(dim, -1)
    s, _ = torch.sort(x, dim=-1, descending=True)
    cs = torch.cumsum(s, dim=-1)
    k = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=torch.float64)
    sup = (1 + k * s) > cs
    kk = sup.sum(-1, keepdim=True).clamp(min=1)
    tau = ((s * sup).sum(-1, keepdim=True) - 1) / kk
    return (x - tau).clamp(min=0).movedim(-1, dim)


def ref_dx64(y, g, dim=-1):
    y, g = y.movedim(dim, -1), g.double().movedim(dim, -1)
    sup = y > 0
    mean = (g * sup).sum(-1, keepdim=True) / sup.sum(-1, keepdim=True).clamp(min=1)
    return torch.where(sup, g - mean, torch.zeros_like(g)).movedim(-1, dim)


def rel(a, b):
    a, b = a.detach().double().flatten(), b.detach().double().flatten()
    den = b.norm()
    return float((a - b).norm() / den) if den > 0 else float((a - b).norm())


def fmt(x, digits=4):
    if x is None:
        return "-"
    if isinstance(x, (bool, str)):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e5):
        return f"{x:.{digits - 1}e}"
    return f"{x:.{digits}g}"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(c) for c in r) + " |")
    return "\n".join(out)


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def run_case(fn, x, g):
    """Forward + backward of one impl; returns (y, dx, exception string)."""
    xr = x.clone().requires_grad_(True)
    try:
        y = fn(xr)
        y.backward(g)
        return y.detach(), xr.grad, None
    except Exception as e:  # noqa: BLE001
        return None, None, f"{type(e).__name__}: {str(e).splitlines()[0][:110]}"


def metrics(y, dx, yref, dxref, floor_y, floor_dx, g=None):
    out = {}
    if y is None:
        return out
    if g is not None:
        out["dx_own_rel"] = rel(dx, ref_dx64(y.detach().double(), g))
    yd = y.double()
    out["y_rel"] = rel(yd, yref)
    out["y_absmax"] = float((yd - yref).abs().max())
    out["sum_err"] = float((yd.sum(-1) - 1).abs().max())
    out["supp_mismatch"] = int(((yd > 0) != (yref > 0)).sum())
    out["supp_mismatch_rows"] = int(((yd > 0) != (yref > 0)).any(-1).sum())
    out["dx_rel"] = rel(dx, dxref)
    out["y_floor"], out["dx_floor"] = floor_y, floor_dx
    return out


# ---------------------------------------------------------------- 1. numerics
def sec_numerics():
    gen = torch.Generator(device=DEV).manual_seed(SEED)
    rows = []
    for dname, dt in DTYPES.items():
        for V in NUM_VS:
            for scale in NUM_SCALES:
                x = (torch.randn(NUM_BT, V, device=DEV, generator=gen) * scale).to(dt)
                g = torch.randn(NUM_BT, V, device=DEV, generator=gen).to(dt)
                yref = ref_sm64(x)
                dxref = ref_dx64(yref, g)
                fy = rel(yref.to(dt), yref)
                fd = rel(dxref.to(dt), dxref)
                r = {"dtype": dname, "V": V, "scale": scale,
                     "support_mean": float((yref > 0).sum(-1).double().mean())}
                for name, fn in IMPLS.items():
                    y, dx, exc = run_case(fn, x, g)
                    m = metrics(y, dx, yref, dxref, fy, fd, g)
                    if name == "rows" and exc is None:
                        run_case(fn, x, g)
                        r["rows_iters_mean"] = float(RowsSparsemax.last_iters.double().mean())
                        r["rows_iters_max"] = int(RowsSparsemax.last_iters.max())
                    r[name] = dict(m, exc=exc)
                rows.append(r)
                print("  num", dname, V, scale, {k: fmt(r[k].get("y_rel")) for k in IMPLS}, flush=True)
                del x, g, yref, dxref
                cleanup()
    return {"rows": rows}


# ---------------------------------------------------------------- 2. shift invariance
def sec_shift():
    gen = torch.Generator(device=DEV).manual_seed(SEED + 1)
    rows = []
    base = torch.randn(SHIFT_BT, SHIFT_V, device=DEV, generator=gen)
    g = torch.randn(SHIFT_BT, SHIFT_V, device=DEV, generator=gen)
    impls = dict(IMPLS)
    impls["liger+preshift"] = lambda x: liger_sm(x - x.amax(-1, keepdim=True).detach())
    for C in SHIFTS:
        x = base + C
        yref = ref_sm64(x)
        dxref = ref_dx64(yref, g)
        r = {"shift": C, "support_mean": float((yref > 0).sum(-1).double().mean())}
        for name, fn in impls.items():
            y, dx, exc = run_case(fn, x, g)
            r[name] = dict(metrics(y, dx, yref, dxref, 0.0, 0.0), exc=exc)
        rows.append(r)
        print("  shift", C, {k: fmt(r[k].get("y_rel")) for k in impls}, flush=True)
    return {"BT": SHIFT_BT, "V": SHIFT_V, "rows": rows}


# ---------------------------------------------------------------- 2b. exact boundary ties
def sec_ties():
    """An element exactly at tau has y = 0 whether or not it is counted in the support, but the
    Jacobian is discontinuous there: bf16 / fp16 inputs hit exact ties often, and which side
    each implementation lands on depends on its fp32 rounding of tau."""
    gen = torch.Generator(device=DEV).manual_seed(SEED + 5)
    rows = []
    for dname in ["fp32", "fp16", "bf16"]:
        dt = DTYPES[dname]
        for V in [4096, 32768]:
            BT = 2048 if V <= 4096 else 512
            x = torch.randn(BT, V, device=DEV, generator=gen).to(dt)
            g = torch.randn(BT, V, device=DEV, generator=gen).to(dt)
            yref = ref_sm64(x)
            sup = yref > 0
            tau = torch.where(sup, x.double() - yref, torch.full_like(yref, -float("inf"))).amax(-1, keepdim=True)
            tie_rows = int((x.double() == tau).any(-1).sum())
            dxref = ref_dx64(yref, g)
            r = {"dtype": dname, "V": V, "BT": BT, "exact_tie_rows": tie_rows}
            for name, fn in IMPLS.items():
                y, dx, exc = run_case(fn, x, g)
                bad = ((y.double() > 0) != sup).any(-1)
                r[name] = {"mismatch_rows": int(bad.sum()),
                           "dx_rel_all": rel(dx, dxref),
                           "dx_rel_own_support": rel(dx, ref_dx64(y.double(), g))}
            rows.append(r)
            print("  ties", dname, V, tie_rows, {k: r[k]["mismatch_rows"] for k in IMPLS}, flush=True)
            del x, g, yref, dxref
            cleanup()
    return {"rows": rows}


# ---------------------------------------------------------------- 3. edge cases
def edge_rows():
    inf = float("inf")
    sc = []

    def add(name, rows, expected=None):
        sc.append((name, torch.tensor(rows, device=DEV, dtype=torch.float32), expected))

    add("uniform V=1000", [[0.7] * 1000], [[1e-3] * 1000])
    add("top tie [5, 5, 0...]", [[5.0, 5.0] + [0.0] * 6], [[0.5, 0.5] + [0.0] * 6])
    add("n_cols = 1", [[3.5]], [[1.0]])
    add("n_cols = 2", [[0.3, 0.1]], [[0.6, 0.4]])
    pad = torch.randn(1, 1000, device=DEV, generator=torch.Generator(device=DEV).manual_seed(5))
    exp = torch.zeros(1, 1000, dtype=torch.float64, device=DEV)
    exp[:, :976] = ref_sm64(pad[:, :976])
    padded = pad.clone()
    padded[:, 976:] = -inf
    sc.append(("-inf padding (24 of 1000)", padded, exp))
    add("one finite, rest -inf", [[3.0] + [-inf] * 7], [[1.0] + [0.0] * 7])
    add("one +inf", [[1.0, inf, 0.5, 0.0]])
    add("one NaN", [[1.0, float("nan"), 0.5, 0.0]])
    add("all -inf", [[-inf] * 8])
    add("NaN + -inf + two finite", [[float("nan"), -inf, 2.0, 1.5]])
    add("all NaN", [[float("nan")] * 4])
    return sc


def sec_edge():
    rows = []
    for name, x, expected in edge_rows():
        g = torch.randn_like(x)
        r = {"scenario": name}
        for iname, fn in IMPLS.items():
            y, dx, exc = run_case(fn, x, g)
            if exc:
                r[iname] = {"exc": exc}
                continue
            yd = y.double()
            r[iname] = {"nan_frac": float(torch.isnan(yd).double().mean()), "sum": float(yd.sum(-1).max()),
                        "err": None if expected is None else float((yd - torch.as_tensor(expected, device=DEV, dtype=torch.float64)).abs().max()),
                        "dx_nan_frac": float(torch.isnan(dx).double().mean())}
            if x.shape[-1] <= 8:
                r[iname]["y"] = y[0].tolist()
        rows.append(r)
        print("  edge", name, {k: v.get("exc") or v.get("err") for k, v in r.items() if k in IMPLS}, flush=True)
    big = []
    gen = torch.Generator(device=DEV).manual_seed(SEED + 2)
    for V in [65536, 65537, 151936]:
        x = torch.randn(4, V, device=DEV, generator=gen)
        g = torch.randn_like(x)
        yref = ref_sm64(x)
        dxref = ref_dx64(yref, g)
        r = {"V": V, "support_mean": float((yref > 0).sum(-1).double().mean())}
        for iname, fn in IMPLS.items():
            y, dx, exc = run_case(fn, x, g)
            r[iname] = {"exc": exc} if exc else {"y_rel": rel(y, yref), "dx_rel": rel(dx, dxref)}
        big.append(r)
        print("  big", V, {k: v.get("exc") or v.get("y_rel") for k, v in r.items() if k in IMPLS}, flush=True)
    return {"rows": rows, "big_v": big}


# ---------------------------------------------------------------- 4. API / autograd
def sec_api():
    gen = torch.Generator(device=DEV).manual_seed(SEED + 3)
    res = {}
    x = torch.randn(4, 33, 129, device=DEV, generator=gen)
    g = torch.randn_like(x)
    dims = {}
    for dim in [0, 1, 2, -2]:
        xr = x.clone().requires_grad_(True)
        y = LigerSparsemaxFunction.apply(xr, dim)
        y.backward(g)
        yref = ref_sm64(x, dim)
        dims[str(dim)] = {"y_rel": rel(y, yref), "dx_rel": rel(xr.grad, ref_dx64(yref, g, dim)),
                          "y_contiguous": y.is_contiguous(), "dx_contiguous": xr.grad.is_contiguous(),
                          "y_stride": list(y.stride())}
    res["dim"] = dims

    big = torch.randn(64, 2048, device=DEV, generator=gen)
    xs = big[:, ::2]
    gs = torch.randn(64, 1024, device=DEV, generator=gen)
    xr = xs.clone().requires_grad_(True)
    y = liger_sm(xr)
    y.backward(gs)
    yref = ref_sm64(xs)
    res["strided_input"] = {"y_rel": rel(y, yref), "dx_rel": rel(xr.grad, ref_dx64(yref, gs))}

    x2 = torch.randn(32, 2048, device=DEV, generator=gen)
    g2 = torch.randn_like(x2)
    yref = ref_sm64(x2)
    dxref = ref_dx64(yref, g2)
    for tag, edit in [("y.mul_(2)", lambda y: y.mul_(2)), ("y.clamp_(max=0.01)", lambda y: y.clamp_(max=0.01)),
                      ("y.zero_()", lambda y: y.zero_())]:
        out = {}
        for iname in ["liger", "rows"]:
            xr = x2.clone().requires_grad_(True)
            y = IMPLS[iname](xr)
            try:
                edit(y)
                y.backward(g2)
                out[iname] = {"raised": False, "dx_rel_vs_unedited": rel(xr.grad, dxref)}
            except Exception as e:  # noqa: BLE001
                out[iname] = {"raised": True, "msg": str(e).splitlines()[0][:100]}
        res.setdefault("inplace_output", {})[tag] = out

    xr = x2.clone().requires_grad_(True)
    y = liger_sm(xr)
    y.backward(g2, retain_graph=True)
    d1 = xr.grad.clone()
    xr.grad = None
    y.backward(g2 * 3, retain_graph=True)
    res["retain_graph"] = {"second_backward_equals_3x_first": bool(torch.allclose(xr.grad, d1 * 3, atol=1e-6))}

    off = []
    x3 = torch.randn(64, 8192, device=DEV, generator=gen) * 0.1
    noise = torch.randn(64, 8192, device=DEV, generator=gen)
    for dname in ["fp32", "fp16", "bf16"]:
        dt = DTYPES[dname]
        xh = x3.to(dt)
        for c in [0.0, 1.0, 10.0, 100.0, 1000.0]:
            gh = (noise + c).to(dt)
            dxref = ref_dx64(ref_sm64(xh), gh)
            r = {"dtype": dname, "offset": c, "floor": rel(dxref.to(dt), dxref)}
            for iname in ["liger", "rows"]:
                yi, dx, exc = run_case(IMPLS[iname], xh, gh)
                r[iname] = rel(dx, ref_dx64(yi.double(), gh))
            off.append(r)
    res["grad_offset"] = off
    print("  api done", flush=True)
    return res


# ---------------------------------------------------------------- 5. kernels, memory, latency
def kernel_meta():
    rows = []
    for n in [128, 1024, 4096, 16384, 32768, 65536]:
        block, nw = calculate_settings(n)
        x = torch.zeros(1, n, device=DEV)
        xs = torch.zeros(1, n, device=DEV)
        o = torch.zeros(1, n, device=DEV)
        r = {"n": n, "BLOCK": block, "num_warps": nw}
        try:
            k = _sparsemax_forward_kernel.warmup(x, x.stride(0), xs, xs.stride(0), o, o.stride(0), n,
                                                 BLOCK_SIZE=block, num_warps=nw, grid=(1,))
            k._init_handles()
            r.update(fwd_regs=k.n_regs, fwd_spills=k.n_spills, fwd_shared=k.metadata.shared)
            k = _sparsemax_backward_kernel.warmup(o, x, xs, o.stride(0), n, BLOCK_SIZE=block, num_warps=nw, grid=(1,))
            k._init_handles()
            r.update(bwd_regs=k.n_regs, bwd_spills=k.n_spills)
            b2, w2 = _rows_cfg(n)
            k = _michelot_fwd_kernel.warmup(x, o, torch.zeros(1, 2, device=DEV), torch.zeros(1, dtype=torch.int32, device=DEV),
                                            x.stride(0), n, MAX_IT=MAX_IT, BLOCK=b2, num_warps=w2, grid=(1,))
            k._init_handles()
            r.update(rows_regs=k.n_regs, rows_spills=k.n_spills)
        except Exception as e:  # noqa: BLE001
            r["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:100]}"
        rows.append(r)
    return rows


def measure(fn, x, g):
    res = {}
    x.grad = None
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    y = fn(x)
    torch.cuda.synchronize()
    res["held_after_fwd_mib"] = round((torch.cuda.memory_allocated() - base) / 2 ** 20, 1)
    res["peak_fwd_mib"] = round((torch.cuda.max_memory_allocated() - base) / 2 ** 20, 1)
    torch.cuda.reset_peak_memory_stats()
    y.backward(g)
    torch.cuda.synchronize()
    res["peak_bwd_mib"] = round((torch.cuda.max_memory_allocated() - base) / 2 ** 20, 1)
    del y
    x.grad = None

    def fwd_bwd():
        x.grad = None
        fn(x).backward(g)

    res["fwd_bwd_ms"] = do_bench(fwd_bwd, warmup=10, rep=40)
    with torch.no_grad():
        res["nograd_fwd_ms"] = do_bench(lambda: fn(x), warmup=10, rep=40)
    return res


def perf_cell(V, BT, dname, impls, sort_ms=True):
    dt = DTYPES[dname]
    gen = torch.Generator(device=DEV).manual_seed(SEED + 4)
    out = []
    for name, fn in impls.items():
        r = {"V": V, "BT": BT, "dtype": dname, "impl": name}
        try:
            x = torch.randn(BT, V, device=DEV, generator=gen).to(dt).requires_grad_(True)
            g = torch.randn(BT, V, device=DEV, generator=gen).to(dt)
            r.update(measure(fn, x, g))
            r["gbps"] = 3 * BT * V * x.element_size() / (r["fwd_bwd_ms"] * 1e-3) / 1e9
            if name == "liger" and sort_ms:
                xf = x.detach()
                r["sort_ms"] = do_bench(lambda: torch.sort(xf.float(), dim=-1, descending=True), warmup=5, rep=20)
            r["oom"] = False
        except torch.OutOfMemoryError:
            r["oom"] = True
        except Exception as e:  # noqa: BLE001
            r["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:100]}"
        finally:
            x = g = xf = None
            cleanup()
        out.append(r)
        print("  perf", {k: fmt(v) for k, v in r.items()}, flush=True)
    return out


def sec_perf():
    res = {"kernels": kernel_meta(), "shape_sweep": [], "mem_sweep": []}
    w = torch.randn(4096, 4096, device=DEV)
    t_end = time.time() + 3
    while time.time() < t_end:
        w @ w
    torch.cuda.synchronize()
    del w
    compiled = torch.compile(torch_sm)
    for dname in ["bf16", "fp32"]:
        for V in PERF_VS:
            impls = dict(IMPLS)
            if V in COMPILE_VS:
                impls["torch.compile"] = compiled
            res["shape_sweep"] += perf_cell(V, ELEMS // V, dname, impls)
    for V, bts in MEM_CELLS:
        for BT in bts:
            res["mem_sweep"] += perf_cell(V, BT, "bf16", dict(IMPLS), sort_ms=False)
    return res


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    res = {"gpu": GPU, "triton": triton.__version__,
           "liger_commit": subprocess.run(["git", "-C", str(LIGER_DIR), "rev-parse", "--short", "HEAD"],
                                          capture_output=True, text=True).stdout.strip()}
    sections = [("numerics", sec_numerics), ("shift", sec_shift), ("ties", sec_ties), ("edge", sec_edge),
                ("api", sec_api), ("perf", sec_perf)]
    only = set(sys.argv[1:])
    for name, fn in sections:
        if only and name not in only:
            continue
        print(f"== {name}", flush=True)
        res[name] = fn()
    res["peak_allocated_mib"] = gpu_guard.peak_memory_mib()
    res["runtime_s"] = round(time.time() - t0, 1)
    if only:
        print(json.dumps(res, indent=1, default=str)[:6000])
        return
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=str))
    (HERE / "results.md").write_text(render_md(res))
    print("done in", res["runtime_s"], "s")


def render_md(res):
    L = [f"# 0026 results\n\nGPU: {res['gpu']['name']}, torch {res['gpu']['torch']}, triton {res['triton']}, "
         f"Liger {res['liger_commit']}; peak allocated {res['peak_allocated_mib']} MiB; runtime {res['runtime_s']} s\n"]
    L.append("## 1. Numerics vs fp64 exact sparsemax (BT=64; rel L2 error; floor = rounding of the fp64 result to the dtype)\n")
    hdr = ["dtype", "V", "scale", "support", "y floor", "liger y", "rows y", "torch y", "dx floor", "liger dx", "rows dx",
           "torch dx", "liger dx (own supp)", "rows dx (own supp)", "liger supp mismatch rows", "rows supp mismatch rows",
           "liger |sum-1|", "rows |sum-1|", "rows iters mean/max"]
    rows = []
    for r in res["numerics"]["rows"]:
        lg, rw, tr = r["liger"], r["rows"], r["torch"]
        rows.append([r["dtype"], r["V"], r["scale"], r["support_mean"], lg.get("y_floor"), lg.get("y_rel"), rw.get("y_rel"),
                     tr.get("y_rel"), lg.get("dx_floor"), lg.get("dx_rel"), rw.get("dx_rel"), tr.get("dx_rel"),
                     lg.get("dx_own_rel"), rw.get("dx_own_rel"),
                     lg.get("supp_mismatch_rows", lg.get("exc")), rw.get("supp_mismatch_rows", rw.get("exc")),
                     lg.get("sum_err"), rw.get("sum_err"),
                     f"{fmt(r.get('rows_iters_mean'))} / {r.get('rows_iters_max', '-')}"])
    L.append(table(hdr, rows))
    s = res["shift"]
    L.append(f"\n## 2. Shift invariance (fp32, BT={s['BT']}, V={s['V']}, x = randn + C; y rel err vs fp64 / support-mismatch rows)\n")
    names = [k for k in s["rows"][0] if k not in ("shift", "support_mean")]
    L.append(table(["C", "support"] + [f"{n} y" for n in names] + [f"{n} dx" for n in names] + [f"{n} mismatch rows" for n in names],
                   [[r["shift"], r["support_mean"]] + [r[n].get("y_rel", r[n].get("exc")) for n in names]
                    + [r[n].get("dx_rel") for n in names] + [r[n].get("supp_mismatch_rows") for n in names] for r in s["rows"]]))
    L.append("\n### Exact boundary ties (BT=2048 for V=4096, 512 for V=32768, scale 1; rows whose support differs from fp64 / dX error vs fp64 support and vs own support)\n")
    L.append(table(["dtype", "V", "rows with exact tie at tau", "liger mismatch", "rows mismatch", "torch mismatch",
                    "liger dx (fp64 supp / own)", "rows dx (fp64 supp / own)", "torch dx (fp64 supp / own)"],
                   [[r["dtype"], r["V"], r["exact_tie_rows"]] + [r[k]["mismatch_rows"] for k in IMPLS]
                    + [f"{fmt(r[k]['dx_rel_all'])} / {fmt(r[k]['dx_rel_own_support'])}" for k in IMPLS]
                    for r in res["ties"]["rows"]]))
    e = res["edge"]
    L.append("\n## 3. Edge cases (fp32)\n")
    for r in e["rows"]:
        L.append(f"- **{r['scenario']}**: " + "; ".join(f"{k}: " + json.dumps(r[k]) for k in IMPLS))
    L.append("\n### Large V (BT=4, randn)\n")
    L.append(table(["V", "support", "liger", "rows", "torch"],
                   [[r["V"], r["support_mean"]] + [r[k].get("exc") or f"y {fmt(r[k]['y_rel'])} / dx {fmt(r[k]['dx_rel'])}" for k in IMPLS]
                    for r in e["big_v"]]))
    a = res["api"]
    L.append("\n## 4. API / autograd\n")
    for k in ["dim", "strided_input", "inplace_output", "retain_graph"]:
        L.append(f"**{k}**: " + json.dumps(a[k]) + "\n")
    L.append("\n### grad_output with a common offset (exact dX is independent of the offset; rel L2 err vs fp64 dX on each impl's own support)\n")
    L.append(table(["dtype", "offset", "floor", "liger", "rows"],
                   [[r["dtype"], r["offset"], r["floor"], r["liger"], r["rows"]] for r in a["grad_offset"]]))
    p = res["perf"]
    L.append("\n## 5. Kernels, memory, latency\n")
    L.append(table(["n", "BLOCK", "warps", "fwd regs", "fwd spills", "fwd shared B", "bwd regs", "bwd spills", "rows regs", "rows spills", "error"],
                   [[k["n"], k["BLOCK"], k["num_warps"], k.get("fwd_regs"), k.get("fwd_spills"), k.get("fwd_shared"),
                     k.get("bwd_regs"), k.get("bwd_spills"), k.get("rows_regs"), k.get("rows_spills"), k.get("error")]
                    for k in p["kernels"]]))
    hdr = ["V", "BT", "dtype", "impl", "held after fwd MiB", "peak fwd MiB", "peak bwd MiB", "fwd+bwd ms", "GB/s (3 accesses)",
           "no_grad fwd ms", "sort ms", "OOM", "error"]

    def prow(r):
        return [r["V"], r["BT"], r["dtype"], r["impl"], r.get("held_after_fwd_mib"), r.get("peak_fwd_mib"), r.get("peak_bwd_mib"),
                r.get("fwd_bwd_ms"), r.get("gbps"), r.get("nograd_fwd_ms"), r.get("sort_ms"), r.get("oom"), r.get("error")]

    L.append("\n### Shape sweep at 2^22 elements (MiB over the live x, g)\n")
    L.append(table(hdr, [prow(r) for r in p["shape_sweep"]]))
    L.append("\n### Memory sweep (bf16)\n")
    L.append(table(hdr, [prow(r) for r in p["mem_sweep"]]))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
