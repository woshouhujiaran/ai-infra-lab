"""Liger-Kernel RMSNorm (`LigerRMSNormFunction`) on an RTX 4060 Laptop.

`LigerRMSNormFunction` wraps Triton kernels that compute
`y = (x / rms(x)) * (w + offset)` with three casting modes that match HF
Transformers (llama / gemma / none), an optional Gemma-style offset, a
default in-place backward that overwrites `dY` with `dX`, and an automatic
switch between a one-row-per-program kernel and a 16-row block kernel.

1. Aliasing: graphs where the saved `dY` storage is also needed elsewhere
   (Gemma2-style two RMSNorms with a residual add in between; an aux loss
   that saved the same tensor before RMSNorm; leaf inputs; retain_graph
   twice). Upstream `in_place=True` vs `in_place=False`.
2. Numerics: Y / dX / dW error vs an fp64 reference for llama / gemma / none
   casting, offsets 0 and 1, fp32 / fp16 / bf16, against HF-style eager
   references, `F.rms_norm`, and `torch.compile`.
3. Kernel path: when the wrapper picks the row kernel vs the block kernel
   (H, T, `row_mode`), and whether the two paths agree bit-for-bit.
4. Memory and bandwidth: held / extra peak and fwd+bwd latency for
   in_place True/False vs eager / compile at Llama-3.2-1B / Qwen3-0.6B
   hidden sizes; elementwise bandwidth vs H incl. non-power-of-two.

Liger-Kernel is imported unmodified from the submodule.
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
import torch.nn.functional as F  # noqa: E402
import triton  # noqa: E402
from triton.testing import do_bench  # noqa: E402
from liger_kernel.ops.rms_norm import (  # noqa: E402
    LigerRMSNormFunction,
    _rms_norm_forward_kernel,
    calculate_settings,
    rms_norm_backward,
    rms_norm_forward,
)

SEED = 0
EPS = 1e-6
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
ALIAS_T, ALIAS_H = 256, 1024
NUM_T, NUM_H = 2048, 4096
CAST_MODES = ["llama", "gemma", "none"]
OFFSETS = [0.0, 1.0]
MEM_SHAPES = [("Qwen3-0.6B", 1024), ("Llama-3.2-1B", 2048)]
MEM_TOKENS = [2048, 8192, 16384]
BW_T = 4096
BW_HS = [512, 1024, 2048, 2049, 3072, 4096, 8192, 11008]
BW_ROUNDS = 3
PATH_CASES = [
    # (label, T, H, row_mode) — probes the wrapper's kernel-selection rule
    ("row_forced_wide", 8192, 4096, True),
    ("auto_wide_few_rows", 1024, 4096, None),   # BLOCK>256 → row
    ("auto_narrow_many_rows", 4096 * 8, 128, None),  # block path
    ("row_forced_narrow_many", 4096 * 8, 128, True),
    ("auto_narrow_few_rows", 1024, 128, None),  # few rows → row
]

torch._dynamo.config.recompile_limit = 64
torch._functorch.config.donated_buffer = False


def fmt(x, digits=4):
    if x is None or isinstance(x, str):
        return x
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x
    if abs(x) >= 1e3 or (abs(x) < 1e-3 and x != 0):
        return f"{x:.3e}"
    return round(x, digits)


def rel_l2(a, b):
    a64, b64 = a.detach().float().double(), b.detach().float().double()
    denom = b64.norm().clamp_min(1e-30)
    return (a64 - b64).norm().item() / denom.item()


def hf_llama_rms(x, w, eps, offset=0.0):
    """HF LlamaRMSNorm: variance in fp32; weight multiply in input dtype."""
    var = x.pow(2).mean(-1, keepdim=True)
    y = x * torch.rsqrt(var + eps)
    return y * (w + offset).to(x.dtype)


def hf_gemma_rms(x, w, eps, offset=1.0):
    """HF GemmaRMSNorm: cast everything to fp32, then cast y back."""
    x32 = x.float()
    var = x32.pow(2).mean(-1, keepdim=True)
    y = x32 * torch.rsqrt(var + eps)
    return (y * (w.float() + offset)).to(x.dtype)


def hf_none_rms(x, w, eps, offset=0.0):
    """No upcast: compute in the input dtype (Liger 'none')."""
    var = x.pow(2).mean(-1, keepdim=True)
    y = x * torch.rsqrt(var + eps)
    return y * (w + offset)


def fp64_rms(x, w, eps, offset=0.0):
    x64 = x.detach().double()
    w64 = w.detach().double()
    var = x64.pow(2).mean(-1, keepdim=True)
    y = x64 * torch.rsqrt(var + eps)
    return y * (w64 + offset)


def liger_apply(x, w, eps, offset, casting_mode, in_place=True, row_mode=None):
    return LigerRMSNormFunction.apply(x, w, eps, offset, casting_mode, in_place, row_mode)


# ---------------------------------------------------------------- aliasing

def alias():
    """Cases where in-place dY→dX overwrites storage other ops still need."""
    dtype = torch.bfloat16
    rows = []
    torch.manual_seed(SEED)
    x0 = torch.randn(ALIAS_T, ALIAS_H, device="cuda", dtype=dtype)
    w1 = torch.randn(ALIAS_H, device="cuda", dtype=dtype)
    w2 = torch.randn(ALIAS_H, device="cuda", dtype=dtype)

    def run_case(name, build, in_place):
        torch.manual_seed(SEED)
        try:
            out = build(in_place)
            rows.append({"case": name, "in_place": in_place, **out})
            print("alias", rows[-1], flush=True)
        except Exception as e:
            rows.append({"case": name, "in_place": in_place, "error": f"{type(e).__name__}: {e}"})
            print("alias", rows[-1], flush=True)

    # Plain single RMSNorm (baseline: should be fine either way).
    def plain(in_place):
        torch.manual_seed(SEED + 1)
        loss_w = torch.randn_like(x0)
        x = x0.clone().requires_grad_(True)
        w = w1.clone().requires_grad_(True)
        y = liger_apply(x, w, EPS, 0.0, "llama", in_place=in_place)
        (y * loss_w).sum().backward()
        xref = x0.detach().double().requires_grad_(True)
        wref = w1.detach().double().requires_grad_(True)
        yref = fp64_rms(xref, wref, EPS, 0.0)
        (yref * loss_w.double()).sum().backward()
        return {
            "dx_rel": rel_l2(x.grad, xref.grad),
            "dw_rel": rel_l2(w.grad, wref.grad),
            "stash_changed": None,
        }

    # Gemma2 pattern: y1 = rms(x); h = y1 + residual; y2 = rms(h).
    # With in_place=True the first backward overwrites dY1 which the residual
    # add still needs when propagating into y1.
    def gemma2_two_norm(in_place):
        torch.manual_seed(SEED + 2)
        residual = torch.randn_like(x0)
        loss_w = torch.randn_like(x0)
        x = x0.clone().requires_grad_(True)
        wa = w1.clone().requires_grad_(True)
        wb = w2.clone().requires_grad_(True)
        y1 = liger_apply(x, wa, EPS, 1.0, "gemma", in_place=in_place)
        h = y1 + residual
        y2 = liger_apply(h, wb, EPS, 1.0, "gemma", in_place=in_place)
        (y2 * loss_w).sum().backward()
        # fp64 reference with same graph
        xr = x0.detach().double().requires_grad_(True)
        war = w1.detach().double().requires_grad_(True)
        wbr = w2.detach().double().requires_grad_(True)
        y1r = fp64_rms(xr, war, EPS, 1.0)
        hr = y1r + residual.double()
        y2r = fp64_rms(hr, wbr, EPS, 1.0)
        (y2r * loss_w.double()).sum().backward()
        return {
            "dx_rel": rel_l2(x.grad, xr.grad),
            "dw_rel": max(rel_l2(wa.grad, war.grad), rel_l2(wb.grad, wbr.grad)),
            "stash_changed": None,
        }

    # Aux loss on the input built before RMSNorm: in-place bwd may race.
    def aux_before(in_place):
        torch.manual_seed(SEED + 3)
        loss_w = torch.randn_like(x0)
        x = x0.clone().requires_grad_(True)
        w = w1.clone().requires_grad_(True)
        aux = 0.5 * x.pow(2).sum()
        y = liger_apply(x, w, EPS, 0.0, "llama", in_place=in_place)
        (y * loss_w).sum().add(aux).backward()
        xr = x0.detach().double().requires_grad_(True)
        wr = w1.detach().double().requires_grad_(True)
        auxr = 0.5 * xr.pow(2).sum()
        yr = fp64_rms(xr, wr, EPS, 0.0)
        (yr * loss_w.double()).sum().add(auxr).backward()
        return {
            "dx_rel": rel_l2(x.grad, xr.grad),
            "dw_rel": rel_l2(w.grad, wr.grad),
            "stash_changed": None,
        }

    # Stash the forward output, then backward — in_place writes into dY which
    # is a fresh buffer from autograd, so stash of Y should be fine; stash of
    # a view into the same storage as will become dY is the interesting case.
    # Here: call backward on y while also keeping a clone of dY's base via
    # retain_graph second call.
    def retain_twice(in_place):
        torch.manual_seed(SEED + 4)
        loss_w = torch.randn_like(x0)
        x = x0.clone().requires_grad_(True)
        w = w1.clone().requires_grad_(True)
        y = liger_apply(x, w, EPS, 0.0, "llama", in_place=in_place)
        g1 = torch.autograd.grad(y, (x, w), loss_w, retain_graph=True, allow_unused=True)
        try:
            g2 = torch.autograd.grad(y, (x, w), loss_w, retain_graph=True, allow_unused=True)
            second_err = max(rel_l2(g2[0], g1[0]), rel_l2(g2[1], g1[1]))
            second_error = None
        except Exception as e:
            second_err = None
            second_error = f"{type(e).__name__}: {e}"
        xr = x0.detach().double().requires_grad_(True)
        wr = w1.detach().double().requires_grad_(True)
        yr = fp64_rms(xr, wr, EPS, 0.0)
        gr = torch.autograd.grad(yr, (xr, wr), loss_w.double())
        return {
            "dx_rel": rel_l2(g1[0], gr[0]),
            "dw_rel": rel_l2(g1[1], gr[1]),
            "second_call_rel_vs_first": second_err,
            "second_call_error": second_error,
            "stash_changed": None,
        }

    # Leaf input: in-place shouldn't touch x (dY is separate), but verify.
    def leaf_input(in_place):
        torch.manual_seed(SEED + 5)
        loss_w = torch.randn_like(x0)
        x = x0.clone().requires_grad_(True)
        w = w1.clone().requires_grad_(True)
        x_before = x.detach().clone()
        y = liger_apply(x, w, EPS, 0.0, "llama", in_place=in_place)
        y.backward(loss_w)
        changed = (x.detach() != x_before).float().mean().item()
        return {
            "dx_rel": None,
            "dw_rel": None,
            "stash_changed": {"x": round(changed, 6)},
            "grad_aliases_input": (x.grad is not None and x.grad.data_ptr() == x.data_ptr()),
        }

    for fn, name in [
        (plain, "plain"),
        (gemma2_two_norm, "gemma2_two_norm"),
        (aux_before, "aux_pow_before"),
        (retain_twice, "retain_graph_twice"),
        (leaf_input, "leaf_input"),
    ]:
        for ip in (True, False):
            run_case(name, fn, ip)
    return rows


# ---------------------------------------------------------------- numerics

def numerics():
    rows = []
    for dtype_name, dtype in DTYPES.items():
        for mode in CAST_MODES:
            for offset in OFFSETS:
                if mode == "llama" and offset == 1.0:
                    # Llama HF uses offset 0; still measure the kernel path.
                    pass
                if mode == "gemma" and offset == 0.0:
                    pass
                torch.manual_seed(SEED)
                x = torch.randn(NUM_T, NUM_H, device="cuda", dtype=dtype)
                w = torch.randn(NUM_H, device="cuda", dtype=dtype)
                dy = 0.1 * torch.randn_like(x)

                y_ref = fp64_rms(x, w, EPS, offset)
                floor_y = y_ref.to(dtype)
                # grads vs fp64
                xd = x.double().requires_grad_(True)
                wd = w.double().requires_grad_(True)
                yd = fp64_rms(xd, wd, EPS, offset)
                torch.autograd.backward(yd, dy.double())
                floor_dx = xd.grad.to(dtype)
                floor_dw = wd.grad.to(dtype)

                def measure(y, dx, dw):
                    return {
                        "y": rel_l2(y, y_ref),
                        "dx": rel_l2(dx, xd.grad),
                        "dw": rel_l2(dw, wd.grad),
                        "y_vs_floor": rel_l2(y, floor_y),
                        "dx_vs_floor": rel_l2(dx, floor_dx),
                        "dw_vs_floor": rel_l2(dw, floor_dw),
                    }

                # Liger
                xl = x.clone().requires_grad_(True)
                wl = w.clone().requires_grad_(True)
                yl = liger_apply(xl, wl, EPS, offset, mode, in_place=False)
                torch.autograd.backward(yl, dy)
                liger_err = measure(yl.detach(), xl.grad, wl.grad)

                # HF-style eager matching the casting mode
                if mode == "llama":
                    eager_fn = hf_llama_rms
                elif mode == "gemma":
                    eager_fn = hf_gemma_rms
                else:
                    eager_fn = hf_none_rms
                xe = x.clone().requires_grad_(True)
                we = w.clone().requires_grad_(True)
                ye = eager_fn(xe, we, EPS, offset)
                torch.autograd.backward(ye, dy)
                eager_err = measure(ye.detach(), xe.grad, we.grad)

                # F.rms_norm (offset folded into weight; no gemma upcast)
                xf = x.clone().requires_grad_(True)
                wf = (w + offset).clone().requires_grad_(True)
                try:
                    yf = F.rms_norm(xf, (NUM_H,), wf, EPS)
                    torch.autograd.backward(yf, dy)
                    # Map dw back: d(w+offset)/dw = 1
                    frmsnorm_err = measure(yf.detach(), xf.grad, wf.grad)
                except Exception as e:
                    frmsnorm_err = {"error": f"{type(e).__name__}: {e}"}

                # torch.compile of the matching HF eager fn
                compiled = torch.compile(eager_fn, fullgraph=False)
                xc = x.clone().requires_grad_(True)
                wc = w.clone().requires_grad_(True)
                try:
                    yc = compiled(xc, wc, EPS, offset)
                    torch.autograd.backward(yc, dy)
                    compile_err = measure(yc.detach(), xc.grad, wc.grad)
                except Exception as e:
                    compile_err = {"error": f"{type(e).__name__}: {e}"}

                # Match vs HF eager (same casting): should be near floor for llama/gemma
                match_hf = {
                    "y": rel_l2(yl.detach(), ye.detach()),
                    "dx": rel_l2(xl.grad, xe.grad),
                    "dw": rel_l2(wl.grad, we.grad),
                }

                row = {
                    "dtype": dtype_name,
                    "casting_mode": mode,
                    "offset": offset,
                    "floor": {
                        "y": rel_l2(floor_y, y_ref),
                        "dx": rel_l2(floor_dx, xd.grad),
                        "dw": rel_l2(floor_dw, wd.grad),
                    },
                    "liger": liger_err,
                    "eager_hf": eager_err,
                    "F_rms_norm": frmsnorm_err,
                    "torch_compile": compile_err,
                    "liger_vs_hf": match_hf,
                }
                rows.append(row)
                print("numerics", {k: row[k] for k in ("dtype", "casting_mode", "offset", "liger", "liger_vs_hf")}, flush=True)
                del xl, wl, yl, xe, we, ye, xd, wd
                gc.collect()
                torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------- kernel path

def kernel_paths():
    rows = []
    dtype = torch.bfloat16
    for label, T, H, row_mode in PATH_CASES:
        torch.manual_seed(SEED)
        x = torch.randn(T, H, device="cuda", dtype=dtype)
        w = torch.randn(H, device="cuda", dtype=dtype)
        block, warps = calculate_settings(H)
        # What the wrapper will pick
        n_rows = T
        use_row = block > 256 or n_rows < 4096 * 8 or row_mode
        path = "row" if use_row else "block"
        y, X_saved, RSTD, BLOCK_SIZE, num_warps, cmode = rms_norm_forward(
            x, w, EPS, 0.0, "llama", row_mode
        )
        # Compare against forced opposite path when possible
        other_mode = True if not use_row else None
        # Always get an explicit row-mode reference
        y_row, _, _, _, _, _ = rms_norm_forward(x, w, EPS, 0.0, "llama", True)
        bit_identical_to_row = bool(torch.equal(y, y_row))
        # Backward agreement
        dy = torch.randn_like(y)
        dX, dW = rms_norm_backward(
            dy, X_saved, w, RSTD, 0.0, cmode, BLOCK_SIZE, num_warps, False, row_mode, needs_dw=True
        )
        y2, X2, R2, B2, W2, c2 = rms_norm_forward(x, w, EPS, 0.0, "llama", True)
        dX2, dW2 = rms_norm_backward(dy, X2, w, R2, 0.0, c2, B2, W2, False, True, needs_dw=True)
        row = {
            "label": label,
            "T": T,
            "H": H,
            "row_mode": row_mode,
            "BLOCK_SIZE": BLOCK_SIZE,
            "num_warps": num_warps,
            "selected_path": path,
            "y_bit_identical_to_row": bit_identical_to_row,
            "y_rel_vs_row": rel_l2(y, y_row),
            "dx_rel_vs_row": rel_l2(dX, dX2),
            "dw_rel_vs_row": rel_l2(dW, dW2),
            "dx_bit_identical_to_row": bool(torch.equal(dX, dX2)),
            "dw_bit_identical_to_row": bool(torch.equal(dW, dW2)),
        }
        rows.append(row)
        print("path", row, flush=True)
        del x, w, y, dy, dX, dW
        gc.collect()
        torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------- memory / latency

def memory_bench():
    rows = []
    dtype = torch.bfloat16
    for name, H in MEM_SHAPES:
        for T in MEM_TOKENS:
            torch.manual_seed(SEED)
            x_base = torch.randn(T, H, device="cuda", dtype=dtype)
            w_base = torch.randn(H, device="cuda", dtype=dtype)
            dy = 0.1 * torch.randn_like(x_base)
            act_mib = round(x_base.numel() * x_base.element_size() / 2**20, 2)
            cell = {"shape": name, "T": T, "H": H, "act_MiB": act_mib}

            def one(label, fwd_fn, in_place_flag=None):
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                base = torch.cuda.memory_allocated()
                x = x_base.clone().requires_grad_(True)
                w = w_base.clone().requires_grad_(True)
                # account for inputs in base
                base_with_in = torch.cuda.memory_allocated()
                y = fwd_fn(x, w)
                torch.cuda.synchronize()
                held = torch.cuda.memory_allocated() - base_with_in
                torch.autograd.backward(y, dy)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() - base_with_in
                # latency
                def bench_once():
                    x2 = x_base.clone().requires_grad_(True)
                    w2 = w_base.clone().requires_grad_(True)
                    y2 = fwd_fn(x2, w2)
                    torch.autograd.backward(y2, dy)

                ms = do_bench(bench_once, return_mode="median")
                cell[label] = {
                    "held_after_fwd_MiB": round(held / 2**20, 1),
                    "extra_peak_MiB": round(peak / 2**20, 1),
                    "fwd_bwd_ms": round(ms, 3),
                    "in_place": in_place_flag,
                }
                del x, w, y
                gc.collect()
                torch.cuda.empty_cache()

            one("liger_inplace", lambda x, w: liger_apply(x, w, EPS, 0.0, "llama", True), True)
            one("liger_oop", lambda x, w: liger_apply(x, w, EPS, 0.0, "llama", False), False)
            one("eager_hf", lambda x, w: hf_llama_rms(x, w, EPS, 0.0))
            compiled = torch.compile(hf_llama_rms, fullgraph=False)
            one("torch_compile", lambda x, w: compiled(x, w, EPS, 0.0))
            one("F_rms_norm", lambda x, w: F.rms_norm(x, (H,), w, EPS))
            rows.append(cell)
            print("mem", {k: cell[k] for k in cell if k != "act_MiB"}, flush=True)
    return rows


# ---------------------------------------------------------------- bandwidth

def bandwidth():
    rows = []
    kernels = []
    dtype = torch.bfloat16
    for H in BW_HS:
        torch.manual_seed(SEED)
        x = torch.randn(BW_T, H, device="cuda", dtype=dtype)
        w = torch.randn(H, device="cuda", dtype=dtype)
        dy = torch.randn_like(x)
        nbytes = x.numel() * x.element_size()
        block, warps = calculate_settings(H)
        compiled = torch.compile(lambda a, b: hf_llama_rms(a, b, EPS, 0.0), fullgraph=False)

        # Warmup compile
        xc = x.clone().requires_grad_(True)
        wc = w.clone().requires_grad_(True)
        yc = compiled(xc, wc)
        torch.autograd.grad(yc, (xc, wc), dy, retain_graph=True)
        del xc, wc, yc

        xg = x.clone().requires_grad_(True)
        wg = w.clone().requires_grad_(True)
        yg = hf_llama_rms(xg, wg, EPS, 0.0)
        xl = x.clone().requires_grad_(True)
        wl = w.clone().requires_grad_(True)
        yl = liger_apply(xl, wl, EPS, 0.0, "llama", False)
        xo = x.clone().requires_grad_(True)
        wo = w.clone().requires_grad_(True)
        yo = liger_apply(xo, wo, EPS, 0.0, "llama", True)
        xf = x.clone().requires_grad_(True)
        wf = w.clone().requires_grad_(True)
        yf = F.rms_norm(xf, (H,), wf, EPS)
        xcc = x.clone().requires_grad_(True)
        wcc = w.clone().requires_grad_(True)
        ycc = compiled(xcc, wcc)

        # Low-level forward/backward for pure kernel bandwidth
        Y, Xs, RSTD, BS, NW, CM = rms_norm_forward(x, w, EPS, 0.0, "llama", True)

        fns = {
            "copy": (lambda: x.clone(), 1),
            "fwd_liger": (lambda: rms_norm_forward(x, w, EPS, 0.0, "llama", True)[0], 2),
            "fwd_eager": (lambda: hf_llama_rms(x, w, EPS, 0.0), 2),
            "fwd_compile": (lambda: compiled(x, w), 2),
            "fwd_F": (lambda: F.rms_norm(x, (H,), w, EPS), 2),
            "bwd_liger_oop": (
                lambda: rms_norm_backward(dy, Xs, w, RSTD, 0.0, CM, BS, NW, False, True, needs_dw=True),
                3,
            ),
            "bwd_liger_inplace": (
                lambda: rms_norm_backward(dy.clone(), Xs, w, RSTD, 0.0, CM, BS, NW, True, True, needs_dw=True),
                3,
            ),
            "bwd_eager": (lambda: torch.autograd.grad(yg, (xg, wg), dy, retain_graph=True), 3),
            "bwd_compile": (lambda: torch.autograd.grad(ycc, (xcc, wcc), dy, retain_graph=True), 3),
            "bwd_F": (lambda: torch.autograd.grad(yf, (xf, wf), dy, retain_graph=True), 3),
        }
        do_bench(fns["copy"][0], rep=200)
        times = {k: [] for k in fns}
        for _ in range(BW_ROUNDS):
            for k, (fn, _) in fns.items():
                times[k].append(do_bench(fn, return_mode="median"))
        row = {"T": BW_T, "H": H, "BLOCK_SIZE": block, "num_warps": warps,
               "lane_util": round(H / block, 3), "MiB_per_tensor": round(nbytes / 2**20, 1)}
        for k, (_, mult) in fns.items():
            ms = sorted(times[k])[len(times[k]) // 2]
            row[k + "_ms"] = round(ms, 4)
            row[k + "_gbps"] = round(mult * nbytes * 1e-9 / (ms * 1e-3), 1)
        for k in ("fwd_liger", "fwd_compile", "fwd_F"):
            row[k + "_vs_eager"] = round(row["fwd_eager_ms"] / max(row[k + "_ms"], 1e-9), 2)
        for k in ("bwd_liger_oop", "bwd_liger_inplace", "bwd_compile", "bwd_F"):
            row[k + "_vs_eager"] = round(row["bwd_eager_ms"] / max(row[k + "_ms"], 1e-9), 2)
        rows.append(row)
        print("bw", row, flush=True)

        # Kernel resources
        try:
            kfwd = _rms_norm_forward_kernel.warmup(
                Y, Y.stride(0), x, x.stride(0), w, w.stride(0), RSTD, RSTD.stride(0),
                H, EPS, 0.0, 0, True, BLOCK_SIZE=block, num_warps=warps, grid=(1,),
            )
            kfwd._init_handles()
            kernels.append({
                "H": H, "kernel": "fwd_row", "BLOCK_SIZE": block, "num_warps": warps,
                "n_regs": kfwd.n_regs, "n_spills": kfwd.n_spills, "shared": kfwd.metadata.shared,
            })
        except Exception as e:
            kernels.append({"H": H, "kernel": "fwd_row", "error": f"{type(e).__name__}: {e}"})

        del x, w, dy, xg, wg, yg, xl, wl, yl, xo, wo, yo, xf, wf, yf, xcc, wcc, ycc, Y, Xs, RSTD
        gc.collect()
        torch.cuda.empty_cache()
    return rows, kernels


# ---------------------------------------------------------------- report

def upstream_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(LIGER_DIR), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()


def write_md(result):
    gpu = result["gpu"]
    lines = [
        f"Liger-Kernel {result['upstream_commit'][:10]}, triton-windows {triton.__version__}, "
        f"torch {torch.__version__}, {gpu.get('name', 'cuda')} "
        f"(memory cap {gpu.get('memory_fraction_cap')}), runtime {result['runtime_s']} s, "
        f"peak allocated {result['peak_memory_mib']} MiB",
        "",
        f"## 1. Aliasing (bf16, T = {ALIAS_T}, H = {ALIAS_H}; rel L2 vs fp64; "
        "`in_place` overwrites dY with dX)",
        "",
        "| case | in_place | dx rel | dw rel | 2nd call vs 1st | stash changed | grad aliases input | error |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in result["alias"]:
        lines.append(
            f"| {r['case']} | {r['in_place']} | {fmt(r.get('dx_rel'))} | {fmt(r.get('dw_rel'))} | "
            f"{fmt(r.get('second_call_rel_vs_first'))} | {r.get('stash_changed')} | "
            f"{r.get('grad_aliases_input', '')} | "
            f"{(r.get('error') or r.get('second_call_error') or '')} |"
        )
    lines += [
        "",
        f"## 2. Error vs fp64 (T = {NUM_T}, H = {NUM_H}; rel L2; "
        "`floor` = fp64 result rounded to the dtype; `liger_vs_hf` = Liger vs matching HF casting)",
        "",
        "| dtype | mode | offset | out | floor | liger | eager_hf | F.rms_norm | compile | liger vs hf |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in result["numerics"]:
        for o, key in (("y", "y"), ("dx", "dx"), ("dw", "dw")):
            fr = r["F_rms_norm"].get(key, r["F_rms_norm"].get("error", ""))
            cr = r["torch_compile"].get(key, r["torch_compile"].get("error", ""))
            lines.append(
                f"| {r['dtype']} | {r['casting_mode']} | {r['offset']} | {o} | "
                f"{fmt(r['floor'][key])} | {fmt(r['liger'][key])} | {fmt(r['eager_hf'][key])} | "
                f"{fmt(fr) if not isinstance(fr, str) else fr} | "
                f"{fmt(cr) if not isinstance(cr, str) else cr} | "
                f"{fmt(r['liger_vs_hf'][key])} |"
            )
    lines += [
        "",
        "## 3. Kernel path selection (bf16; wrapper picks row if "
        "`BLOCK_SIZE > 256 or T < 32768 or row_mode`)",
        "",
        "| label | T | H | row_mode | BLOCK | warps | path | y == row | y rel | dx rel | dw rel |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in result["paths"]:
        lines.append(
            f"| {r['label']} | {r['T']} | {r['H']} | {r['row_mode']} | {r['BLOCK_SIZE']} | "
            f"{r['num_warps']} | {r['selected_path']} | {r['y_bit_identical_to_row']} | "
            f"{fmt(r['y_rel_vs_row'])} | {fmt(r['dx_rel_vs_row'])} | {fmt(r['dw_rel_vs_row'])} |"
        )
    names = ["liger_inplace", "liger_oop", "eager_hf", "torch_compile", "F_rms_norm"]
    lines += [
        "",
        "## 4. Memory + latency (bf16; held = after forward incl. output; "
        "extra peak = max during fwd+bwd above inputs; do_bench median)",
        "",
    ]
    for metric, title in (
        ("held_after_fwd_MiB", "held after forward (MiB)"),
        ("extra_peak_MiB", "extra peak (MiB)"),
        ("fwd_bwd_ms", "fwd + bwd (ms)"),
    ):
        lines += [
            f"### {title}",
            "",
            "| shape | T | H | " + " | ".join(names) + " |",
            "|---|---|---|" + "---|" * len(names),
        ]
        for r in result["memory"]:
            cells = [str(r[n].get(metric, r[n].get("error"))) for n in names]
            lines.append(f"| {r['shape']} | {r['T']} | {r['H']} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        f"## 5. Bandwidth (bf16, T = {BW_T}; fwd 2·T·H·2 bytes, bwd 3·T·H·2 bytes; "
        "do_bench median of 3 rounds; `xE` = speedup vs eager HF)",
        "",
        "| H | BLOCK | lanes | copy | fwd liger | fwd eager | fwd compile | fwd F | "
        "bwd liger oop | bwd liger in-place | bwd eager | bwd compile | bwd F | "
        "fwd liger xE | bwd liger oop xE |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in result["bandwidth"]:
        lines.append(
            f"| {r['H']} | {r['BLOCK_SIZE']} | {r['lane_util']} | {r['copy_gbps']} | "
            f"{r['fwd_liger_gbps']} | {r['fwd_eager_gbps']} | {r['fwd_compile_gbps']} | "
            f"{r['fwd_F_gbps']} | {r['bwd_liger_oop_gbps']} | {r['bwd_liger_inplace_gbps']} | "
            f"{r['bwd_eager_gbps']} | {r['bwd_compile_gbps']} | {r['bwd_F_gbps']} | "
            f"{r['fwd_liger_vs_eager']} | {r['bwd_liger_oop_vs_eager']} |"
        )
    lines += [
        "",
        "### Kernel resources (row forward)",
        "",
        "| H | kernel | BLOCK | warps | regs/thread | spills (bytes) | shared |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in result["kernels"]:
        if "error" in k:
            lines.append(f"| {k['H']} | {k['kernel']} |  |  |  |  | {k['error']} |")
        else:
            lines.append(
                f"| {k['H']} | {k['kernel']} | {k['BLOCK_SIZE']} | {k['num_warps']} | "
                f"{k['n_regs']} | {k['n_spills']} | {k.get('shared', '')} |"
            )
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    t_start = time.time()
    torch.manual_seed(SEED)
    alias_rows = alias()
    numeric_rows = numerics()
    path_rows = kernel_paths()
    mem_rows = memory_bench()
    bw_rows, kernel_rows = bandwidth()
    result = {
        "experiment": HERE.name,
        "upstream": "linkedin/Liger-Kernel",
        "upstream_commit": upstream_commit(),
        "component": "src/liger_kernel/ops/rms_norm.py (LigerRMSNormFunction)",
        "triton": triton.__version__,
        "gpu": GPU,
        "alias": alias_rows,
        "numerics": numeric_rows,
        "paths": path_rows,
        "memory": mem_rows,
        "bandwidth": bw_rows,
        "kernels": kernel_rows,
        "peak_memory_mib": gpu_guard.peak_memory_mib(),
        "runtime_s": round(time.time() - t_start, 1),
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_md(result)


if __name__ == "__main__":
    main()
