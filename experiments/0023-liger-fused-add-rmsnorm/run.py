"""Liger-Kernel fused residual-add + RMSNorm on an RTX 4060 Laptop.

`LigerFusedAddRMSNormFunction` implements the decoder-layer pattern
    hidden = residual + hidden; residual = hidden; hidden = rmsnorm(hidden)
in one Triton forward (returns Y and the post-add residual S) and a fused
backward that folds `dS_out` into `dX` (= `dR`). Casting modes match the plain
Liger RMSNorm (`llama` / `gemma` / `none`); `in_place` overwrites `dY` with `dX`.

1. Output branches: loss on both Y and S / Y only / S only. On current torch,
   unused Function outputs arrive as **zero tensors** (not None), so the
   Triton's `has_dS_out = dS_out is not None` guard is always true in practice
   and a None would still crash on `.view` before that check.
2. Aliasing: `in_place` True/False with reused `dY`, and whether `dX` and `dR`
   are the same storage (`return dX, dX, dW`).
3. Numerics: Y / S / dX / dR / dW vs fp64 and HF-style eager refs for the three
   casting modes and offsets 0 / 1 in fp32 / fp16 / bf16.
4. Memory and latency: fused vs separate `add` + `LigerRMSNormFunction` /
   eager HF / `torch.compile` at Qwen3-0.6B / Llama-3.2-1B shapes.
5. Bandwidth vs H (incl. non-power-of-two).

Liger-Kernel is imported unmodified from the submodule.
"""
import gc
import json
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
from triton.testing import do_bench  # noqa: E402
from liger_kernel.ops.fused_add_rms_norm import (  # noqa: E402
    LigerFusedAddRMSNormFunction,
    fused_add_rms_norm_backward,
    fused_add_rms_norm_forward,
)
from liger_kernel.ops.rms_norm import LigerRMSNormFunction  # noqa: E402
from liger_kernel.ops.utils import calculate_settings  # noqa: E402
from liger_kernel.transformers.functional import liger_fused_add_rms_norm  # noqa: E402

SEED = 0
EPS = 1e-6
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
BRANCH_T, BRANCH_H = 256, 1024
ALIAS_T, ALIAS_H = 256, 1024
NUM_T, NUM_H = 2048, 4096
CAST_MODES = ["llama", "gemma", "none"]
OFFSETS = [0.0, 1.0]
MEM_SHAPES = [("Qwen3-0.6B", 1024), ("Llama-3.2-1B", 2048)]
MEM_TOKENS = [2048, 8192, 16384]
BW_T = 4096
BW_HS = [512, 1024, 2048, 2049, 3072, 4096, 8192, 11008]
BW_ROUNDS = 3

torch._dynamo.config.recompile_limit = 64
torch._functorch.config.donated_buffer = False


def fmt(x, digits=4):
    if x is None:
        return x
    if isinstance(x, str):
        return x.splitlines()[0][:80]
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x
    if abs(x) >= 1e3 or (abs(x) < 1e-3 and x != 0):
        return f"{x:.3e}"
    return round(x, digits)


def short_err(err):
    if not err:
        return ""
    return str(err).splitlines()[0][:80]


def rel_l2(a, b):
    a64, b64 = a.detach().float().double(), b.detach().float().double()
    denom = b64.norm().clamp_min(1e-30)
    return (a64 - b64).norm().item() / denom.item()


def fp64_fused(x, r, w, eps, offset=0.0):
    s = x.to(torch.float64) + r.to(torch.float64)
    var = s.pow(2).mean(-1, keepdim=True)
    y = s * torch.rsqrt(var + eps)
    return y * (w.to(torch.float64) + offset), s


def hf_llama_fused(x, r, w, eps, offset=0.0):
    s = x + r
    s32 = s.to(torch.float32)
    var = s32.pow(2).mean(-1, keepdim=True)
    y = s32 * torch.rsqrt(var + eps)
    return (w + offset).to(x.dtype) * y.to(x.dtype), s


def hf_gemma_fused(x, r, w, eps, offset=1.0):
    s = x + r
    y = s.float() * torch.rsqrt(s.float().pow(2).mean(-1, keepdim=True) + eps)
    return (y * (w.float() + offset)).to(x.dtype), s


def hf_none_fused(x, r, w, eps, offset=0.0):
    s = x + r
    var = s.pow(2).mean(-1, keepdim=True)
    y = s * torch.rsqrt(var + eps)
    return y * (w + offset), s


def liger_apply(x, r, w, eps, offset, casting_mode, in_place=False):
    return LigerFusedAddRMSNormFunction.apply(x, r, w, eps, offset, casting_mode, in_place)


# ---------------------------------------------------------------- defaults / identity

def api_defaults():
    """Module / Function / functional disagree on the in_place default."""
    from liger_kernel.transformers.fused_add_rms_norm import LigerFusedAddRMSNorm
    import inspect

    fn_default = inspect.signature(LigerFusedAddRMSNormFunction.forward).parameters["in_place"].default
    functional_default = inspect.signature(liger_fused_add_rms_norm).parameters["in_place"].default
    mod = LigerFusedAddRMSNorm(16)
    return {
        "function_forward_default": fn_default,
        "functional_default": functional_default,
        "module_default": mod.in_place,
        "docstring_claims_true": "defaults to `True`" in (LigerFusedAddRMSNormFunction.__doc__ or ""),
    }


# ---------------------------------------------------------------- output branches

def branches():
    """both / norm-only / residual-only; None grads hit the Triton .view() path."""
    rows = []
    dtype = torch.bfloat16
    for branch in ("both", "norm_only", "residual_only"):
        for in_place in (False, True):
            torch.manual_seed(SEED)
            x0 = torch.randn(BRANCH_T, BRANCH_H, device="cuda", dtype=dtype)
            r0 = torch.randn(BRANCH_T, BRANCH_H, device="cuda", dtype=dtype)
            w0 = torch.randn(BRANCH_H, device="cuda", dtype=dtype)
            gy = torch.randn_like(x0)
            gs = torch.randn_like(x0)
            row = {"branch": branch, "in_place": in_place}
            try:
                x = x0.clone().requires_grad_(True)
                r = r0.clone().requires_grad_(True)
                w = w0.clone().requires_grad_(True)
                y, s = liger_apply(x, r, w, EPS, 0.0, "llama", in_place=in_place)
                if branch == "both":
                    torch.autograd.backward((y, s), (gy.clone(), gs.clone()))
                elif branch == "norm_only":
                    y.backward(gy.clone())
                else:
                    s.backward(gs.clone())

                # Reference with both grads when possible; residual-only has dW = 0.
                xr = x0.double().requires_grad_(True)
                rr = r0.double().requires_grad_(True)
                wr = w0.double().requires_grad_(True)
                yr, sr = fp64_fused(xr, rr, wr, EPS, 0.0)
                if branch == "both":
                    torch.autograd.backward((yr, sr), (gy.double(), gs.double()))
                elif branch == "norm_only":
                    yr.backward(gy.double())
                else:
                    sr.backward(gs.double())

                row.update({
                    "ok": True,
                    "dx_rel": rel_l2(x.grad, xr.grad),
                    "dr_rel": rel_l2(r.grad, rr.grad),
                    "dw_rel": rel_l2(w.grad, wr.grad) if wr.grad is not None else rel_l2(w.grad, torch.zeros_like(w)),
                    "dx_is_dr_same_tensor": x.grad.data_ptr() == r.grad.data_ptr(),
                    "dx_equals_dr": bool(torch.equal(x.grad, r.grad)),
                    "y_held": True,
                    "s_held": True,
                })
            except Exception as e:
                row.update({
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                })
            rows.append(row)
            print("branch", row, flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------- aliasing

def alias():
    rows = []
    dtype = torch.bfloat16
    torch.manual_seed(SEED)
    x0 = torch.randn(ALIAS_T, ALIAS_H, device="cuda", dtype=dtype)
    r0 = torch.randn(ALIAS_T, ALIAS_H, device="cuda", dtype=dtype)
    w0 = torch.randn(ALIAS_H, device="cuda", dtype=dtype)

    def run_case(name, build, in_place):
        try:
            out = build(in_place)
            rows.append({"case": name, "in_place": in_place, **out})
        except Exception as e:
            rows.append({"case": name, "in_place": in_place, "error": f"{type(e).__name__}: {e}"})
        print("alias", rows[-1], flush=True)

    def plain(in_place):
        torch.manual_seed(SEED + 1)
        gy = torch.randn_like(x0)
        gs = torch.randn_like(x0)
        # Clone grad buffers: in_place=True overwrites the caller's dY.
        gy_ref, gs_ref = gy.clone(), gs.clone()
        x = x0.clone().requires_grad_(True)
        r = r0.clone().requires_grad_(True)
        w = w0.clone().requires_grad_(True)
        y, s = liger_apply(x, r, w, EPS, 0.0, "llama", in_place=in_place)
        torch.autograd.backward((y, s), (gy, gs))
        xr = x0.double().requires_grad_(True)
        rr = r0.double().requires_grad_(True)
        wr = w0.double().requires_grad_(True)
        yr, sr = fp64_fused(xr, rr, wr, EPS, 0.0)
        torch.autograd.backward((yr, sr), (gy_ref.double(), gs_ref.double()))
        return {
            "dx_rel": rel_l2(x.grad, xr.grad),
            "dr_rel": rel_l2(r.grad, rr.grad),
            "dw_rel": rel_l2(w.grad, wr.grad),
            "dx_is_dr_same_tensor": x.grad.data_ptr() == r.grad.data_ptr(),
            "caller_dY_overwritten": not torch.equal(gy, gy_ref),
        }

    def retain_reuse_dy(in_place):
        torch.manual_seed(SEED + 2)
        gy = torch.randn_like(x0)
        gs = torch.randn_like(x0)
        x = x0.clone().requires_grad_(True)
        r = r0.clone().requires_grad_(True)
        w = w0.clone().requires_grad_(True)
        y, s = liger_apply(x, r, w, EPS, 0.0, "llama", in_place=in_place)
        g1 = torch.autograd.grad(y, (x, r, w), gy.clone(), retain_graph=True, allow_unused=True)
        # Fresh clones: second call should match.
        g2 = torch.autograd.grad(y, (x, r, w), gy.clone(), retain_graph=True, allow_unused=True)
        second_vs_first = max(rel_l2(g2[0], g1[0]), rel_l2(g2[1], g1[1]), rel_l2(g2[2], g1[2]))
        # Reuse caller's dY buffer (the in_place footgun). Pass None for unused S.
        try:
            g_reuse = torch.autograd.grad(y, (x, r, w), gy, retain_graph=True, allow_unused=True)
            g_reuse2 = torch.autograd.grad(y, (x, r, w), gy, retain_graph=True, allow_unused=True)
            reuse_first_vs_clean = max(
                rel_l2(g_reuse[0], g1[0]), rel_l2(g_reuse[1], g1[1]), rel_l2(g_reuse[2], g1[2])
            )
            reuse_second_vs_first = max(
                rel_l2(g_reuse2[0], g_reuse[0]),
                rel_l2(g_reuse2[1], g_reuse[1]),
                rel_l2(g_reuse2[2], g_reuse[2]),
            )
            reuse_error = None
        except Exception as e:
            reuse_first_vs_clean = None
            reuse_second_vs_first = None
            reuse_error = f"{type(e).__name__}: {e}"
        return {
            "dx_rel": None,
            "dr_rel": None,
            "dw_rel": None,
            "second_call_rel_vs_first": second_vs_first,
            "reuse_dY_first_vs_clean": reuse_first_vs_clean,
            "reuse_dY_second_vs_first": reuse_second_vs_first,
            "reuse_error": reuse_error,
            "dx_is_dr_same_tensor": g1[0].data_ptr() == g1[1].data_ptr(),
        }

    def d_out_same_object(in_place):
        """Low-level backward returns (dX, dX, dW) — same tensor twice."""
        torch.manual_seed(SEED + 3)
        x = x0.clone()
        r = r0.clone()
        w = w0.clone()
        y, s, rstd, bs, nw, ns, cm = fused_add_rms_norm_forward(x, r, w, EPS, 0.0, "llama")
        dy = torch.randn_like(y)
        ds = torch.randn_like(s)
        dy_in = dy  # in_place writes into this exact object
        dx, dr, dw = fused_add_rms_norm_backward(
            dy_in, ds, s, w, rstd, 0.0, cm, bs, nw, ns, in_place, needs_dw=True
        )
        return {
            "dx_is_dr_same_tensor": dx.data_ptr() == dr.data_ptr(),
            "dx_equals_dr": bool(torch.equal(dx, dr)),
            "dx_aliases_dY_input": dx.data_ptr() == dy_in.data_ptr(),
        }

    for fn, name in [
        (plain, "plain_both"),
        (retain_reuse_dy, "retain_reuse_dY"),
        (d_out_same_object, "bwd_returns_dX_twice"),
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
                torch.manual_seed(SEED)
                x = torch.randn(NUM_T, NUM_H, device="cuda", dtype=dtype)
                r = torch.randn(NUM_T, NUM_H, device="cuda", dtype=dtype)
                w = torch.randn(NUM_H, device="cuda", dtype=dtype)
                gy = 0.1 * torch.randn_like(x)
                gs = 0.1 * torch.randn_like(x)

                y_ref, s_ref = fp64_fused(x, r, w, EPS, offset)
                floor_y, floor_s = y_ref.to(dtype), s_ref.to(dtype)
                xd = x.double().requires_grad_(True)
                rd = r.double().requires_grad_(True)
                wd = w.double().requires_grad_(True)
                yd, sd = fp64_fused(xd, rd, wd, EPS, offset)
                torch.autograd.backward((yd, sd), (gy.double(), gs.double()))
                floor_dx = xd.grad.to(dtype)
                floor_dr = rd.grad.to(dtype)
                floor_dw = wd.grad.to(dtype)

                def measure(y, s, dx, dr, dw):
                    return {
                        "y": rel_l2(y, y_ref),
                        "s": rel_l2(s, s_ref),
                        "dx": rel_l2(dx, xd.grad),
                        "dr": rel_l2(dr, rd.grad),
                        "dw": rel_l2(dw, wd.grad),
                        "y_vs_floor": rel_l2(y, floor_y),
                        "dx_vs_floor": rel_l2(dx, floor_dx),
                        "dw_vs_floor": rel_l2(dw, floor_dw),
                    }

                xl = x.clone().requires_grad_(True)
                rl = r.clone().requires_grad_(True)
                wl = w.clone().requires_grad_(True)
                yl, sl = liger_apply(xl, rl, wl, EPS, offset, mode, in_place=False)
                torch.autograd.backward((yl, sl), (gy, gs))
                liger_err = measure(yl.detach(), sl.detach(), xl.grad, rl.grad, wl.grad)

                if mode == "llama":
                    eager_fn = hf_llama_fused
                elif mode == "gemma":
                    eager_fn = hf_gemma_fused
                else:
                    eager_fn = hf_none_fused
                xe = x.clone().requires_grad_(True)
                re = r.clone().requires_grad_(True)
                we = w.clone().requires_grad_(True)
                ye, se = eager_fn(xe, re, we, EPS, offset)
                torch.autograd.backward((ye, se), (gy, gs))
                eager_err = measure(ye.detach(), se.detach(), xe.grad, re.grad, we.grad)

                off = offset

                def _eager_closed(a, b, c, _fn=eager_fn, _off=off):
                    return _fn(a, b, c, EPS, _off)

                compiled = torch.compile(_eager_closed, fullgraph=False)
                xc = x.clone().requires_grad_(True)
                rc = r.clone().requires_grad_(True)
                wc = w.clone().requires_grad_(True)
                try:
                    yc, sc = compiled(xc, rc, wc)
                    torch.autograd.backward((yc, sc), (gy, gs))
                    compile_err = measure(yc.detach(), sc.detach(), xc.grad, rc.grad, wc.grad)
                except Exception as e:
                    compile_err = {"error": f"{type(e).__name__}: {e}"}

                match_hf = {
                    "y": rel_l2(yl.detach(), ye.detach()),
                    "s": rel_l2(sl.detach(), se.detach()),
                    "dx": rel_l2(xl.grad, xe.grad),
                    "dr": rel_l2(rl.grad, re.grad),
                    "dw": rel_l2(wl.grad, we.grad),
                }
                row = {
                    "dtype": dtype_name,
                    "casting_mode": mode,
                    "offset": offset,
                    "floor": {
                        "y": rel_l2(floor_y, y_ref),
                        "s": rel_l2(floor_s, s_ref),
                        "dx": rel_l2(floor_dx, xd.grad),
                        "dr": rel_l2(floor_dr, rd.grad),
                        "dw": rel_l2(floor_dw, wd.grad),
                    },
                    "liger": liger_err,
                    "eager_hf": eager_err,
                    "torch_compile": compile_err,
                    "liger_vs_hf": match_hf,
                    "dx_equals_dr_liger": bool(torch.equal(xl.grad, rl.grad)),
                }
                rows.append(row)
                print(
                    "numerics",
                    {k: row[k] for k in ("dtype", "casting_mode", "offset", "liger", "liger_vs_hf")},
                    flush=True,
                )
                del xl, rl, wl, yl, sl, xe, re, we, ye, se, xd, rd, wd
                gc.collect()
                torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------- memory / latency

def memory_bench():
    rows = []
    dtype = torch.bfloat16

    def _llama_closed(a, b, c):
        return hf_llama_fused(a, b, c, EPS, 0.0)

    compiled = torch.compile(_llama_closed, fullgraph=False)
    for name, H in MEM_SHAPES:
        for T in MEM_TOKENS:
            torch.manual_seed(SEED)
            x_base = torch.randn(T, H, device="cuda", dtype=dtype)
            r_base = torch.randn(T, H, device="cuda", dtype=dtype)
            w_base = torch.randn(H, device="cuda", dtype=dtype)
            gy = 0.1 * torch.randn_like(x_base)
            gs = 0.1 * torch.randn_like(x_base)
            act_mib = round(x_base.numel() * x_base.element_size() / 2**20, 2)
            cell = {"shape": name, "T": T, "H": H, "act_MiB": act_mib}

            def one(label, fwd_fn, in_place_flag=None):
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                x = x_base.clone().requires_grad_(True)
                r = r_base.clone().requires_grad_(True)
                w = w_base.clone().requires_grad_(True)
                base = torch.cuda.memory_allocated()
                y, s = fwd_fn(x, r, w)
                torch.cuda.synchronize()
                held = torch.cuda.memory_allocated() - base
                torch.autograd.backward((y, s), (gy, gs))
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() - base

                def bench_once():
                    x2 = x_base.clone().requires_grad_(True)
                    r2 = r_base.clone().requires_grad_(True)
                    w2 = w_base.clone().requires_grad_(True)
                    y2, s2 = fwd_fn(x2, r2, w2)
                    torch.autograd.backward((y2, s2), (gy, gs))

                ms = do_bench(bench_once, return_mode="median")
                cell[label] = {
                    "held_after_fwd_MiB": round(held / 2**20, 1),
                    "extra_peak_MiB": round(peak / 2**20, 1),
                    "fwd_bwd_ms": round(ms, 3),
                    "in_place": in_place_flag,
                }
                del x, r, w, y, s
                gc.collect()
                torch.cuda.empty_cache()

            one(
                "liger_fused_inplace",
                lambda x, r, w: liger_apply(x, r, w, EPS, 0.0, "llama", True),
                True,
            )
            one(
                "liger_fused_oop",
                lambda x, r, w: liger_apply(x, r, w, EPS, 0.0, "llama", False),
                False,
            )

            def separate_liger(x, r, w):
                s = x + r
                y = LigerRMSNormFunction.apply(s, w, EPS, 0.0, "llama", False, None)
                return y, s

            one("separate_add_liger_rms", separate_liger)
            one("eager_hf", lambda x, r, w: hf_llama_fused(x, r, w, EPS, 0.0))
            try:
                # Warmup compile once per cell.
                _ = compiled(x_base.clone().requires_grad_(True), r_base.clone().requires_grad_(True), w_base.clone())
                one("torch_compile", compiled)
            except Exception as e:
                cell["torch_compile"] = {"error": f"{type(e).__name__}: {e}"}
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
        r = torch.randn(BW_T, H, device="cuda", dtype=dtype)
        w = torch.randn(H, device="cuda", dtype=dtype)
        gy = torch.randn_like(x)
        gs = torch.randn_like(x)
        nbytes = x.numel() * x.element_size()
        block, warps = calculate_settings(H)

        def _llama_closed(a, b, c):
            return hf_llama_fused(a, b, c, EPS, 0.0)

        compiled = torch.compile(_llama_closed, fullgraph=False)
        xc = x.clone().requires_grad_(True)
        rc = r.clone().requires_grad_(True)
        wc = w.clone().requires_grad_(True)
        try:
            yc, sc = compiled(xc, rc, wc)
            torch.autograd.grad((yc, sc), (xc, rc, wc), (gy, gs), retain_graph=True)
            del xc, rc, wc, yc, sc
            compile_ok = True
            compile_err = None
        except Exception as e:
            compile_ok = False
            compile_err = f"{type(e).__name__}: {e}"
            del xc, rc, wc

        xg = x.clone().requires_grad_(True)
        rg = r.clone().requires_grad_(True)
        wg = w.clone().requires_grad_(True)
        yg, sg = hf_llama_fused(xg, rg, wg, EPS, 0.0)
        xl = x.clone().requires_grad_(True)
        rl = r.clone().requires_grad_(True)
        wl = w.clone().requires_grad_(True)
        yl, sl = liger_apply(xl, rl, wl, EPS, 0.0, "llama", False)

        Y, S, RSTD, BS, NW, NS, CM = fused_add_rms_norm_forward(x, r, w, EPS, 0.0, "llama")

        def separate_fwd():
            s = x + r
            y = LigerRMSNormFunction.apply(s, w, EPS, 0.0, "llama", False, None)
            return y, s

        fns = {
            "copy": (lambda: x.clone(), 1),
            "fwd_fused": (lambda: fused_add_rms_norm_forward(x, r, w, EPS, 0.0, "llama")[0], 3),
            "fwd_separate": (separate_fwd, 3),
            "fwd_eager": (lambda: hf_llama_fused(x, r, w, EPS, 0.0), 3),
            "bwd_fused_oop": (
                lambda: fused_add_rms_norm_backward(
                    gy, gs, S, w, RSTD, 0.0, CM, BS, NW, NS, False, needs_dw=True
                ),
                4,
            ),
            "bwd_fused_inplace": (
                lambda: fused_add_rms_norm_backward(
                    gy.clone(), gs, S, w, RSTD, 0.0, CM, BS, NW, NS, True, needs_dw=True
                ),
                4,
            ),
            "bwd_eager": (
                lambda: torch.autograd.grad((yg, sg), (xg, rg, wg), (gy, gs), retain_graph=True),
                4,
            ),
            "bwd_liger_graph": (
                lambda: torch.autograd.grad((yl, sl), (xl, rl, wl), (gy, gs), retain_graph=True),
                4,
            ),
        }
        if compile_ok:
            xcc = x.clone().requires_grad_(True)
            rcc = r.clone().requires_grad_(True)
            wcc = w.clone().requires_grad_(True)
            ycc, scc = compiled(xcc, rcc, wcc)
            fns["fwd_compile"] = (lambda: compiled(x, r, w), 3)
            fns["bwd_compile"] = (
                lambda: torch.autograd.grad((ycc, scc), (xcc, rcc, wcc), (gy, gs), retain_graph=True),
                4,
            )
        do_bench(fns["copy"][0], rep=200)
        times = {k: [] for k in fns}
        for _ in range(BW_ROUNDS):
            for k, (fn, _) in fns.items():
                times[k].append(do_bench(fn, return_mode="median"))
        row = {
            "T": BW_T,
            "H": H,
            "BLOCK_SIZE": block,
            "num_warps": warps,
            "lane_util": round(H / block, 3),
            "MiB_per_tensor": round(nbytes / 2**20, 1),
            "compile_ok": compile_ok,
        }
        if not compile_ok:
            row["compile_error"] = compile_err
        for k, (_, mult) in fns.items():
            ms = sorted(times[k])[len(times[k]) // 2]
            row[k + "_ms"] = round(ms, 4)
            row[k + "_gbps"] = round(mult * nbytes * 1e-9 / (ms * 1e-3), 1)
        row["fwd_fused_vs_eager"] = round(row["fwd_eager_ms"] / max(row["fwd_fused_ms"], 1e-9), 2)
        row["fwd_fused_vs_separate"] = round(row["fwd_separate_ms"] / max(row["fwd_fused_ms"], 1e-9), 2)
        row["bwd_fused_oop_vs_eager"] = round(row["bwd_eager_ms"] / max(row["bwd_fused_oop_ms"], 1e-9), 2)
        rows.append(row)
        print("bw", row, flush=True)

        try:
            from liger_kernel.ops.fused_add_rms_norm import _fused_add_rms_norm_forward_kernel

            Y2 = torch.empty_like(Y)
            S2 = torch.empty_like(S)
            RSTD2 = torch.empty_like(RSTD)
            kfwd = _fused_add_rms_norm_forward_kernel.warmup(
                Y2, Y2.stride(0), S2, S2.stride(0), x, x.stride(0), r, r.stride(0),
                w, w.stride(0), RSTD2, RSTD2.stride(0), H, EPS, 0.0, 0,
                BLOCK_SIZE=block, num_warps=warps, grid=(1,),
            )
            kfwd._init_handles()
            kernels.append({
                "H": H, "kernel": "fwd", "BLOCK_SIZE": block, "num_warps": warps,
                "n_regs": kfwd.n_regs, "n_spills": kfwd.n_spills, "shared": kfwd.metadata.shared,
            })
        except Exception as e:
            kernels.append({"H": H, "kernel": "fwd", "error": f"{type(e).__name__}: {e}"})

        del x, r, w, gy, gs, xg, rg, wg, yg, sg, xl, rl, wl, yl, sl, Y, S, RSTD
        if compile_ok:
            del xcc, rcc, wcc, ycc, scc
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
        "## 0. `in_place` defaults",
        "",
        f"- Function.forward default: `{result['api_defaults']['function_forward_default']}`",
        f"- `liger_fused_add_rms_norm` functional default: `{result['api_defaults']['functional_default']}`",
        f"- `LigerFusedAddRMSNorm` module default: `{result['api_defaults']['module_default']}`",
        f"- Docstring claims default True: `{result['api_defaults']['docstring_claims_true']}`",
        "",
        f"## 1. Output branches (bf16, T = {BRANCH_T}, H = {BRANCH_H}; "
        "on this torch, unused Function outputs get **zero tensors**, not None)",
        "",
        "| branch | in_place | ok | dx rel | dr rel | dw rel | dx is dr (same ptr) | dx==dr | error |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in result["branches"]:
        lines.append(
            f"| {r['branch']} | {r['in_place']} | {r.get('ok')} | {fmt(r.get('dx_rel'))} | "
            f"{fmt(r.get('dr_rel'))} | {fmt(r.get('dw_rel'))} | {r.get('dx_is_dr_same_tensor')} | "
            f"{r.get('dx_equals_dr')} | {short_err(r.get('error'))} |"
        )
    lines += [
        "",
        f"## 2. Aliasing (bf16, T = {ALIAS_T}, H = {ALIAS_H})",
        "",
        "| case | in_place | dx rel | dr rel | dw rel | 2nd vs 1st | reuse dY vs clean | "
        "reuse 2nd vs 1st | dx is dr | error |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in result["alias"]:
        lines.append(
            f"| {r['case']} | {r['in_place']} | {fmt(r.get('dx_rel'))} | {fmt(r.get('dr_rel'))} | "
            f"{fmt(r.get('dw_rel'))} | {fmt(r.get('second_call_rel_vs_first'))} | "
            f"{fmt(r.get('reuse_dY_first_vs_clean'))} | {fmt(r.get('reuse_dY_second_vs_first'))} | "
            f"{r.get('dx_is_dr_same_tensor')} | "
            f"{short_err(r.get('error') or r.get('reuse_error'))} |"
        )
    lines += [
        "",
        f"## 3. Error vs fp64 (T = {NUM_T}, H = {NUM_H}; rel L2)",
        "",
        "| dtype | mode | offset | out | floor | liger | eager_hf | compile | liger vs hf |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in result["numerics"]:
        for o in ("y", "s", "dx", "dr", "dw"):
            cr = r["torch_compile"].get(o, r["torch_compile"].get("error", ""))
            cr_s = fmt(cr) if not isinstance(cr, str) else short_err(cr)
            lines.append(
                f"| {r['dtype']} | {r['casting_mode']} | {r['offset']} | {o} | "
                f"{fmt(r['floor'][o])} | {fmt(r['liger'][o])} | {fmt(r['eager_hf'][o])} | "
                f"{cr_s} | {fmt(r['liger_vs_hf'][o])} |"
            )
    names = ["liger_fused_inplace", "liger_fused_oop", "separate_add_liger_rms", "eager_hf", "torch_compile"]
    lines += [
        "",
        "## 4. Memory + latency (bf16; held after forward; extra peak through bwd; do_bench median)",
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
            cells = []
            for n in names:
                cell = r.get(n, {})
                cells.append(str(cell.get(metric, cell.get("error", ""))))
            lines.append(f"| {r['shape']} | {r['T']} | {r['H']} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        f"## 5. Bandwidth (bf16, T = {BW_T}; fwd counted as 3·T·H·2 bytes (X+R→Y+S traffic), "
        "bwd as 4·T·H·2; do_bench median of 3 rounds)",
        "",
        "| H | BLOCK | lanes | copy | fwd fused | fwd separate | fwd eager | fwd compile | "
        "bwd fused oop | bwd fused in-place | bwd eager | bwd graph | "
        "fused vs eager | fused vs separate | bwd vs eager |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in result["bandwidth"]:
        lines.append(
            f"| {r['H']} | {r['BLOCK_SIZE']} | {r['lane_util']} | {r['copy_gbps']} | "
            f"{r['fwd_fused_gbps']} | {r['fwd_separate_gbps']} | {r['fwd_eager_gbps']} | "
            f"{r.get('fwd_compile_gbps', '')} | {r['bwd_fused_oop_gbps']} | "
            f"{r['bwd_fused_inplace_gbps']} | {r['bwd_eager_gbps']} | {r['bwd_liger_graph_gbps']} | "
            f"{r['fwd_fused_vs_eager']} | {r['fwd_fused_vs_separate']} | {r['bwd_fused_oop_vs_eager']} |"
        )
    lines += [
        "",
        "### Kernel resources (fused forward)",
        "",
        "| H | kernel | BLOCK | warps | regs/thread | spills (bytes) | shared |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in result["kernels"]:
        if "error" in k:
            lines.append(f"| {k['H']} | {k['kernel']} |  |  |  |  | {short_err(k['error'])} |")
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
    defaults = api_defaults()
    print("api_defaults", defaults, flush=True)
    branch_rows = branches()
    alias_rows = alias()
    numeric_rows = numerics()
    mem_rows = memory_bench()
    bw_rows, kernel_rows = bandwidth()
    result = {
        "experiment": HERE.name,
        "upstream": "linkedin/Liger-Kernel",
        "upstream_commit": upstream_commit(),
        "component": "src/liger_kernel/ops/fused_add_rms_norm.py (LigerFusedAddRMSNormFunction)",
        "triton": triton.__version__,
        "gpu": GPU,
        "api_defaults": defaults,
        "branches": branch_rows,
        "alias": alias_rows,
        "numerics": numeric_rows,
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
