"""Liger-Kernel Triton JSD (`LigerJSDFunction`) on an RTX 4060 Laptop.

`jsd_forward` clones the student log-probs, launches `_jsd_kernel` (one program
per row, 4096-column blocks, dX written in place into the clone), writes a
per-element fp32 loss tile of shape (BT, V), sums it and casts to the input
dtype. Backward returns the saved dX (scaled by grad_output unless it is 1).

1. Numerics vs an fp64 analytic reference (and an eager TRL-style
   `F.kl_div` + `logsumexp` JSD) for beta in {0 .. 1}, fp32/fp16/bf16, "far"
   and "close" student/teacher pairs; loss output dtype; the beta -> 0 / 1
   scale cliff of the special-cased forward / reverse KL branches.
2. Non-finite log-probs: columns that are -inf in both models (padded vocab),
   in the teacher only (top-k truncation), in the student only; whole -inf
   4096-blocks; finite sentinels; end-to-end through log_softmax; a fix.
3. API / autograd behavior: ignore_index rows, all rows ignored, grad_output
   != 1, aliasing of the returned gradient with the saved tensor under
   retain_graph, gradient w.r.t. the target, non-contiguous inputs.
4. Memory / latency vs token count at 32k / 128k / 152k vocabularies for
   upstream vs a row-loss rewrite vs eager TRL-style vs torch.compile (fp32).

Liger-Kernel is imported unmodified from the submodule; the rewrite lives here.
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
import triton.language as tl  # noqa: E402
from triton.testing import do_bench  # noqa: E402
from liger_kernel.ops.jsd import LigerJSDFunction  # noqa: E402

SEED = 0
DEV = "cuda"
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
NUM_BT, NUM_V = 64, 32000
BETAS = [0.0, 0.1, 0.5, 0.9, 1.0]
CLIFF_BETAS = [0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0]
NF_BT, NF_V = 16, 16384
PERF_VOCABS = [("32k", 32000), ("Llama-3 128k", 128256), ("Qwen3 152k", 151936)]
PERF_BTS = [256, 1024, 2048, 3072, 4096]
COMPILE_CELLS = {(32000, 2048), (128256, 2048), (151936, 1024), (151936, 3072)}

torch._dynamo.config.recompile_limit = 64
torch._functorch.config.donated_buffer = False


@triton.jit
def _jsd_rows_kernel(
    X_ptr, Y_ptr, dX_ptr, loss_ptr, label_ptr, scale_ptr, g_ptr, stride,
    beta: tl.constexpr, ignore_index: tl.constexpr, n_cols,
    BLOCK: tl.constexpr, HAS_LABEL: tl.constexpr,
    WRITE_LOSS: tl.constexpr, WRITE_GRAD: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    X_ptr += pid * stride
    Y_ptr += pid * stride
    dX_ptr += pid * stride
    inv_n = tl.load(scale_ptr)
    skip = False
    if HAS_LABEL:
        skip = tl.load(label_ptr + pid) == ignore_index
    scale = inv_n
    if WRITE_GRAD:
        scale = inv_n * tl.load(g_ptr)
    acc = 0.0
    for i in range(0, n_cols, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        mask = offs < n_cols
        if skip:
            if WRITE_GRAD:
                tl.store(dX_ptr + offs, 0.0, mask=mask)
        else:
            X = tl.load(X_ptr + offs, mask=mask, other=float("-inf")).to(tl.float32)
            Y = tl.load(Y_ptr + offs, mask=mask, other=float("-inf")).to(tl.float32)
            P = tl.exp(Y)
            Q = tl.exp(X)
            if beta == 0.0:
                loss = tl.where(P > 0, P * (Y - X), 0.0)
                dX = -P
            elif beta == 1.0:
                loss = tl.where(Q > 0, Q * (X - Y), 0.0)
                dX = loss + Q
            else:
                m = tl.maximum(X, Y)
                m = tl.where(m == float("-inf"), 0.0, m)
                log_m = m + tl.log(beta * tl.exp(Y - m) + (1 - beta) * tl.exp(X - m))
                lp = tl.where(P > 0, beta * P * (Y - log_m), 0.0)
                lq = tl.where(Q > 0, (1 - beta) * Q * (X - log_m), 0.0)
                loss = lp + lq
                dX = lq
            if WRITE_LOSS:
                acc += tl.sum(loss, axis=0)
            if WRITE_GRAD:
                tl.store(dX_ptr + offs, (dX * scale).to(dX_ptr.dtype.element_ty), mask=mask)
    if WRITE_LOSS:
        tl.store(loss_ptr + pid, acc * inv_n)


def _row_args(X, labels, ignore_index):
    BT = X.shape[0]
    if labels is None:
        inv_n = torch.full((1,), 1.0 / BT, device=X.device, dtype=torch.float32)
        label_arg = torch.empty(1, device=X.device, dtype=torch.int64)
    else:
        n = (labels != ignore_index).sum().clamp(min=1).to(torch.float32)
        inv_n = (1.0 / n).reshape(1)
        label_arg = labels.contiguous()
    return inv_n, label_arg


class RowJSD(torch.autograd.Function):
    """Loss-only forward (per-row loss, no (BT, V) tile, nothing saved but X / Y),
    gradient recomputed in backward with grad_output read on the device."""

    @staticmethod
    def forward(ctx, X, Y, labels, beta, ignore_index):
        X, Y = X.contiguous(), Y.contiguous()
        BT, V = X.shape
        inv_n, label_arg = _row_args(X, labels, ignore_index)
        loss_rows = torch.empty(BT, device=X.device, dtype=torch.float32)
        _jsd_rows_kernel[(BT,)](
            X, Y, X, loss_rows, label_arg, inv_n, inv_n, V,
            beta=float(beta), ignore_index=ignore_index, n_cols=V,
            BLOCK=min(4096, triton.next_power_of_2(V)),
            HAS_LABEL=labels is not None, WRITE_LOSS=True, WRITE_GRAD=False,
        )
        ctx.save_for_backward(X, Y, label_arg, inv_n)
        ctx.beta, ctx.ignore_index, ctx.has_label = float(beta), ignore_index, labels is not None
        return loss_rows.sum()

    @staticmethod
    def backward(ctx, grad_output):
        X, Y, label_arg, inv_n = ctx.saved_tensors
        BT, V = X.shape
        dX = torch.empty_like(X)
        g = grad_output.to(torch.float32).reshape(1).contiguous()
        _jsd_rows_kernel[(BT,)](
            X, Y, dX, g, label_arg, inv_n, g, V,
            beta=ctx.beta, ignore_index=ctx.ignore_index, n_cols=V,
            BLOCK=min(4096, triton.next_power_of_2(V)),
            HAS_LABEL=ctx.has_label, WRITE_LOSS=False, WRITE_GRAD=True,
        )
        return dX, None, None, None, None


def liger_jsd(X, Y, beta, labels=None, ignore_index=-100):
    return LigerJSDFunction.apply(X, Y, labels, beta, ignore_index)


def rows_jsd(X, Y, beta, labels=None, ignore_index=-100):
    return RowJSD.apply(X, Y, labels, beta, ignore_index)


def eager_jsd(X, Y, beta):
    """TRL-style GKD loss: F.kl_div + logsumexp mixture in the input dtype."""
    BT = X.shape[0]
    if beta == 0.0:
        return F.kl_div(X, Y, reduction="sum", log_target=True) / BT
    if beta == 1.0:
        return F.kl_div(Y, X, reduction="sum", log_target=True) / BT
    lm = torch.logsumexp(torch.stack([X + math.log(1 - beta), Y + math.log(beta)]), 0)
    return (beta * F.kl_div(lm, Y, reduction="sum", log_target=True)
            + (1 - beta) * F.kl_div(lm, X, reduction="sum", log_target=True)) / BT


def _compiled_body(X, Y, beta):
    X, Y = X.float(), Y.float()
    lm = torch.logaddexp(Y + math.log(beta), X + math.log(1 - beta))
    P, Q = Y.exp(), X.exp()
    return (beta * P * (Y - lm) + (1 - beta) * Q * (X - lm)).sum() / X.shape[0]


_compiled = torch.compile(_compiled_body)


def compile_jsd(X, Y, beta):
    return _compiled(X, Y, beta)


def ref_loss_grad(X, Y, beta, labels=None, ignore_index=-100):
    """fp64 loss and d loss / d X treating log Q as free variables (what the kernels return).
    Zero-probability entries contribute 0 (xlogy convention); +inf where KL is infinite."""
    X, Y = X.double(), Y.double()
    BT = X.shape[0]
    P, Q = Y.exp(), X.exp()
    zero = torch.zeros_like(P)
    if beta == 0.0:
        el = torch.where(P > 0, P * (Y - X), zero)
        dX = -P
    elif beta == 1.0:
        el = torch.where(Q > 0, Q * (X - Y), zero)
        dX = el + Q
    else:
        lm = torch.logaddexp(Y + math.log(beta), X + math.log(1 - beta))
        lp = torch.where(P > 0, beta * P * (Y - lm), zero)
        lq = torch.where(Q > 0, (1 - beta) * Q * (X - lm), zero)
        el = lp + lq
        dX = lq
    if labels is not None:
        keep = (labels != ignore_index).double().unsqueeze(1)
        n = max(int(keep.sum().item()), 1)
        return (el * keep).sum() / n, dX * keep / n
    return el.sum() / BT, dX / BT


def autograd_ref_grad(X, Y, beta, wrt_target=False):
    """fp64 autograd of the finite-input JSD (used to validate ref_loss_grad and the target grad)."""
    X = X.double().clone().requires_grad_(not wrt_target)
    Y = Y.double().clone().requires_grad_(wrt_target)
    BT = X.shape[0]
    if beta in (0.0, 1.0):
        P, Q = Y.exp(), X.exp()
        el = P * (Y - X) if beta == 0.0 else Q * (X - Y)
    else:
        lm = torch.logaddexp(Y + math.log(beta), X + math.log(1 - beta))
        el = beta * Y.exp() * (Y - lm) + (1 - beta) * X.exp() * (X - lm)
    (el.sum() / BT).backward()
    return (Y if wrt_target else X).grad


def logp(z, dtype):
    return z.double().log_softmax(-1).to(dtype)


def make_pair(BT, V, dtype, kind, gen):
    zs = torch.randn(BT, V, device=DEV, dtype=torch.float64, generator=gen) * 2
    if kind == "far":
        zt = zs * 0.7 + torch.randn(BT, V, device=DEV, dtype=torch.float64, generator=gen) * 1.5
    else:
        zt = zs + torch.randn(BT, V, device=DEV, dtype=torch.float64, generator=gen) * 0.1
    return logp(zs, dtype), logp(zt, dtype)


def run_fb(fn, X, Y, *args, **kw):
    Xg = X.detach().clone().requires_grad_(True)
    loss = fn(Xg, Y, *args, **kw)
    loss.backward()
    return loss.detach(), Xg.grad.detach()


def rel(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    den = b.norm()
    return float((a - b).norm() / den) if den > 0 else float((a - b).norm())


def fmt(x, digits=4):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, str):
        return x
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


# ---------------------------------------------------------------- 1. numerics
def sec_numerics():
    gen = torch.Generator(device=DEV).manual_seed(SEED)
    rows = []
    checks = {}
    X, Y = make_pair(8, 4096, torch.float32, "far", gen)
    for beta in BETAS:
        _, gref = ref_loss_grad(X, Y, beta)
        checks[str(beta)] = rel(gref, autograd_ref_grad(X, Y, beta))
    for kind in ["far", "close"]:
        for dname, dt in DTYPES.items():
            X, Y = make_pair(NUM_BT, NUM_V, dt, kind, gen)
            for beta in BETAS:
                lref, gref = ref_loss_grad(X, Y, beta)
                floor = abs(float(lref.to(dt).double() - lref)) / float(lref)
                r = {"pair": kind, "dtype": dname, "beta": beta, "loss_ref": float(lref),
                     "loss_round_floor": floor}
                for name, fn in [("liger", liger_jsd), ("rows", rows_jsd), ("eager", eager_jsd)]:
                    loss, g = run_fb(fn, X, Y, beta)
                    r[name + "_loss_dtype"] = str(loss.dtype).replace("torch.", "")
                    r[name + "_loss_rel"] = abs(float(loss.double()) - float(lref)) / float(lref)
                    r[name + "_grad_rel"] = rel(g, gref)
                rows.append(r)
    return {"autograd_vs_analytic_grad_rel": checks, "rows": rows}


def sec_cliff():
    gen = torch.Generator(device=DEV).manual_seed(SEED + 1)
    X, Y = make_pair(NUM_BT, NUM_V, torch.float32, "far", gen)
    base = {}
    for beta in (0.0, 1.0):
        loss, g = run_fb(liger_jsd, X, Y, beta)
        base[beta] = (float(loss), float(g.norm()))
    rows = []
    for beta in CLIFF_BETAS:
        loss, g = run_fb(liger_jsd, X, Y, beta)
        rows.append({
            "beta": beta, "loss": float(loss), "grad_norm": float(g.norm()),
            "loss_over_beta": float(loss) / beta if beta > 0 else None,
            "loss_over_1mbeta": float(loss) / (1 - beta) if beta < 1 else None,
            "grad_norm_ratio_vs_beta0": float(g.norm()) / base[0.0][1],
            "grad_norm_ratio_vs_beta1": float(g.norm()) / base[1.0][1],
        })
    return {"kl_fwd_loss": base[0.0][0], "kl_rev_loss": base[1.0][0], "rows": rows}


# ------------------------------------------------------------- 2. non-finite
def nonfinite_scenarios(gen):
    BT, V = NF_BT, NF_V
    zs = torch.randn(BT, V, device=DEV, dtype=torch.float64, generator=gen) * 2
    zt = zs * 0.7 + torch.randn(BT, V, device=DEV, dtype=torch.float64, generator=gen) * 1.5
    inf = float("-inf")
    out = {}
    s, t = zs.clone(), zt.clone()
    s[:, V - 384:] = inf
    t[:, V - 384:] = inf
    out["pad_both_384"] = (s, t)
    s, t = zs.clone(), zt.clone()
    s[:, V - 4096:] = inf
    t[:, V - 4096:] = inf
    out["pad_both_whole_block"] = (s, t)
    s, t = zs.clone(), zt.clone()
    keep = torch.topk(t, 50, dim=-1).indices
    mask = torch.ones_like(t, dtype=torch.bool).scatter_(1, keep, False)
    t[mask] = inf
    out["teacher_top50"] = (s, t)
    s, t = zs.clone(), zt.clone()
    keep = torch.topk(s, 50, dim=-1).indices
    mask = torch.ones_like(s, dtype=torch.bool).scatter_(1, keep, False)
    s[mask] = inf
    out["student_top50"] = (s, t)
    return out


def sec_nonfinite():
    gen = torch.Generator(device=DEV).manual_seed(SEED + 2)
    rows = []
    for dname in ["fp32", "bf16"]:
        dt = DTYPES[dname]
        for sname, (zs, zt) in nonfinite_scenarios(gen).items():
            X, Y = logp(zs, dt), logp(zt, dt)
            for beta in [0.0, 0.5, 1.0]:
                lref, gref = ref_loss_grad(X, Y, beta)
                r = {"dtype": dname, "scenario": sname, "beta": beta, "loss_ref": float(lref),
                     "grad_ref_finite": bool(torch.isfinite(gref).all())}
                for name, fn in [("liger", liger_jsd), ("rows", rows_jsd), ("eager", eager_jsd)]:
                    loss, g = run_fb(fn, X, Y, beta)
                    gf = g.double()
                    ok = torch.isfinite(gref) & torch.isfinite(gf)
                    r[name + "_loss"] = float(loss)
                    r[name + "_grad_nan_frac"] = float(torch.isnan(gf).float().mean())
                    r[name + "_grad_inf_frac"] = float(torch.isinf(gf).float().mean())
                    r[name + "_grad_rel_on_finite"] = (
                        rel(gf[ok], gref[ok]) if ok.any() and not torch.isnan(gf).any() else None
                    )
                rows.append(r)
    sentinels = {"-inf": float("-inf"), "-1e9": -1e9, "fp16 min": -65504.0,
                 "fp32 min": float(torch.finfo(torch.float32).min)}
    srows = []
    zs = torch.randn(NF_BT, NF_V, device=DEV, dtype=torch.float64, generator=gen) * 2
    zt = zs * 0.7 + torch.randn(NF_BT, NF_V, device=DEV, dtype=torch.float64, generator=gen) * 1.5
    for sname, fill in sentinels.items():
        s, t = zs.clone(), zt.clone()
        s[:, NF_V - 384:] = fill
        t[:, NF_V - 384:] = fill
        X, Y = logp(s, torch.float32), logp(t, torch.float32)
        for beta in [0.0, 0.5, 1.0]:
            lref, gref = ref_loss_grad(X, Y, beta)
            r = {"fill": sname, "beta": beta, "loss_ref": float(lref)}
            for name, fn in [("liger", liger_jsd), ("rows", rows_jsd)]:
                loss, g = run_fb(fn, X, Y, beta)
                r[name + "_loss"] = float(loss)
                r[name + "_grad_nan_frac"] = float(torch.isnan(g).float().mean())
                r[name + "_grad_rel"] = rel(g, gref) if not torch.isnan(g).any() else None
            srows.append(r)
    return {"rows": rows, "sentinel_rows": srows, "end_to_end": nonfinite_e2e(gen)}


def nonfinite_e2e(gen):
    BT, V, valid = 64, 16384, 16000
    zs = torch.randn(BT, V, device=DEV, dtype=torch.float32, generator=gen) * 2
    zt = zs * 0.7 + torch.randn(BT, V, device=DEV, dtype=torch.float32, generator=gen) * 1.5
    zt[:, valid:] = float("-inf")
    out = {}
    fmin = torch.finfo(torch.float32).min

    def grad_for(fn, clamp):
        z = zs.clone()
        z[:, valid:] = float("-inf")
        z.requires_grad_(True)
        X = z.log_softmax(-1)
        Yt = zt.log_softmax(-1)
        if clamp:
            X, Yt = X.clamp(min=fmin), Yt.clamp(min=fmin)
        fn(X, Yt, 0.5).backward()
        return z.grad

    zr = zs[:, :valid].double().clone().requires_grad_(True)
    X = zr.log_softmax(-1)
    Yt = zt[:, :valid].double().log_softmax(-1)
    lm = torch.logaddexp(Yt + math.log(0.5), X + math.log(0.5))
    ((0.5 * Yt.exp() * (Yt - lm) + 0.5 * X.exp() * (X - lm)).sum() / BT).backward()
    gref = zr.grad
    for name, fn, clamp in [("liger", liger_jsd, False), ("liger_clamped", liger_jsd, True),
                            ("rows", rows_jsd, False)]:
        g = grad_for(fn, clamp)
        out[name] = {
            "grad_nan_frac": float(torch.isnan(g).float().mean()),
            "grad_rel_valid_cols": rel(g[:, :valid], gref) if not torch.isnan(g[:, :valid]).any() else None,
            "pad_grad_abs_max": float(g[:, valid:].abs().nan_to_num(nan=-1).max()),
        }
    return out


# -------------------------------------------------------------- 3. API / grad
def sec_api():
    gen = torch.Generator(device=DEV).manual_seed(SEED + 3)
    out = {}
    X, Y = make_pair(NUM_BT, 8192, torch.float32, "far", gen)
    beta = 0.5
    labels = torch.randint(0, 100, (NUM_BT,), device=DEV, generator=gen)
    labels[torch.rand(NUM_BT, device=DEV, generator=gen) < 0.25] = -100
    lref, gref = ref_loss_grad(X, Y, beta, labels)
    r = {"ignored_rows": int((labels == -100).sum())}
    for name, fn in [("liger", liger_jsd), ("rows", rows_jsd)]:
        loss, g = run_fb(lambda a, b, be: fn(a, b, be, labels), X, Y, beta)
        r[name + "_loss_rel"] = abs(float(loss) - float(lref)) / float(lref)
        r[name + "_grad_rel"] = rel(g, gref)
        r[name + "_ignored_grad_absmax"] = float(g[labels == -100].abs().max())
    all_ign = torch.full((NUM_BT,), -100, device=DEV)
    for name, fn in [("liger", liger_jsd), ("rows", rows_jsd)]:
        loss, g = run_fb(lambda a, b, be: fn(a, b, be, all_ign), X, Y, beta)
        r[name + "_all_ignored_loss"] = float(loss)
        r[name + "_all_ignored_grad_nan"] = bool(torch.isnan(g).any())
    out["ignore_index"] = r

    r = {}
    _, gref = ref_loss_grad(X, Y, beta)
    for name, fn in [("liger", liger_jsd), ("rows", rows_jsd)]:
        Xg = X.clone().requires_grad_(True)
        (fn(Xg, Y, beta) * 0.37).backward()
        r[name + "_scaled_grad_rel"] = rel(Xg.grad, gref * 0.37)
        Xg = X.clone().requires_grad_(True)
        (fn(Xg, Y, beta) * torch.tensor(1.0, device=DEV)).backward()
        r[name + "_one_grad_rel"] = rel(Xg.grad, gref)
        Xg = X.clone().requires_grad_(True)
        fn(Xg, Y, beta).backward(retain_graph=True) if False else None
    out["grad_output"] = r

    r = {}
    for name, fn in [("liger", liger_jsd), ("rows", rows_jsd)]:
        Xg = X.clone().requires_grad_(True)
        loss = fn(Xg, Y, beta)
        g1 = torch.autograd.grad(loss, Xg, retain_graph=True)[0]
        g1_copy = g1.clone()
        saved = [t for t in getattr(loss.grad_fn, "saved_tensors", ())]
        r[name + "_grad_aliases_saved_tensor"] = any(
            t.data_ptr() == g1.data_ptr() for t in saved if t.is_cuda
        )
        g1.zero_()
        g2 = torch.autograd.grad(loss, Xg)[0]
        r[name + "_second_backward_after_inplace_rel"] = rel(g2, g1_copy)
        Xg = X.clone().requires_grad_(True)
        loss = fn(Xg, Y, beta)
        loss.backward(retain_graph=True)
        loss.backward()
        r[name + "_leaf_grad_accum_ratio"] = float(Xg.grad.norm() / g1_copy.norm())
    out["retain_graph_inplace_edit"] = r

    r = {}
    Yg = Y.clone().requires_grad_(True)
    Xd = X.clone()
    loss = liger_jsd(Xd.requires_grad_(True), Yg, beta)
    loss.backward()
    r["liger_target_grad_is_none"] = Yg.grad is None
    r["ref_target_grad_norm"] = float(autograd_ref_grad(X, Y, beta, wrt_target=True).norm())
    out["target_grad"] = r

    r = {}
    big = torch.cat([X, torch.zeros(NUM_BT, 128, device=DEV)], 1)
    Xv = big[:, :8192]
    Xv_before = big.clone()
    for name, fn in [("liger", liger_jsd), ("rows", rows_jsd)]:
        Xg = Xv.clone().requires_grad_(True)
        loss, g = run_fb(fn, Xv, Y, beta)
        _, gref = ref_loss_grad(Xv, Y, beta)
        r[name + "_noncontig_grad_rel"] = rel(g, gref)
    r["input_unchanged"] = bool(torch.equal(big, Xv_before))
    out["noncontiguous"] = r
    return out


# ------------------------------------------------------------ 4. mem / latency
def measure(fn, X, Y, beta, bench=True):
    cleanup()
    X.grad = None
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    loss = fn(X, Y, beta)
    torch.cuda.synchronize()
    held = torch.cuda.memory_allocated() - base
    peak_fwd = torch.cuda.max_memory_allocated() - base
    loss.backward()
    torch.cuda.synchronize()
    peak_total = torch.cuda.max_memory_allocated() - base
    del loss
    X.grad = None
    cleanup()
    res = {
        "held_after_fwd_mib": round(held / 2**20, 1),
        "peak_fwd_mib": round(peak_fwd / 2**20, 1),
        "peak_fwd_bwd_mib": round(peak_total / 2**20, 1),
    }
    if bench:
        def step():
            X.grad = None
            fn(X, Y, beta).backward()
        res["fwd_bwd_ms"] = do_bench(step, warmup=25, rep=100)
        X.grad = None
        with torch.no_grad():
            res["nograd_fwd_ms"] = do_bench(lambda: fn(X, Y, beta), warmup=25, rep=100)
        cleanup()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        with torch.no_grad():
            fn(X, Y, beta)
        torch.cuda.synchronize()
        res["nograd_peak_mib"] = round((torch.cuda.max_memory_allocated() - base) / 2**20, 1)
    return res


def sec_perf():
    gen = torch.Generator(device=DEV).manual_seed(SEED + 4)
    beta = 0.5
    rows = []
    impls = [("liger", lambda a, b, be: liger_jsd(a, b, be)),
             ("rows", lambda a, b, be: rows_jsd(a, b, be)),
             ("eager_bf16", eager_jsd),
             ("compile_fp32", compile_jsd)]
    for vname, V in PERF_VOCABS:
        for BT in PERF_BTS:
            for name, fn in impls:
                if name == "compile_fp32" and (V, BT) not in COMPILE_CELLS:
                    continue
                r = {"vocab": vname, "V": V, "BT": BT, "impl": name}
                try:
                    with torch.no_grad():
                        X = (torch.randn(BT, V, device=DEV, dtype=torch.float32, generator=gen) * 2).log_softmax(-1).bfloat16()
                        Y = (torch.randn(BT, V, device=DEV, dtype=torch.float32, generator=gen) * 2).log_softmax(-1).bfloat16()
                    X.requires_grad_(True)
                    r.update(measure(fn, X, Y, beta))
                    r["gbps_5x"] = 5 * 2 * BT * V / (r["fwd_bwd_ms"] * 1e-3) / 1e9
                    r["oom"] = False
                except torch.OutOfMemoryError:
                    r["oom"] = True
                finally:
                    X = Y = None
                    cleanup()
                rows.append(r)
                print("  perf", {k: fmt(v) for k, v in r.items()}, flush=True)
    return {"beta": beta, "rows": rows}


# ------------------------------------------------------------------- main
def main():
    t0 = time.time()
    res = {"gpu": GPU, "triton": triton.__version__,
           "liger_commit": subprocess.run(["git", "-C", str(LIGER_DIR), "rev-parse", "--short", "HEAD"],
                                          capture_output=True, text=True).stdout.strip()}
    for name, fn in [("numerics", sec_numerics), ("cliff", sec_cliff), ("nonfinite", sec_nonfinite),
                     ("api", sec_api), ("perf", sec_perf)]:
        print(f"== {name}", flush=True)
        res[name] = fn()
    res["peak_allocated_mib"] = gpu_guard.peak_memory_mib()
    res["runtime_s"] = round(time.time() - t0, 1)
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=str))
    (HERE / "results.md").write_text(render_md(res))
    print("done in", res["runtime_s"], "s")


def render_md(res):
    L = [f"# 0025 results\n\nGPU: {res['gpu']['name']}, torch {res['gpu']['torch']}, triton {res['triton']}, "
         f"Liger {res['liger_commit']}; peak allocated {res['peak_allocated_mib']} MiB; runtime {res['runtime_s']} s\n"]
    n = res["numerics"]
    L.append("## 1. Numerics vs fp64 (BT=64, V=32000)\n")
    L.append(f"fp64 autograd vs analytic dX rel err: {n['autograd_vs_analytic_grad_rel']}\n")
    hdr = ["pair", "dtype", "beta", "floor", "liger loss", "rows loss", "eager loss",
           "liger dX", "rows dX", "eager dX", "liger loss dtype"]
    L.append(table(hdr, [[r["pair"], r["dtype"], r["beta"], r["loss_round_floor"],
                          r["liger_loss_rel"], r["rows_loss_rel"], r["eager_loss_rel"],
                          r["liger_grad_rel"], r["rows_grad_rel"], r["eager_grad_rel"],
                          r["liger_loss_dtype"]] for r in n["rows"]]))
    c = res["cliff"]
    L.append(f"\n### beta scale cliff (fp32; KL(P||Q) = {fmt(c['kl_fwd_loss'])}, KL(Q||P) = {fmt(c['kl_rev_loss'])})\n")
    L.append(table(["beta", "loss", "loss / beta", "loss / (1-beta)", "|dX| / |dX|(beta=0)", "|dX| / |dX|(beta=1)"],
                   [[r["beta"], r["loss"], r["loss_over_beta"], r["loss_over_1mbeta"],
                     r["grad_norm_ratio_vs_beta0"], r["grad_norm_ratio_vs_beta1"]] for r in c["rows"]]))
    nf = res["nonfinite"]
    L.append("\n## 2. Non-finite log-probs\n")
    L.append(table(["dtype", "scenario", "beta", "ref loss", "liger loss", "liger dX nan", "liger dX inf",
                    "rows loss", "rows dX nan", "rows dX inf", "rows dX err", "eager loss", "eager dX nan"],
                   [[r["dtype"], r["scenario"], r["beta"], r["loss_ref"], r["liger_loss"],
                     r["liger_grad_nan_frac"], r["liger_grad_inf_frac"], r["rows_loss"],
                     r["rows_grad_nan_frac"], r["rows_grad_inf_frac"], r["rows_grad_rel_on_finite"],
                     r["eager_loss"], r["eager_grad_nan_frac"]] for r in nf["rows"]]))
    L.append("\n### finite sentinels for the 384 padded columns (fp32)\n")
    L.append(table(["fill", "beta", "ref loss", "liger loss", "liger dX nan", "liger dX err",
                    "rows loss", "rows dX err"],
                   [[r["fill"], r["beta"], r["loss_ref"], r["liger_loss"], r["liger_grad_nan_frac"],
                     r["liger_grad_rel"], r["rows_loss"], r["rows_grad_rel"]] for r in nf["sentinel_rows"]]))
    L.append("\n### end to end: -inf padded logits -> log_softmax -> JSD -> backward (d logits)\n")
    L.append(table(["variant", "dlogits nan frac", "valid-column err vs fp64", "|pad grad| max"],
                   [[k, v["grad_nan_frac"], v["grad_rel_valid_cols"], v["pad_grad_abs_max"]]
                    for k, v in nf["end_to_end"].items()]))
    L.append("\n## 3. API / autograd\n")
    for k, v in res["api"].items():
        L.append(f"**{k}**: " + json.dumps(v) + "\n")
    p = res["perf"]
    L.append("\n## 4. Memory / latency (bf16 log-probs, beta 0.5; MiB over the live X, Y)\n")
    L.append(table(["vocab", "BT", "impl", "held after fwd", "peak fwd", "peak fwd+bwd", "fwd+bwd ms",
                    "GB/s (5 accesses)", "no_grad fwd ms", "no_grad peak", "OOM"],
                   [[r["vocab"], r["BT"], r["impl"], r.get("held_after_fwd_mib"), r.get("peak_fwd_mib"),
                     r.get("peak_fwd_bwd_mib"), r.get("fwd_bwd_ms"), r.get("gbps_5x"),
                     r.get("nograd_fwd_ms"), r.get("nograd_peak_mib"), r["oom"]] for r in p["rows"]]))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
