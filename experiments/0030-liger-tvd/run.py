"""Liger-Kernel Triton Total Variation Distance (`LigerTVDLossFunction`).

`tv_distance_forward_triton` launches one program per row with
BLOCK_SIZE = min(16384, next_power_of_2(V)), computes TVD = 0.5 * |p - q|
into an always-fp32 loss buffer, and fuses the reduction scale into both the
per-row loss and the stored d/dp gradient (0.5 * sgn(p - q), 0 on ties).
Backward only returns a gradient for `p` (q / labels / reduction /
ignore_index get None). The `grad_output == 1.0` path returns the saved
grads buffer by reference via `torch.equal`.

1. Numerics vs fp64 analytic + eager torch TVD for all reductions, with /
   without ignore_index, fp32 / fp16 / bf16 (upstream tests skip fp16).
2. Ties and near-ties: exact p == q zeros the subgradient; half-precision
   "ties" that differ in fp32; self-distillation; padded equal zeros.
3. Input convention: probabilities vs raw logits (upstream unit tests feed
   raw randn as p); negative / non-normalized mass; non-finite values.
4. Autograd API: no q gradient, ignore_index (incl. all-ignored rows),
   last-layer vs scaled grad_output, retain_graph aliasing of the saved
   grads buffer, torch.compile, always-fp32 loss dtype.
5. Memory / latency vs eager torch and torch.compile at 32k / 128k / 152k
   vocab for batchmean and for reduction=none's fp32 (BT, V) tile.

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
from triton.testing import do_bench  # noqa: E402
from liger_kernel.ops.tvd import LigerTVDLossFunction  # noqa: E402
from liger_kernel.transformers.tvd import LigerTVDLoss  # noqa: E402

SEED = 0
DEV = "cuda"
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
REDUCTIONS = ["batchmean", "sum", "mean", "none"]
NUM_BT, NUM_V = 64, 32000
PERF_VOCABS = [("32k", 32000), ("Llama-3 128k", 128256), ("Qwen3 152k", 151936)]
PERF_BTS = [256, 1024, 2048, 3072, 4096]
COMPILE_CELLS = {(32000, 2048), (128256, 1024), (151936, 1024)}

torch._dynamo.config.recompile_limit = 64
torch._functorch.config.donated_buffer = False


def liger_tvd(p, q, shift_labels=None, reduction="batchmean", ignore_index=-100):
    return LigerTVDLossFunction.apply(p, q, shift_labels, reduction, ignore_index)


def torch_tvd(p, q, shift_labels=None, reduction="batchmean", ignore_index=-100):
    """Eager reference matching upstream test TorchTVDLoss."""
    tvd = torch.abs(p - q) / 2.0
    n_non_ignore = p.size(0)
    if shift_labels is not None:
        tvd = torch.where(shift_labels.unsqueeze(1) != ignore_index, tvd, torch.zeros_like(tvd))
        n_non_ignore = int((shift_labels != ignore_index).sum().item())
        if n_non_ignore == 0:
            return torch.tensor(0.0, device=tvd.device, dtype=torch.float32)
    if reduction == "mean":
        return torch.sum(tvd) / (n_non_ignore * p.size(1))
    if reduction == "sum":
        return torch.sum(tvd)
    if reduction == "none":
        return tvd
    if reduction == "batchmean":
        return torch.sum(tvd) / n_non_ignore
    raise ValueError(reduction)


def ref_tvd(p, q, shift_labels=None, reduction="batchmean", ignore_index=-100):
    """fp64 analytic TVD = 0.5 * |p - q| with the same reductions."""
    tvd = 0.5 * (p.double() - q.double()).abs()
    n_non_ignore = p.size(0)
    if shift_labels is not None:
        mask = shift_labels != ignore_index
        tvd = torch.where(mask.unsqueeze(1), tvd, torch.zeros_like(tvd))
        n_non_ignore = int(mask.sum().item())
        if n_non_ignore == 0:
            return torch.tensor(0.0, device=tvd.device, dtype=torch.float64)
    if reduction == "none":
        return tvd
    if reduction == "sum":
        return tvd.sum()
    if reduction == "mean":
        return tvd.sum() / (n_non_ignore * p.size(1))
    if reduction == "batchmean":
        return tvd.sum() / n_non_ignore
    raise ValueError(reduction)


def ref_grad_p(p, q, shift_labels=None, reduction="batchmean", ignore_index=-100):
    """Analytic d TVD / d p (subgradient 0 on ties)."""
    pd, qd = p.double(), q.double()
    g = torch.where(pd == qd, torch.zeros_like(pd), torch.where(pd > qd, 0.5, -0.5))
    n_non_ignore = p.size(0)
    if shift_labels is not None:
        mask = shift_labels != ignore_index
        g = torch.where(mask.unsqueeze(1), g, torch.zeros_like(g))
        n_non_ignore = int(mask.sum().item())
        if n_non_ignore == 0:
            return torch.zeros_like(pd)
    if reduction == "batchmean":
        g = g / n_non_ignore
    elif reduction == "mean":
        g = g / (n_non_ignore * p.size(1))
    elif reduction in ("sum", "none"):
        pass
    else:
        raise ValueError(reduction)
    return g


def make_probs(BT, V, dtype, gen, kind="soft"):
    logits = torch.randn(BT, V, device=DEV, dtype=torch.float64, generator=gen)
    if kind == "peaked":
        logits = logits * 4.0
    elif kind == "flat":
        logits = logits * 0.05
    p = logits.softmax(-1).to(dtype)
    q_logits = logits * 0.7 + torch.randn(BT, V, device=DEV, dtype=torch.float64, generator=gen)
    q = q_logits.softmax(-1).to(dtype)
    return p, q


def make_raw(BT, V, dtype, gen):
    """Upstream unit-test convention: raw randn as p, softmax as q."""
    p = torch.randn(BT, V, device=DEV, dtype=dtype, generator=gen)
    q = torch.randn(BT, V, device=DEV, dtype=torch.float32, generator=gen).softmax(-1).to(dtype)
    return p, q


def run_fb(fn, p, q, shift_labels=None, reduction="batchmean", ignore_index=-100):
    x = p.detach().clone().requires_grad_(True)
    loss = fn(x, q, shift_labels=shift_labels, reduction=reduction, ignore_index=ignore_index)
    if loss.ndim > 0:
        loss.sum().backward()
    else:
        loss.backward()
    return loss.detach(), x.grad.detach() if x.grad is not None else None


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
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    if isinstance(x, float) and math.isinf(x):
        return "inf" if x > 0 else "-inf"
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


def loss_scalar(loss):
    if not torch.isfinite(loss).all():
        if torch.isnan(loss).any():
            return "nan"
        return "inf" if (loss == float("inf")).any() else "-inf"
    if loss.ndim > 0:
        return float(loss.double().sum())
    v = float(loss.double())
    if math.isnan(v) or math.isinf(v):
        return "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")
    return v


# ---------------------------------------------------------------- 1. numerics
def sec_numerics():
    gen = torch.Generator(device=DEV).manual_seed(SEED)
    rows = []
    for kind in ["probs", "raw"]:
        for dname, dt in DTYPES.items():
            if kind == "probs":
                p, q = make_probs(NUM_BT, NUM_V, dt, gen)
            else:
                p, q = make_raw(NUM_BT, NUM_V, dt, gen)
            for reduction in REDUCTIONS:
                lref = ref_tvd(p, q, reduction=reduction)
                gref = ref_grad_p(p, q, reduction=reduction)
                lt, gt = run_fb(torch_tvd, p, q, reduction=reduction)
                if reduction == "none":
                    floor = rel(lref.to(dt), lref)
                else:
                    floor = abs(float(lref.to(dt).double()) - float(lref)) / max(abs(float(lref)), 1e-30)
                r = {
                    "kind": kind,
                    "dtype": dname,
                    "reduction": reduction,
                    "loss_ref": float(lref.double().sum()) if reduction == "none" else float(lref),
                    "loss_round_floor": floor,
                }
                for name, fn in [("liger", liger_tvd), ("torch", torch_tvd)]:
                    loss, g = run_fb(fn, p, q, reduction=reduction)
                    r[name + "_loss_dtype"] = str(loss.dtype).replace("torch.", "")
                    if reduction == "none":
                        r[name + "_loss_rel"] = rel(loss, lref)
                        r[name + "_grad_rel"] = rel(g, gref) if g is not None else None
                    else:
                        r[name + "_loss_rel"] = abs(float(loss.double()) - float(lref)) / max(
                            abs(float(lref)), 1e-30
                        )
                        r[name + "_grad_rel"] = rel(g, gref) if g is not None else None
                    r[name + "_vs_torch_loss_rel"] = rel(loss, lt)
                    if g is not None and gt is not None:
                        r[name + "_vs_torch_grad_rel"] = rel(g, gt)
                rows.append(r)
    return {"rows": rows}


def sec_ignore():
    gen = torch.Generator(device=DEV).manual_seed(SEED + 1)
    BT, V = 64, 8192
    p, q = make_probs(BT, V, torch.float32, gen)
    labels = torch.randint(0, V, (BT,), device=DEV, generator=gen)
    # ignore ~40%
    n_ignore = BT * 2 // 5
    gen_ign = torch.Generator(device=DEV).manual_seed(SEED + 11)
    labels[torch.randperm(BT, device=DEV, generator=gen_ign)[:n_ignore]] = -100
    rows = []
    for reduction in REDUCTIONS:
        lref = ref_tvd(p, q, labels, reduction)
        gref = ref_grad_p(p, q, labels, reduction)
        r = {"reduction": reduction, "n_ignore": n_ignore, "n_keep": BT - n_ignore}
        for name, fn in [("liger", liger_tvd), ("torch", torch_tvd)]:
            try:
                loss, g = run_fb(fn, p, q, shift_labels=labels, reduction=reduction)
                r[name + "_loss"] = loss_scalar(loss)
                if reduction == "none":
                    r[name + "_loss_rel"] = rel(loss, lref)
                else:
                    r[name + "_loss_rel"] = abs(float(loss.double()) - float(lref)) / max(
                        abs(float(lref)), 1e-30
                    )
                r[name + "_grad_rel"] = rel(g, gref) if g is not None else None
                ignored_grad_abs = float(g[labels == -100].double().abs().max()) if g is not None else None
                r[name + "_ignored_grad_abs_max"] = ignored_grad_abs
            except Exception as e:
                r[name + "_loss"] = f"ERR:{type(e).__name__}: {e}"
                r[name + "_loss_rel"] = None
                r[name + "_grad_rel"] = None
                r[name + "_ignored_grad_abs_max"] = None
        rows.append(r)

    # all ignored
    all_ignore = torch.full((BT,), -100, device=DEV, dtype=torch.long)
    all_rows = []
    for reduction in ["batchmean", "sum", "mean"]:
        cell = {"reduction": reduction}
        for name, fn in [("liger", liger_tvd), ("torch", torch_tvd)]:
            try:
                loss, g = run_fb(fn, p, q, shift_labels=all_ignore, reduction=reduction)
                cell[name + "_loss"] = loss_scalar(loss)
                cell[name + "_grad_abs_max"] = float(g.double().abs().max()) if g is not None else None
            except Exception as e:
                cell[name + "_loss"] = f"ERR:{type(e).__name__}"
                cell[name + "_grad_abs_max"] = None
                cell[name + "_err"] = str(e)[:200]
        all_rows.append(cell)
    return {"partial": rows, "all_ignored": all_rows}


# ---------------------------------------------------------------- 2. ties
def sec_ties():
    gen = torch.Generator(device=DEV).manual_seed(SEED + 2)
    BT, V = 32, 4096
    rows = []

    # self-distillation: p is q
    for dname, dt in DTYPES.items():
        p, _ = make_probs(BT, V, dt, gen)
        q = p.detach().clone()
        for reduction in ["batchmean", "none"]:
            loss, g = run_fb(liger_tvd, p, q, reduction=reduction)
            lt, gt = run_fb(torch_tvd, p, q, reduction=reduction)
            rows.append({
                "case": "self_distill",
                "dtype": dname,
                "reduction": reduction,
                "liger_loss": loss_scalar(loss),
                "torch_loss": loss_scalar(lt),
                "liger_grad_abs_max": float(g.double().abs().max()) if g is not None else None,
                "torch_grad_abs_max": float(gt.double().abs().max()) if gt is not None else None,
                "liger_grad_nonzero_frac": float((g != 0).float().mean()) if g is not None else None,
            })

    # partial trailing ties (padded vocab style)
    for dname, dt in DTYPES.items():
        p, q = make_probs(BT, V, dt, gen)
        n_tied = V // 2
        q = q.clone()
        q[:, V - n_tied :] = p[:, V - n_tied :]
        loss, g = run_fb(liger_tvd, p, q, reduction="batchmean")
        gref = ref_grad_p(p, q, reduction="batchmean")
        tied_nz = float((g[:, V - n_tied :] != 0).float().mean())
        untied_match = rel(g[:, : V - n_tied], gref[:, : V - n_tied])
        rows.append({
            "case": "half_tied_pad",
            "dtype": dname,
            "reduction": "batchmean",
            "liger_loss": loss_scalar(loss),
            "tied_grad_nonzero_frac": tied_nz,
            "untied_grad_rel": untied_match,
            "liger_grad_rel_vs_ref": rel(g, gref),
        })

    # near-ties: equal in half precision after cast, but differ in fp32 source
    p32 = torch.randn(BT, V, device=DEV, dtype=torch.float32, generator=gen).softmax(-1)
    q32 = p32 + 1e-5  # small fp32 difference
    for dname, dt in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
        p = p32.to(dt)
        q = q32.to(dt)
        # how many exact ties after cast?
        n_exact = int((p == q).sum().item())
        loss, g = run_fb(liger_tvd, p, q, reduction="batchmean")
        gref = ref_grad_p(p, q, reduction="batchmean")
        rows.append({
            "case": "near_tie_1e-5",
            "dtype": dname,
            "reduction": "batchmean",
            "exact_ties_after_cast": n_exact,
            "tie_frac": n_exact / (BT * V),
            "liger_loss": loss_scalar(loss),
            "grad_zero_frac": float((g == 0).float().mean()),
            "liger_grad_rel_vs_ref": rel(g, gref),
        })

    # equal zeros padding
    p, q = make_probs(BT, V, torch.bfloat16, gen)
    p = p.clone()
    q = q.clone()
    p[:, -512:] = 0
    q[:, -512:] = 0
    # renormalize the live part only for a fair TVD, but also test raw zero pad
    loss, g = run_fb(liger_tvd, p, q, reduction="batchmean")
    rows.append({
        "case": "equal_zero_pad_512",
        "dtype": "bf16",
        "reduction": "batchmean",
        "liger_loss": loss_scalar(loss),
        "pad_grad_nonzero_frac": float((g[:, -512:] != 0).float().mean()),
        "live_grad_nonzero_frac": float((g[:, :-512] != 0).float().mean()),
    })
    return {"rows": rows}


# ---------------------------------------------------------------- 3. edges
def sec_edges():
    gen = torch.Generator(device=DEV).manual_seed(SEED + 3)
    BT, V = 16, 8192
    base_p, base_q = make_probs(BT, V, torch.float32, gen)
    scenarios = {}

    scenarios["probs_ok"] = (base_p.clone(), base_q.clone())

    p, q = base_p.clone(), base_q.clone()
    p[:, -512:] = 0
    q[:, -512:] = 0
    scenarios["both_zero_pad"] = (p, q)

    p, q = base_p.clone(), base_q.clone()
    p[:, 0] = -0.2
    scenarios["neg_p"] = (p, q)

    p, q = base_p.clone(), base_q.clone()
    q[:, 0] = -0.2
    scenarios["neg_q"] = (p, q)

    p, q = base_p.clone(), base_q.clone()
    p[0, 0] = float("nan")
    scenarios["nan_p"] = (p, q)

    p, q = base_p.clone(), base_q.clone()
    q[0, 0] = float("nan")
    scenarios["nan_q"] = (p, q)

    p, q = base_p.clone(), base_q.clone()
    p[0, 0] = float("inf")
    scenarios["inf_p"] = (p, q)

    p, q = base_p.clone(), base_q.clone()
    q[:, :64] = float("-inf")
    scenarios["neg_inf_q"] = (p, q)

    # unnormalized: rows sum to ~2
    p = base_p.clone() * 2
    q = base_q.clone()
    scenarios["unnormalized_p"] = (p, q)

    # V = 1
    scenarios["v1"] = (
        torch.ones(BT, 1, device=DEV),
        torch.ones(BT, 1, device=DEV),
    )

    # V > 16384 to force multi-block loop (MAX_FUSED_SIZE = 16384)
    p_big, q_big = make_probs(8, 32768, torch.float32, gen)
    scenarios["v32768_multiblock"] = (p_big, q_big)

    rows = []
    for name, (p, q) in scenarios.items():
        r = {"case": name, "shape": list(p.shape)}
        for impl, fn in [("liger", liger_tvd), ("torch", torch_tvd)]:
            try:
                loss, g = run_fb(fn, p, q, reduction="batchmean")
                r[impl + "_loss"] = loss_scalar(loss)
                r[impl + "_grad_finite"] = bool(g is not None and torch.isfinite(g).all())
                r[impl + "_grad_nan"] = bool(g is not None and torch.isnan(g).any())
                r[impl + "_grad_abs_max"] = (
                    float(g.double().abs().max()) if g is not None and torch.isfinite(g).any() else None
                )
            except Exception as e:
                r[impl + "_loss"] = f"ERR:{type(e).__name__}"
                r[impl + "_grad_finite"] = False
                r[impl + "_grad_nan"] = None
                r[impl + "_grad_abs_max"] = None
        if "liger_loss" in r and "torch_loss" in r:
            try:
                if isinstance(r["liger_loss"], float) and isinstance(r["torch_loss"], float):
                    r["loss_rel"] = abs(r["liger_loss"] - r["torch_loss"]) / max(abs(r["torch_loss"]), 1e-30)
            except Exception:
                pass
        rows.append(r)
    return {"rows": rows}


# ---------------------------------------------------------------- 4. API
def sec_api():
    gen = torch.Generator(device=DEV).manual_seed(SEED + 4)
    BT, V = 32, 4096
    p, q = make_probs(BT, V, torch.float32, gen)
    out = {}

    # last-layer grad_output == 1.0
    x = p.clone().requires_grad_(True)
    loss = liger_tvd(x, q, reduction="batchmean")
    loss.backward()
    g_last = x.grad.detach().clone()
    out["last_layer_grad_rel_vs_ref"] = rel(g_last, ref_grad_p(p, q, reduction="batchmean"))

    # scaled 2.5
    x = p.clone().requires_grad_(True)
    loss = liger_tvd(x, q, reduction="batchmean")
    (loss * 2.5).backward()
    g_scaled = x.grad.detach().clone()
    out["scaled_2p5_grad_rel_vs_2p5_ref"] = rel(g_scaled, 2.5 * ref_grad_p(p, q, reduction="batchmean"))
    out["scaled_over_last"] = float(g_scaled.norm() / g_last.norm())

    # reduction none with ones grad_output
    x = p.clone().requires_grad_(True)
    loss_none = liger_tvd(x, q, reduction="none")
    loss_none.backward(torch.ones_like(loss_none))
    g_none = x.grad.detach().clone()
    out["none_ones_grad_rel_vs_ref"] = rel(g_none, ref_grad_p(p, q, reduction="none"))

    # q has no gradient
    x = p.clone().requires_grad_(True)
    y = q.clone().requires_grad_(True)
    loss = liger_tvd(x, y, reduction="batchmean")
    loss.backward()
    out["q_grad_is_none"] = y.grad is None
    out["p_grad_is_not_none"] = x.grad is not None
    # eager torch WOULD give q a gradient
    x2 = p.clone().requires_grad_(True)
    y2 = q.clone().requires_grad_(True)
    loss2 = torch_tvd(x2, y2, reduction="batchmean")
    loss2.backward()
    out["torch_q_grad_is_none"] = y2.grad is None
    out["torch_q_grad_abs_max"] = float(y2.grad.double().abs().max()) if y2.grad is not None else None

    # retain_graph: last-layer path returns the SAME saved grads buffer
    x = p.clone().requires_grad_(True)
    loss = liger_tvd(x, q, reduction="batchmean")
    loss.backward(retain_graph=True)
    g1 = x.grad.detach().clone()
    # mutate x.grad storage that might alias saved grads if returned by reference
    if x.grad is not None:
        x.grad.zero_()
    x.grad = None
    loss.backward()
    g2 = x.grad.detach().clone()
    out["retain_graph_second_rel"] = rel(g2, g1)
    out["retain_graph_second_abs_max"] = float(g2.double().abs().max())
    out["retain_graph_second_zero"] = bool(torch.all(g2 == 0))

    # After first backward without retain, does a second graph still work?
    # Probe aliasing more directly via the Function internals: compare data_ptr
    # of returned grad vs saved buffer when grad_output is exactly 1.0.
    from liger_kernel.ops.tvd import tvd_backward_triton

    fake_grads = torch.randn(BT, V, device=DEV)
    go = torch.tensor(1.0, device=DEV)
    out_ptr = tvd_backward_triton(go, fake_grads)
    out["grad_output_1_returns_same_storage"] = out_ptr.data_ptr() == fake_grads.data_ptr()
    go2 = torch.tensor(2.0, device=DEV)
    out_ptr2 = tvd_backward_triton(go2, fake_grads)
    out["grad_output_2_returns_same_storage"] = out_ptr2.data_ptr() == fake_grads.data_ptr()

    # non-contiguous q
    q_big = torch.randn(BT, V * 2, device=DEV, generator=gen).softmax(-1)
    q_nc = q_big[:, :V]
    assert not q_nc.is_contiguous()
    x = p.clone().requires_grad_(True)
    try:
        loss = liger_tvd(x, q_nc, reduction="batchmean")
        loss.backward()
        out["noncontig_q_ok"] = True
        out["noncontig_q_grad_rel"] = rel(x.grad, ref_grad_p(p, q_nc.contiguous(), reduction="batchmean"))
    except Exception as e:
        out["noncontig_q_ok"] = False
        out["noncontig_q_err"] = f"{type(e).__name__}: {e}"

    # module wrapper
    mod = LigerTVDLoss(reduction="batchmean").cuda()
    x = p.clone().requires_grad_(True)
    loss = mod(x, q)
    loss.backward()
    out["module_wrapper_grad_rel"] = rel(x.grad, ref_grad_p(p, q, reduction="batchmean"))
    out["module_wrapper_loss"] = float(loss.detach())

    # torch.compile
    torch._dynamo.reset()
    compiled = torch.compile(LigerTVDLoss(reduction="batchmean").cuda())
    x = p.clone().requires_grad_(True)
    loss_c = compiled(x, q)
    loss_c.backward()
    x2 = p.clone().requires_grad_(True)
    loss_e = LigerTVDLoss(reduction="batchmean").cuda()(x2, q)
    loss_e.backward()
    out["compile_loss_equal"] = bool(torch.equal(loss_c.detach(), loss_e.detach()))
    out["compile_grad_equal"] = bool(torch.equal(x.grad, x2.grad))
    out["compile_loss_rel_vs_eager"] = abs(float(loss_c.detach()) - float(loss_e.detach())) / max(
        abs(float(loss_e.detach())), 1e-30
    )
    out["compile_grad_rel_vs_eager"] = rel(x.grad, x2.grad)

    # loss dtypes
    loss_n = liger_tvd(p, q, reduction="none")
    out["liger_none_dtype"] = str(loss_n.dtype).replace("torch.", "")
    out["liger_none_shape"] = list(loss_n.shape)
    out["liger_batchmean_dtype"] = str(liger_tvd(p, q, reduction="batchmean").dtype).replace("torch.", "")
    out["torch_none_dtype"] = str(torch_tvd(p, q, reduction="none").dtype).replace("torch.", "")
    out["torch_batchmean_dtype"] = str(torch_tvd(p, q, reduction="batchmean").dtype).replace("torch.", "")

    p_bf, q_bf = make_probs(BT, V, torch.bfloat16, gen)
    out["liger_bf16_batchmean_dtype"] = str(liger_tvd(p_bf, q_bf).dtype).replace("torch.", "")
    out["torch_bf16_batchmean_dtype"] = str(torch_tvd(p_bf, q_bf).dtype).replace("torch.", "")

    # fp16 mean: |grad| = 0.5 / (BT*V) may underflow
    p16, q16 = make_probs(16, 32768, torch.float16, gen)
    _, g16_mean = run_fb(liger_tvd, p16, q16, reduction="mean")
    _, g16_bm = run_fb(liger_tvd, p16, q16, reduction="batchmean")
    out["fp16_mean_grad_nonzero_frac"] = float((g16_mean != 0).float().mean())
    out["fp16_batchmean_grad_nonzero_frac"] = float((g16_bm != 0).float().mean())
    out["fp16_mean_grad_abs_max"] = float(g16_mean.float().abs().max())
    out["fp16_batchmean_grad_abs_max"] = float(g16_bm.float().abs().max())
    # expected mean scale: 0.5 / (16*32768) = 9.5e-7, fp16 min normal ~6e-8, so should survive
    # but many ties? report anyway
    out["fp16_mean_expected_abs"] = 0.5 / (16 * 32768)

    # label shape assert
    bad_labels = torch.zeros(BT, 2, device=DEV, dtype=torch.long)
    try:
        liger_tvd(p, q, shift_labels=bad_labels, reduction="batchmean")
        out["bad_label_shape_raises"] = False
    except AssertionError:
        out["bad_label_shape_raises"] = True
    except Exception as e:
        out["bad_label_shape_raises"] = f"{type(e).__name__}"

    return out


# ---------------------------------------------------------------- 5. perf
def measure_cell(fn, p, q, reduction, do_bwd=True, shift_labels=None):
    cleanup()
    torch.cuda.synchronize()
    x = p.detach().clone()
    if do_bwd:
        x.requires_grad_(True)
    torch.cuda.reset_peak_memory_stats()
    mem0 = torch.cuda.memory_allocated()
    loss = fn(x, q, shift_labels=shift_labels, reduction=reduction)
    torch.cuda.synchronize()
    held = (torch.cuda.memory_allocated() - mem0) / 2**20
    peak_fwd = (torch.cuda.max_memory_allocated() - mem0) / 2**20
    if do_bwd:
        torch.cuda.reset_peak_memory_stats()
        mem1 = torch.cuda.memory_allocated()
        if loss.ndim > 0:
            loss.sum().backward()
        else:
            loss.backward()
        torch.cuda.synchronize()
        peak_bwd = (torch.cuda.max_memory_allocated() - mem1) / 2**20
    else:
        peak_bwd = None

    def bench_fwd():
        with torch.no_grad():
            fn(p, q, shift_labels=shift_labels, reduction=reduction)

    def bench_fb():
        y = p.detach().clone().requires_grad_(True)
        lo = fn(y, q, shift_labels=shift_labels, reduction=reduction)
        if lo.ndim > 0:
            lo.sum().backward()
        else:
            lo.backward()

    ms_fwd = do_bench(bench_fwd, warmup=5, rep=20)
    ms_fb = do_bench(bench_fb, warmup=5, rep=20) if do_bwd else None
    BT, V = p.shape
    elem = p.element_size()
    gbps_fwd = (BT * V * elem * 2) / (ms_fwd * 1e-3) / 1e9 if ms_fwd > 0 else None
    return {
        "held_mib": round(held, 2),
        "peak_fwd_mib": round(peak_fwd, 2),
        "peak_bwd_mib": round(peak_bwd, 2) if peak_bwd is not None else None,
        "ms_fwd": round(ms_fwd, 4),
        "ms_fb": round(ms_fb, 4) if ms_fb is not None else None,
        "gbps_fwd_est": round(gbps_fwd, 1) if gbps_fwd is not None else None,
        "input_mib": round(BT * V * elem / 2**20, 2),
    }


def make_probs_bf16(BT, V, gen):
    logits = torch.randn(BT, V, device=DEV, dtype=torch.float32, generator=gen)
    p = logits.softmax(-1).to(torch.bfloat16)
    noise = torch.randn(BT, V, device=DEV, dtype=torch.float32, generator=gen)
    q = (logits * 0.7 + noise).softmax(-1).to(torch.bfloat16)
    del logits, noise
    return p, q


def sec_perf():
    gen = torch.Generator(device=DEV).manual_seed(SEED + 5)
    rows = []
    for vname, V in PERF_VOCABS:
        for BT in PERF_BTS:
            # p+q bf16 + bf16 grads buffer + small fp32 loss
            est_mib = BT * V * 2 * 3 / 2**20
            if est_mib > 3500:
                rows.append({
                    "vocab": vname, "V": V, "BT": BT,
                    "skipped": "est_peak_over_budget", "est_mib": round(est_mib, 1),
                })
                continue
            cleanup()
            try:
                p, q = make_probs_bf16(BT, V, gen)
            except RuntimeError as e:
                rows.append({"vocab": vname, "V": V, "BT": BT, "skipped": str(e)[:200]})
                cleanup()
                continue
            for name, fn in [("liger", liger_tvd), ("torch", torch_tvd)]:
                try:
                    m = measure_cell(fn, p, q, "batchmean", do_bwd=True)
                    m.update({"impl": name, "vocab": vname, "V": V, "BT": BT, "dtype": "bf16"})
                    rows.append(m)
                except RuntimeError as e:
                    rows.append({
                        "impl": name, "vocab": vname, "V": V, "BT": BT,
                        "skipped": f"{type(e).__name__}: {str(e)[:160]}",
                    })
                    cleanup()
            if (V, BT) in COMPILE_CELLS:
                torch._dynamo.reset()

                def _compiled_tvd(p_in, q_in, shift_labels=None, reduction="batchmean", ignore_index=-100):
                    tvd = torch.abs(p_in - q_in) / 2.0
                    if reduction == "batchmean":
                        return tvd.sum() / p_in.shape[0]
                    if reduction == "sum":
                        return tvd.sum()
                    if reduction == "mean":
                        return tvd.mean()
                    return tvd

                compiled = torch.compile(_compiled_tvd)
                try:
                    _ = compiled(p[: min(8, BT)].contiguous(), q[: min(8, BT)].contiguous())
                    torch.cuda.synchronize()
                    m = measure_cell(compiled, p, q, "batchmean", do_bwd=True)
                    m.update({"impl": "compile_torch", "vocab": vname, "V": V, "BT": BT, "dtype": "bf16"})
                    rows.append(m)
                except Exception as e:
                    rows.append({
                        "impl": "compile_torch", "vocab": vname, "V": V, "BT": BT,
                        "skipped": f"{type(e).__name__}: {str(e)[:160]}",
                    })
                    cleanup()
            del p, q
            cleanup()
    return {"rows": rows}


def sec_perf_none():
    gen = torch.Generator(device=DEV).manual_seed(SEED + 6)
    rows = []
    for BT, V in [(64, 32000), (256, 32000), (64, 128256), (128, 128256)]:
        est = BT * V * 4 / 2**20
        if est > 3500:
            rows.append({"BT": BT, "V": V, "skipped": "tile_too_big", "tile_mib": round(est, 1)})
            continue
        cleanup()
        p, q = make_probs_bf16(BT, V, gen)
        # promote for fair none comparison of fp32 tile vs bf16 inputs
        for name, fn in [("liger", liger_tvd), ("torch", torch_tvd)]:
            try:
                m = measure_cell(fn, p, q, "none", do_bwd=False)
                m.update({"impl": name, "BT": BT, "V": V, "reduction": "none"})
                m["expected_fp32_tile_mib"] = round(BT * V * 4 / 2**20, 2)
                rows.append(m)
            except RuntimeError as e:
                rows.append({"impl": name, "BT": BT, "V": V, "skipped": str(e)[:160]})
                cleanup()
        del p, q
        cleanup()
    return {"rows": rows}


def main():
    sections = sys.argv[1:] or ["numerics", "ignore", "ties", "edges", "api", "perf", "perf_none"]
    results = {
        "experiment": "0030-liger-tvd",
        "date": "2026-11-23",
        "seed": SEED,
        "gpu": GPU,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    md = [
        "# 0030 — Liger-Kernel Total Variation Distance results",
        "",
        f"GPU: {GPU.get('name')} | torch {torch.__version__} | seed {SEED}",
        "",
    ]
    t0 = time.time()

    if "numerics" in sections:
        print("== numerics ==")
        data = sec_numerics()
        results["numerics"] = data
        md.append("## Numerics (vs fp64 analytic + eager torch)")
        md.append("")
        md.append(table(
            ["kind", "dtype", "red", "floor", "liger_loss_rel", "torch_loss_rel",
             "liger_grad_rel", "torch_grad_rel", "liger_dtype"],
            [[r["kind"], r["dtype"], r["reduction"], r["loss_round_floor"],
              r["liger_loss_rel"], r["torch_loss_rel"], r["liger_grad_rel"],
              r["torch_grad_rel"], r["liger_loss_dtype"]] for r in data["rows"]],
        ))
        md.append("")

    if "ignore" in sections:
        print("== ignore ==")
        data = sec_ignore()
        results["ignore"] = data
        md.append("## ignore_index")
        md.append("")
        md.append(table(
            ["red", "n_ign", "liger_loss_rel", "torch_loss_rel", "liger_g_rel",
             "liger_ign_g_max", "torch_ign_g_max"],
            [[r["reduction"], r["n_ignore"], r.get("liger_loss_rel"), r.get("torch_loss_rel"),
              r.get("liger_grad_rel"), r.get("liger_ignored_grad_abs_max"),
              r.get("torch_ignored_grad_abs_max")] for r in data["partial"]],
        ))
        md.append("")
        md.append("### All labels ignored")
        md.append("")
        md.append(table(
            ["red", "liger_loss", "torch_loss", "liger_g_max", "torch_g_max"],
            [[r["reduction"], r.get("liger_loss"), r.get("torch_loss"),
              r.get("liger_grad_abs_max"), r.get("torch_grad_abs_max")] for r in data["all_ignored"]],
        ))
        md.append("")

    if "ties" in sections:
        print("== ties ==")
        data = sec_ties()
        results["ties"] = data
        md.append("## Ties / near-ties")
        md.append("")
        md.append(table(
            ["case", "dtype", "red", "liger_loss", "extra"],
            [[r["case"], r.get("dtype"), r.get("reduction"), r.get("liger_loss"),
              {k: r[k] for k in r if k not in ("case", "dtype", "reduction", "liger_loss")}]
             for r in data["rows"]],
        ))
        md.append("")

    if "edges" in sections:
        print("== edges ==")
        data = sec_edges()
        results["edges"] = data
        md.append("## Edge cases")
        md.append("")
        md.append(table(
            ["case", "liger_loss", "torch_loss", "liger_finite_g", "torch_finite_g",
             "liger_g_nan", "loss_rel"],
            [[r["case"], r.get("liger_loss"), r.get("torch_loss"),
              r.get("liger_grad_finite"), r.get("torch_grad_finite"),
              r.get("liger_grad_nan"), r.get("loss_rel")] for r in data["rows"]],
        ))
        md.append("")

    if "api" in sections:
        print("== api ==")
        data = sec_api()
        results["api"] = data
        md.append("## API / autograd")
        md.append("")
        for k, v in data.items():
            md.append(f"- `{k}`: {fmt(v) if not isinstance(v, list) else v}")
        md.append("")

    if "perf" in sections:
        print("== perf ==")
        data = sec_perf()
        results["perf"] = data
        md.append("## Memory / latency (bf16, batchmean)")
        md.append("")

        def _perf_row(r):
            if "skipped" in r:
                sk = r["skipped"]
                if "out of memory" in str(sk).lower():
                    sk = "OOM"
                return [r.get("impl", "-"), r.get("vocab"), r.get("BT"), sk, "-", "-", "-", "-", "-"]
            return [r.get("impl"), r.get("vocab"), r.get("BT"),
                    r.get("held_mib"), r.get("peak_fwd_mib"), r.get("peak_bwd_mib"),
                    r.get("ms_fwd"), r.get("ms_fb"), r.get("gbps_fwd_est")]

        md.append(table(
            ["impl", "vocab", "BT", "held", "peak_f", "peak_b", "ms_fwd", "ms_fb", "GB/s"],
            [_perf_row(r) for r in data["rows"]],
        ))
        md.append("")

    if "perf_none" in sections:
        print("== perf_none ==")
        data = sec_perf_none()
        results["perf_none"] = data
        md.append("## reduction=none fp32 loss tile")
        md.append("")
        md.append(table(
            ["impl", "BT", "V", "held", "peak_f", "ms_fwd", "expected_tile"],
            [[r.get("impl", "-"), r.get("BT"), r.get("V"),
              r.get("held_mib", r.get("skipped")), r.get("peak_fwd_mib"),
              r.get("ms_fwd"), r.get("expected_fp32_tile_mib")] for r in data["rows"]],
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
