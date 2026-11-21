"""Liger-Kernel Triton KL divergence (`LigerKLDivLossFunction`) on an RTX 4060 Laptop.

`kldiv_forward_triton` launches one program per row with BLOCK_SIZE =
min(16384, next_power_of_2(V)), accumulates an fp32 per-row (or per-element)
loss buffer, then reduces in the wrapper like PyTorch's KLDivLoss
(batchmean / sum / mean / none). Backward writes -target (or -exp(target)
when log_target) into a fresh buffer, optionally scales by grad_output, and
applies the reduction divisor; target gets no gradient.

1. Numerics vs fp64 analytic + torch.nn.functional.kl_div for all reductions,
   log_target True/False, fp32/fp16/bf16; loss output dtype; eps clamping.
2. Edge cases: zero / negative / >1 targets, -inf / +inf / NaN in input or
   target, sparse one-hot targets, padded vocab zeros, eps sweep.
3. Autograd API: grad_output == 1.0 skip via torch.equal, scaled grad_output,
   reduction="none" (no backward in upstream tests), retain_graph aliasing,
   no target gradient, non-contiguous inputs, torch.compile vs eager.
4. Memory / latency vs token count at 32k / 128k / 152k vocab for upstream vs
   F.kl_div vs torch.compile(F.kl_div).

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
from triton.testing import do_bench  # noqa: E402
from liger_kernel.ops.kl_div import LigerKLDivLossFunction  # noqa: E402
from liger_kernel.transformers.kl_div import LigerKLDIVLoss  # noqa: E402

SEED = 0
DEV = "cuda"
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
REDUCTIONS = ["batchmean", "sum", "mean", "none"]
NUM_BT, NUM_V = 64, 32000
EPS_DEFAULT = 1e-10
PERF_VOCABS = [("32k", 32000), ("Llama-3 128k", 128256), ("Qwen3 152k", 151936)]
PERF_BTS = [256, 1024, 2048, 3072, 4096]
COMPILE_CELLS = {(32000, 2048), (128256, 1024), (151936, 1024)}

torch._dynamo.config.recompile_limit = 64
torch._functorch.config.donated_buffer = False


def liger_kl(input, target, reduction="batchmean", log_target=False, eps=EPS_DEFAULT):
    return LigerKLDivLossFunction.apply(input, target, reduction, log_target, eps)


def torch_kl(input, target, reduction="batchmean", log_target=False, eps=None):
    # torch has no eps; ignore the unused kwarg so call sites stay uniform
    return F.kl_div(input, target, reduction=reduction, log_target=log_target)


def ref_kl(input, target, reduction="batchmean", log_target=False, eps=EPS_DEFAULT):
    """fp64 analytic KL(target || input) with input in log-space."""
    inp = input.double()
    tgt = target.double()
    if log_target:
        # 0 * (-inf) -> 0 via where; match torch xlogy-ish: exp(t)*(t - inp)
        el = torch.where(
            tgt == float("-inf"),
            torch.zeros_like(tgt),
            torch.exp(tgt) * (tgt - inp),
        )
    else:
        # target * (log(max(target, eps)) - input); zeros contribute 0 when using
        # the torch convention target * (log(target) - input) with 0*log(0)=0,
        # but Liger clamps with eps so zeros become eps * (log(eps) - input).
        el = tgt * (torch.log(torch.clamp(tgt, min=eps)) - inp)
    if reduction == "none":
        return el
    if reduction == "sum":
        return el.sum()
    if reduction == "mean":
        return el.sum() / el.numel()
    if reduction == "batchmean":
        return el.sum() / el.shape[0]
    raise ValueError(reduction)


def ref_grad(input, target, reduction="batchmean", log_target=False):
    """Analytic d KL / d input (log-probs as free vars). Independent of eps."""
    tgt = target.double()
    if log_target:
        g = -torch.exp(tgt)
    else:
        g = -tgt
    if reduction == "batchmean":
        g = g / tgt.shape[0]
    elif reduction == "mean":
        g = g / tgt.numel()
    elif reduction in ("sum", "none"):
        pass
    else:
        raise ValueError(reduction)
    return g


def make_pair(BT, V, dtype, log_target, gen, kind="soft"):
    logits = torch.randn(BT, V, device=DEV, dtype=torch.float64, generator=gen)
    if kind == "peaked":
        logits = logits * 3.0
    elif kind == "flat":
        logits = logits * 0.05
    inp = logits.log_softmax(-1).to(dtype)
    if log_target:
        # teacher from a shifted distribution
        t_logits = logits * 0.7 + torch.randn(BT, V, device=DEV, dtype=torch.float64, generator=gen)
        tgt = t_logits.log_softmax(-1).to(dtype)
    else:
        t_logits = logits * 0.7 + torch.randn(BT, V, device=DEV, dtype=torch.float64, generator=gen)
        tgt = t_logits.softmax(-1).to(dtype)
    return inp, tgt


def run_fb(fn, inp, tgt, **kw):
    x = inp.detach().clone().requires_grad_(True)
    loss = fn(x, tgt, **kw)
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


# ---------------------------------------------------------------- 1. numerics
def sec_numerics():
    gen = torch.Generator(device=DEV).manual_seed(SEED)
    rows = []
    for log_target in [False, True]:
        for dname, dt in DTYPES.items():
            inp, tgt = make_pair(NUM_BT, NUM_V, dt, log_target, gen)
            for reduction in REDUCTIONS:
                lref = ref_kl(inp, tgt, reduction, log_target)
                gref = ref_grad(inp, tgt, reduction, log_target)
                # torch reference uses 0*log(0)=0, not eps clamp
                lt = torch_kl(inp, tgt, reduction=reduction, log_target=log_target)
                if reduction == "none":
                    floor = rel(lref.to(dt), lref)
                else:
                    floor = abs(float(lref.to(dt).double()) - float(lref)) / max(abs(float(lref)), 1e-30)
                r = {
                    "log_target": log_target,
                    "dtype": dname,
                    "reduction": reduction,
                    "loss_ref": float(lref.double().sum()) if reduction == "none" else float(lref),
                    "loss_round_floor": floor,
                }
                for name, fn in [("liger", liger_kl), ("torch", torch_kl)]:
                    loss, g = run_fb(fn, inp, tgt, reduction=reduction, log_target=log_target)
                    r[name + "_loss_dtype"] = str(loss.dtype).replace("torch.", "")
                    if reduction == "none":
                        r[name + "_loss_rel"] = rel(loss, lref)
                        r[name + "_grad_rel"] = rel(g, gref) if g is not None else None
                    else:
                        r[name + "_loss_rel"] = abs(float(loss.double()) - float(lref)) / max(abs(float(lref)), 1e-30)
                        r[name + "_grad_rel"] = rel(g, gref) if g is not None else None
                    r[name + "_vs_torch_loss_rel"] = rel(loss, lt)
                rows.append(r)
    return {"rows": rows}


def sec_eps_vs_torch():
    """Liger clamps target with eps before log; torch uses 0*log(0)=0."""
    gen = torch.Generator(device=DEV).manual_seed(SEED + 1)
    BT, V = 32, 4096
    inp = torch.randn(BT, V, device=DEV, generator=gen).log_softmax(-1)
    tgt = torch.randn(BT, V, device=DEV, generator=gen).softmax(-1)
    # force a sparse support: zero out 90% of each row and renormalize
    mask = torch.rand(BT, V, device=DEV, generator=gen) > 0.9
    tgt = tgt * mask
    tgt = tgt / tgt.sum(-1, keepdim=True).clamp_min(1e-30)
    rows = []
    for eps in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
        ll, gl = run_fb(liger_kl, inp, tgt, reduction="batchmean", log_target=False, eps=eps)
        lt, gt = run_fb(torch_kl, inp, tgt, reduction="batchmean", log_target=False)
        lref_liger = ref_kl(inp, tgt, "batchmean", False, eps=eps)
        lref_torch = ref_kl(inp, tgt, "batchmean", False, eps=0.0)
        # torch-style: use where for zeros
        el = torch.where(tgt.double() > 0, tgt.double() * (tgt.double().log() - inp.double()), 0.0)
        lref_xlogy = el.sum() / BT
        rows.append({
            "eps": eps,
            "liger_loss": float(ll),
            "torch_loss": float(lt),
            "ref_eps_clamp": float(lref_liger),
            "ref_xlogy": float(lref_xlogy),
            "liger_vs_torch_loss_rel": abs(float(ll) - float(lt)) / max(abs(float(lt)), 1e-30),
            "liger_vs_ref_eps": abs(float(ll) - float(lref_liger)) / max(abs(float(lref_liger)), 1e-30),
            "torch_vs_xlogy": abs(float(lt) - float(lref_xlogy)) / max(abs(float(lref_xlogy)), 1e-30),
            "grad_rel_liger_torch": rel(gl, gt),
        })
    return {"zero_frac": 0.9, "rows": rows}


# ------------------------------------------------------------- 2. edge cases
def sec_edges():
    gen = torch.Generator(device=DEV).manual_seed(SEED + 2)
    BT, V = 16, 8192
    base_inp = torch.randn(BT, V, device=DEV, generator=gen).log_softmax(-1)
    base_tgt = torch.randn(BT, V, device=DEV, generator=gen).softmax(-1)
    scenarios = {}

    # padded vocab: last 512 cols zero in target
    inp, tgt = base_inp.clone(), base_tgt.clone()
    tgt[:, -512:] = 0
    tgt = tgt / tgt.sum(-1, keepdim=True)
    scenarios["pad_zeros_512"] = (inp, tgt, False)

    # one-hot targets
    inp = base_inp.clone()
    tgt = torch.zeros_like(base_tgt)
    idx = torch.randint(0, V, (BT,), device=DEV, generator=gen)
    tgt[torch.arange(BT, device=DEV), idx] = 1.0
    scenarios["one_hot"] = (inp, tgt, False)

    # negative target mass
    inp, tgt = base_inp.clone(), base_tgt.clone()
    tgt[:, 0] = -0.1
    scenarios["neg_target"] = (inp, tgt, False)

    # target > 1
    inp, tgt = base_inp.clone(), base_tgt.clone()
    tgt[:, 0] = 1.5
    scenarios["gt1_target"] = (inp, tgt, False)

    # -inf in log-input (impossible class under student)
    inp, tgt = base_inp.clone(), base_tgt.clone()
    inp[:, :64] = float("-inf")
    scenarios["neg_inf_input"] = (inp, tgt, False)

    # +inf in input
    inp, tgt = base_inp.clone(), base_tgt.clone()
    inp[0, 0] = float("inf")
    scenarios["pos_inf_input"] = (inp, tgt, False)

    # NaN in input
    inp, tgt = base_inp.clone(), base_tgt.clone()
    inp[0, 0] = float("nan")
    scenarios["nan_input"] = (inp, tgt, False)

    # NaN in target
    inp, tgt = base_inp.clone(), base_tgt.clone()
    tgt[0, 0] = float("nan")
    scenarios["nan_target"] = (inp, tgt, False)

    # log_target with -inf teacher (zero prob)
    inp = base_inp.clone()
    tgt = torch.randn(BT, V, device=DEV, generator=gen).log_softmax(-1)
    tgt[:, -256:] = float("-inf")
    scenarios["log_target_neg_inf_pad"] = (inp, tgt, True)

    # log_target with +inf
    inp = base_inp.clone()
    tgt = torch.randn(BT, V, device=DEV, generator=gen).log_softmax(-1)
    tgt[0, 0] = float("inf")
    scenarios["log_target_pos_inf"] = (inp, tgt, True)

    # empty-ish V=1
    inp1 = torch.zeros(BT, 1, device=DEV)
    tgt1 = torch.ones(BT, 1, device=DEV)
    scenarios["v1"] = (inp1, tgt1, False)

    rows = []
    for name, (inp, tgt, log_target) in scenarios.items():
        r = {"case": name, "log_target": log_target, "shape": list(inp.shape)}
        for impl, fn in [("liger", liger_kl), ("torch", torch_kl)]:
            try:
                loss, g = run_fb(fn, inp, tgt, reduction="batchmean", log_target=log_target)
                lv = float(loss.double()) if torch.isfinite(loss).all() else (
                    "nan" if torch.isnan(loss).any() else ("inf" if (loss == float("inf")).any() else "-inf")
                )
                if isinstance(lv, float) and (math.isnan(lv) or math.isinf(lv)):
                    lv = "nan" if math.isnan(lv) else ("inf" if lv > 0 else "-inf")
                r[impl + "_loss"] = lv
                r[impl + "_grad_finite"] = bool(g is not None and torch.isfinite(g).all())
                r[impl + "_grad_nan"] = bool(g is not None and torch.isnan(g).any())
                r[impl + "_grad_abs_max"] = float(g.double().abs().max()) if g is not None and torch.isfinite(g).any() else None
            except Exception as e:
                r[impl + "_loss"] = f"ERR:{type(e).__name__}"
                r[impl + "_grad_finite"] = False
                r[impl + "_grad_nan"] = None
                r[impl + "_grad_abs_max"] = None
        rows.append(r)
    return {"rows": rows}


# ------------------------------------------------------------- 3. API / autograd
def sec_api():
    gen = torch.Generator(device=DEV).manual_seed(SEED + 3)
    BT, V = 32, 4096
    inp, tgt = make_pair(BT, V, torch.float32, False, gen)
    out = {}

    # grad_output == 1.0 path: loss is last layer
    x = inp.clone().requires_grad_(True)
    loss = liger_kl(x, tgt, reduction="batchmean")
    loss.backward()
    g_last = x.grad.detach().clone()
    out["last_layer_grad_rel_vs_ref"] = rel(g_last, ref_grad(inp, tgt, "batchmean"))

    # scaled: loss * 2.5
    x = inp.clone().requires_grad_(True)
    loss = liger_kl(x, tgt, reduction="batchmean")
    (loss * 2.5).backward()
    g_scaled = x.grad.detach().clone()
    out["scaled_2p5_grad_rel_vs_2p5_ref"] = rel(g_scaled, 2.5 * ref_grad(inp, tgt, "batchmean"))
    out["scaled_over_last"] = float(g_scaled.norm() / g_last.norm())

    # torch.equal(grad_output, tensor(1.0)) — non-scalar? reduction none returns [BT,V]
    # Upstream skips mul only when grad_output equals scalar 1.0 exactly.
    # Probe: does a 1.0-filled tensor of wrong shape still skip?
    x = inp.clone().requires_grad_(True)
    loss_none = liger_kl(x, tgt, reduction="none")
    # backward with ones — grad_output is ones_like(loss), not scalar 1.0
    loss_none.backward(torch.ones_like(loss_none))
    g_none = x.grad.detach().clone()
    out["none_ones_grad_rel_vs_ref"] = rel(g_none, ref_grad(inp, tgt, "none"))
    # Compare elementwise: for none, ref grad is -target; with ones grad_output should match
    out["none_vs_batchmean_scaled"] = rel(g_none / BT, g_last)

    # target has no gradient
    x = inp.clone().requires_grad_(True)
    t = tgt.clone().requires_grad_(True)
    loss = liger_kl(x, t, reduction="batchmean")
    loss.backward()
    out["target_grad_is_none"] = t.grad is None
    out["input_grad_is_not_none"] = x.grad is not None

    # retain_graph second backward
    x = inp.clone().requires_grad_(True)
    loss = liger_kl(x, tgt, reduction="batchmean")
    loss.backward(retain_graph=True)
    g1 = x.grad.detach().clone()
    x.grad = None
    loss.backward()
    g2 = x.grad.detach().clone()
    out["retain_graph_second_rel"] = rel(g2, g1)

    # non-contiguous input / target
    x_nc = inp.t().contiguous().t()  # still contiguous actually
    # make truly non-contiguous via as_strided / slice
    big = torch.randn(BT, V * 2, device=DEV, generator=gen)
    inp_nc = big[:, ::2].log_softmax(-1)  # contiguous after log_softmax
    # force non-contig view of probabilities
    tgt_big = torch.randn(BT, V * 2, device=DEV, generator=gen).softmax(-1)
    tgt_nc = tgt_big[:, :V]
    assert not tgt_nc.is_contiguous()
    x = inp.clone().requires_grad_(True)
    try:
        loss = liger_kl(x, tgt_nc, reduction="batchmean")
        loss.backward()
        out["noncontig_target_ok"] = True
        out["noncontig_target_grad_rel"] = rel(x.grad, ref_grad(inp, tgt_nc.contiguous(), "batchmean"))
    except Exception as e:
        out["noncontig_target_ok"] = False
        out["noncontig_target_err"] = f"{type(e).__name__}: {e}"

    # Module wrapper eps
    mod = LigerKLDIVLoss(reduction="batchmean", log_target=False, eps=1e-6).cuda()
    x = inp.clone().requires_grad_(True)
    loss = mod(x, tgt)
    loss.backward()
    out["module_wrapper_grad_rel"] = rel(x.grad, ref_grad(inp, tgt, "batchmean"))
    out["module_wrapper_loss"] = float(loss.detach())

    # torch.compile
    torch._dynamo.reset()
    compiled = torch.compile(LigerKLDIVLoss(reduction="batchmean", log_target=False).cuda())
    x = inp.clone().requires_grad_(True)
    loss_c = compiled(x, tgt)
    loss_c.backward()
    x2 = inp.clone().requires_grad_(True)
    loss_e = LigerKLDIVLoss(reduction="batchmean", log_target=False).cuda()(x2, tgt)
    loss_e.backward()
    out["compile_loss_rel_vs_eager"] = (
        abs(float(loss_c.detach()) - float(loss_e.detach())) / max(abs(float(loss_e.detach())), 1e-30)
    )
    out["compile_grad_rel_vs_eager"] = rel(x.grad, x2.grad)
    out["compile_loss_dtype"] = str(loss_c.dtype).replace("torch.", "")

    # reduction none: does liger return fp32 always?
    loss_n = liger_kl(inp, tgt, reduction="none")
    out["none_loss_dtype"] = str(loss_n.dtype).replace("torch.", "")
    out["none_loss_shape"] = list(loss_n.shape)
    loss_bm = liger_kl(inp, tgt, reduction="batchmean")
    out["batchmean_loss_dtype"] = str(loss_bm.dtype).replace("torch.", "")
    # torch dtypes for comparison
    out["torch_none_dtype"] = str(torch_kl(inp, tgt, reduction="none").dtype).replace("torch.", "")
    out["torch_batchmean_dtype"] = str(torch_kl(inp, tgt, reduction="batchmean").dtype).replace("torch.", "")

    # bf16 input -> loss dtype
    inp_bf, tgt_bf = make_pair(BT, V, torch.bfloat16, False, gen)
    out["liger_bf16_batchmean_dtype"] = str(liger_kl(inp_bf, tgt_bf).dtype).replace("torch.", "")
    out["torch_bf16_batchmean_dtype"] = str(torch_kl(inp_bf, tgt_bf).dtype).replace("torch.", "")

    # fp16 mean: grad stored in input dtype then / (BT*V) underflows
    inp16, tgt16 = make_pair(16, 32768, torch.float16, False, gen)
    _, g16_mean = run_fb(liger_kl, inp16, tgt16, reduction="mean")
    _, g16_bm = run_fb(liger_kl, inp16, tgt16, reduction="batchmean")
    out["fp16_mean_grad_nonzero_frac"] = float((g16_mean != 0).float().mean())
    out["fp16_batchmean_grad_nonzero_frac"] = float((g16_bm != 0).float().mean())
    out["fp16_mean_grad_abs_max"] = float(g16_mean.float().abs().max())
    out["fp16_batchmean_grad_abs_max"] = float(g16_bm.float().abs().max())

    return out


# ------------------------------------------------------------- 4. memory / latency
def measure_cell(fn, inp, tgt, reduction, log_target, do_bwd=True):
    cleanup()
    torch.cuda.synchronize()
    # held after forward
    x = inp.detach().clone()
    if do_bwd:
        x.requires_grad_(True)
    torch.cuda.reset_peak_memory_stats()
    mem0 = torch.cuda.memory_allocated()
    loss = fn(x, tgt, reduction=reduction, log_target=log_target)
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
            fn(inp, tgt, reduction=reduction, log_target=log_target)

    def bench_fb():
        y = inp.detach().clone().requires_grad_(True)
        lo = fn(y, tgt, reduction=reduction, log_target=log_target)
        if lo.ndim > 0:
            lo.sum().backward()
        else:
            lo.backward()

    ms_fwd = do_bench(bench_fwd, warmup=5, rep=20)
    ms_fb = do_bench(bench_fb, warmup=5, rep=20) if do_bwd else None
    # bytes: input + target read, loss write (rough)
    BT, V = inp.shape
    elem = inp.element_size()
    bytes_rw = BT * V * elem * (2 + (1 if do_bwd else 0))  # rough
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


def make_pair_bf16(BT, V, gen):
    """bf16 pair without a lasting fp64 workspace (perf only)."""
    logits = torch.randn(BT, V, device=DEV, dtype=torch.float32, generator=gen)
    inp = logits.log_softmax(-1).to(torch.bfloat16)
    noise = torch.randn(BT, V, device=DEV, dtype=torch.float32, generator=gen)
    tgt = (logits * 0.7 + noise).softmax(-1).to(torch.bfloat16)
    del logits, noise
    return inp, tgt


def sec_perf():
    gen = torch.Generator(device=DEV).manual_seed(SEED + 4)
    rows = []
    for vname, V in PERF_VOCABS:
        for BT in PERF_BTS:
            # bf16 input+target + bf16 grad + small temps; refuse past ~3.5 GiB of live tensors
            est_mib = BT * V * 2 * 3 / 2**20
            if est_mib > 3500:
                rows.append({"vocab": vname, "V": V, "BT": BT, "skipped": "est_peak_over_budget",
                             "est_mib": round(est_mib, 1)})
                continue
            cleanup()
            try:
                inp, tgt = make_pair_bf16(BT, V, gen)
            except RuntimeError as e:
                rows.append({"vocab": vname, "V": V, "BT": BT, "skipped": str(e)[:200]})
                cleanup()
                continue
            for name, fn in [("liger", liger_kl), ("torch", torch_kl)]:
                try:
                    m = measure_cell(fn, inp, tgt, "batchmean", False, do_bwd=True)
                    m.update({"impl": name, "vocab": vname, "V": V, "BT": BT, "dtype": "bf16"})
                    rows.append(m)
                except RuntimeError as e:
                    rows.append({"impl": name, "vocab": vname, "V": V, "BT": BT,
                                 "skipped": f"{type(e).__name__}: {str(e)[:160]}"})
                    cleanup()
            if (V, BT) in COMPILE_CELLS:
                torch._dynamo.reset()
                _ckl = torch.compile(
                    lambda i, t: F.kl_div(i, t, reduction="batchmean", log_target=False)
                )

                def _compiled_kl(input, target, reduction="batchmean", log_target=False, eps=None):
                    return _ckl(input, target)

                try:
                    _ = _compiled_kl(inp[: min(8, BT)].contiguous(), tgt[: min(8, BT)].contiguous())
                    torch.cuda.synchronize()
                    m = measure_cell(_compiled_kl, inp, tgt, "batchmean", False, do_bwd=True)
                    m.update({"impl": "compile_torch", "vocab": vname, "V": V, "BT": BT, "dtype": "bf16"})
                    rows.append(m)
                except Exception as e:
                    rows.append({"impl": "compile_torch", "vocab": vname, "V": V, "BT": BT,
                                 "skipped": f"{type(e).__name__}: {str(e)[:160]}"})
                    cleanup()
            del inp, tgt
            cleanup()
    return {"rows": rows}


def sec_perf_fp32_none():
    """reduction=none materializes (BT, V) fp32 loss — quantify the tile."""
    gen = torch.Generator(device=DEV).manual_seed(SEED + 5)
    rows = []
    for BT, V in [(64, 32000), (256, 32000), (64, 128256), (128, 128256)]:
        est = BT * V * 4 / 2**20  # fp32 tile alone
        if est > 3500:
            rows.append({"BT": BT, "V": V, "skipped": "tile_too_big", "tile_mib": round(est, 1)})
            continue
        cleanup()
        logits = torch.randn(BT, V, device=DEV, dtype=torch.float32, generator=gen)
        inp = logits.log_softmax(-1)
        tgt = (logits * 0.7 + torch.randn(BT, V, device=DEV, dtype=torch.float32, generator=gen)).softmax(-1)
        del logits
        for name, fn in [("liger", liger_kl), ("torch", torch_kl)]:
            try:
                m = measure_cell(fn, inp, tgt, "none", False, do_bwd=False)
                m.update({"impl": name, "BT": BT, "V": V, "reduction": "none"})
                m["expected_fp32_tile_mib"] = round(BT * V * 4 / 2**20, 2)
                rows.append(m)
            except RuntimeError as e:
                rows.append({"impl": name, "BT": BT, "V": V, "skipped": str(e)[:160]})
                cleanup()
        del inp, tgt
        cleanup()
    return {"rows": rows}


def main():
    sections = sys.argv[1:] or ["numerics", "eps", "edges", "api", "perf", "perf_none"]
    results = {
        "experiment": "0028-liger-kl-div",
        "date": "2026-11-21",
        "seed": SEED,
        "gpu": GPU,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    md = [
        "# 0028 — Liger-Kernel KL divergence results",
        "",
        f"GPU: {GPU.get('name')} | torch {torch.__version__} | seed {SEED}",
        "",
    ]
    t0 = time.time()

    if "numerics" in sections:
        print("== numerics ==")
        data = sec_numerics()
        results["numerics"] = data
        md.append("## Numerics (vs fp64 analytic + torch)")
        md.append("")
        md.append(table(
            ["log_t", "dtype", "red", "floor", "liger_loss_rel", "torch_loss_rel",
             "liger_grad_rel", "torch_grad_rel", "liger_dtype"],
            [[r["log_target"], r["dtype"], r["reduction"], r["loss_round_floor"],
              r["liger_loss_rel"], r["torch_loss_rel"], r["liger_grad_rel"],
              r["torch_grad_rel"], r["liger_loss_dtype"]] for r in data["rows"]],
        ))
        md.append("")

    if "eps" in sections:
        print("== eps ==")
        data = sec_eps_vs_torch()
        results["eps"] = data
        md.append("## Eps clamp vs torch xlogy (90% zeros per row)")
        md.append("")
        md.append(table(
            ["eps", "liger", "torch", "ref_eps", "xlogy", "liger_vs_torch", "grad_rel"],
            [[r["eps"], r["liger_loss"], r["torch_loss"], r["ref_eps_clamp"],
              r["ref_xlogy"], r["liger_vs_torch_loss_rel"], r["grad_rel_liger_torch"]]
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
             "liger_g_nan", "torch_g_nan"],
            [[r["case"], r.get("liger_loss"), r.get("torch_loss"),
              r.get("liger_grad_finite"), r.get("torch_grad_finite"),
              r.get("liger_grad_nan"), r.get("torch_grad_nan")] for r in data["rows"]],
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
        data = sec_perf_fp32_none()
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
