"""Liger-Kernel GeGLU (`LigerGELUMulFunction`) on an RTX 4060 Laptop.

`LigerGELUMulFunction` computes gelu_tanh(a) * b in one Triton kernel (one
program per row, BLOCK_SIZE = next_power_of_2(I)). It saves a and b, recomputes
the tanh-approx GELU in the backward pass and writes da / db *into the storage
of the saved a / b*. Those Triton writes bypass autograd's version counter.
Upstream only implements the tanh approximation (a TODO notes exact GELU); the
HF Gemma MLP uses `gelu_pytorch_tanh`.

1. Aliasing: MLP-style graphs where the saved a / b are also visible elsewhere
   (stashed activations, leaf inputs, aux loss before / after, retain_graph
   twice, non-reentrant checkpoint). Gradients vs fp64 eager; stashed tensors
   checked after backward. `liger_oop` = same kernels with out-of-place bwd.
2. Numerics: c / da / db error vs fp64 exact-erf GELU and vs fp64 tanh-approx
   for fp32 / fp16 / bf16; eager F.gelu(approximate='tanh'|'none') /
   torch.compile / Liger / oop; forward cast of gelu(a) to input dtype.
3. Full Gemma-style MLP (gate / up / down Linear, bf16) at Gemma-2B and
   Gemma-2-2B shapes: held / peak memory and fwd+bwd latency vs token count.
4. Elementwise bandwidth vs intermediate size I incl. non-power-of-two padding.

Liger-Kernel is imported unmodified from the submodule.
"""
import gc
import json
import operator
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
import torch.utils.checkpoint  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.testing import do_bench  # noqa: E402
from liger_kernel.ops.geglu import (  # noqa: E402
    LigerGELUMulFunction,
    _geglu_tanh_backward_kernel,
    _geglu_tanh_forward_kernel,
    geglu_backward,
    geglu_forward,
)
from liger_kernel.ops.utils import (  # noqa: E402
    calculate_settings,
    compare_version,
    ensure_contiguous,
)
from liger_kernel.utils import is_npu_available  # noqa: E402

if compare_version("triton", operator.ge, "3.0.0") and not is_npu_available():
    try:
        from triton.language.extra.libdevice import tanh
    except ModuleNotFoundError:
        from triton.language.extra.cuda.libdevice import tanh
else:
    from triton.language.math import tanh

SEED = 0
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
ALIAS_T, ALIAS_H, ALIAS_I = 256, 512, 1536
AUX_WEIGHT = 0.5
NUM_T, NUM_I = 2048, 8192
GATE_SCALES = [1.0, 4.0]
# Gemma-2B (2048→16384) and Gemma-2-2B (2304→9216); keep token counts inside the 60% cap
MLP_SHAPES = [("Gemma-2B", 2048, 16384), ("Gemma-2-2B", 2304, 9216)]
MLP_TOKENS = [1024, 4096, 8192]
MLP_VARIANTS = ["eager_tanh", "eager_exact", "torch_compile", "liger", "liger_oop"]
BW_T = 4096
BW_IS = [2048, 3072, 4096, 4104, 5632, 8192, 9216, 11008, 16384]
BW_ROUNDS = 3
SQRT_2_OVER_PI = 0.7978845608028654

torch._dynamo.config.recompile_limit = 64
torch._functorch.config.donated_buffer = False


# ---------------------------------------------------------------- variants

@triton.jit
def _geglu_bwd_oop_kernel(dc_ptr, a_ptr, b_ptr, da_ptr, db_ptr, stride, n_cols: tl.constexpr,
                          BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    off = row * stride + cols
    dc = tl.load(dc_ptr + off, mask=mask, other=0)
    a = tl.load(a_ptr + off, mask=mask, other=0).to(tl.float32)
    b = tl.load(b_ptr + off, mask=mask, other=0)
    sqrt_2_over_pi = 0.7978845608028654
    a_cubed = a * a * a
    tanh_arg = sqrt_2_over_pi * (a + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a * (1 + tanh_result)
    geglu_a = geglu_a.to(dc.dtype).to(tl.float32)
    db = dc.cast(tl.float32) * geglu_a
    term1 = 0.5 * (1 + tanh_result)
    tanh_sq = tanh_result * tanh_result
    term2 = 0.5 * a * (1 - tanh_sq) * (sqrt_2_over_pi * (1 + 3 * 0.044715 * a * a))
    da = dc * b * (term1 + term2)
    tl.store(da_ptr + off, da, mask=mask)
    tl.store(db_ptr + off, db.to(dc.dtype), mask=mask)


@triton.jit
def _geglu_fwd_fp32_kernel(a_ptr, b_ptr, c_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Upstream forward without casting gelu(a) to input dtype before the multiply."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    off = row * stride + cols
    a = tl.load(a_ptr + off, mask=mask, other=0).to(tl.float32)
    b = tl.load(b_ptr + off, mask=mask, other=0).to(tl.float32)
    sqrt_2_over_pi = 0.7978845608028654
    a_cubed = a * a * a
    tanh_arg = sqrt_2_over_pi * (a + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a * (1 + tanh_result)
    tl.store(c_ptr + off, geglu_a * b, mask=mask)


def geglu_forward_fp32(a, b):
    n = a.shape[-1]
    a2, b2 = a.view(-1, n), b.view(-1, n)
    c = torch.empty_like(a2)
    block, warps = calculate_settings(n)
    _geglu_fwd_fp32_kernel[(a2.shape[0],)](a2, b2, c, a2.stride(0), n_cols=n, BLOCK_SIZE=block, num_warps=warps)
    return c.view(a.shape)


def geglu_backward_oop(a, b, dc):
    shape = dc.shape
    n = shape[-1]
    a2, b2, dc2 = a.view(-1, n), b.view(-1, n), dc.view(-1, n)
    da, db = torch.empty_like(a2), torch.empty_like(b2)
    block, warps = calculate_settings(n)
    _geglu_bwd_oop_kernel[(dc2.shape[0],)](dc2, a2, b2, da, db, dc2.stride(0), n_cols=n, BLOCK_SIZE=block,
                                           num_warps=warps)
    return da.view(shape), db.view(shape)


class OutOfPlaceGELUMul(torch.autograd.Function):
    """Upstream forward kernel; backward writes da / db to fresh buffers instead of a / b."""

    @staticmethod
    @ensure_contiguous
    def forward(ctx, a, b):
        a, b, c = geglu_forward(a, b)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        return geglu_backward_oop(a, b, dc)


def eager_tanh(g, u):
    return F.gelu(g, approximate="tanh") * u


def eager_exact(g, u):
    return F.gelu(g, approximate="none") * u


def liger_act(g, u):
    return LigerGELUMulFunction.apply(g, u)


def oop_act(g, u):
    return OutOfPlaceGELUMul.apply(g, u)


VARIANTS = {
    "eager_tanh": {"act": eager_tanh},
    "eager_exact": {"act": eager_exact},
    "liger": {"act": liger_act},
    "liger_oop": {"act": oop_act},
}


def cuda_gen(seed=SEED):
    return torch.Generator(device="cuda").manual_seed(seed)


def rel(x, ref):
    x, ref = x.detach().double(), ref.detach().double()
    return float((x - ref).norm() / ref.norm().clamp_min(1e-300))


def fmt(v, digits=3):
    return f"{v:.{digits}e}" if isinstance(v, float) else v


def gelu_tanh_fp64(x):
    """fp64 tanh-approx GELU matching the Liger / HF formula."""
    x = x.double()
    return 0.5 * x * (1.0 + torch.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x.pow(3))))


# ---------------------------------------------------------------- 1. aliasing

def alias_params(dtype):
    g = cuda_gen()
    t, h, i = ALIAS_T, ALIAS_H, ALIAS_I
    wg = torch.randn(i, h, device="cuda", generator=g) / h ** 0.5
    wu = torch.randn(i, h, device="cuda", generator=g) / h ** 0.5
    base = {
        "x": torch.randn(t, h, device="cuda", generator=g),
        "wg": wg,
        "wu": wu,
        "wd": torch.randn(h, i, device="cuda", generator=g) / i ** 0.5,
        "a": torch.randn(t, i, device="cuda", generator=g),
        "b": torch.randn(t, i, device="cuda", generator=g),
    }
    loss_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    weights = {"r": torch.randn(t, h, device="cuda", generator=g).to(loss_dtype),
               "rc": torch.randn(t, i, device="cuda", generator=g).to(loss_dtype)}
    leaves = {k: v.to(torch.bfloat16).to(dtype).requires_grad_() for k, v in base.items()}
    return leaves, weights


def weighted(y, r):
    return (y.to(r.dtype) * r).sum()


def mlp_sep(v, p, stash=None):
    g, u = F.linear(p["x"], p["wg"]), F.linear(p["x"], p["wu"])
    if stash is not None:
        stash += [("gate_out", g), ("up_out", u)]
    return F.linear(v["act"](g, u), p["wd"])


def case_mlp(v, p, w):
    return {"loss": weighted(mlp_sep(v, p), w["r"]), "wrt": ["x", "wg", "wu", "wd"]}


def case_stash(v, p, w):
    stash = []
    y = mlp_sep(v, p, stash)
    return {"loss": weighted(y, w["r"]), "wrt": ["x", "wg", "wu", "wd"], "stash": stash}


def case_leaf(v, p, w):
    c = v["act"](p["a"], p["b"])
    return {"loss": weighted(c, w["rc"]), "wrt": ["a", "b"], "stash": [("a", p["a"]), ("b", p["b"])]}


def case_gate_l2(before):
    def case(v, p, w):
        g, u = F.linear(p["x"], p["wg"]), F.linear(p["x"], p["wu"])
        if before:
            aux = g.pow(2).to(w["r"].dtype).sum()
        c = v["act"](g, u)
        if not before:
            aux = g.pow(2).to(w["r"].dtype).sum()
        y = F.linear(c, p["wd"])
        return {"loss": weighted(y, w["r"]) + AUX_WEIGHT * aux, "wrt": ["x", "wg", "wu", "wd"]}
    return case


def case_retain(v, p, w):
    return {"loss": weighted(mlp_sep(v, p), w["r"]), "wrt": ["x", "wg", "wu", "wd"], "twice": True}


def case_checkpoint(v, p, w):
    y = torch.utils.checkpoint.checkpoint(
        lambda x, wg, wu, wd: mlp_sep(v, {"x": x, "wg": wg, "wu": wu, "wd": wd}),
        p["x"], p["wg"], p["wu"], p["wd"], use_reentrant=False)
    return {"loss": weighted(y, w["r"]), "wrt": ["x", "wg", "wu", "wd"]}


ACT_VARIANTS = ["eager_tanh", "liger", "liger_oop"]
ALIAS_CASES = [
    ("mlp", "HF GemmaMLP graph: down(gelu_tanh(gate(x)) * up(x))", case_mlp, ACT_VARIANTS),
    ("stash_gate_up_outputs", "same, gate/up outputs kept (forward hook / activation logging)", case_stash, ACT_VARIANTS),
    ("leaf_inputs", "act(a, b) on leaf tensors", case_leaf, ACT_VARIANTS),
    ("gate_l2_before", "aux = 0.5 * gate.pow(2).sum() built before act", case_gate_l2(True), ACT_VARIANTS),
    ("gate_l2_after", "aux = 0.5 * gate.pow(2).sum() built after act", case_gate_l2(False), ACT_VARIANTS),
    ("retain_graph_twice", "MLP, autograd.grad twice with retain_graph=True", case_retain, ACT_VARIANTS),
    ("checkpoint", "MLP under torch.utils.checkpoint (non-reentrant)", case_checkpoint, ACT_VARIANTS),
]


def run_alias(case_fn, vname, dtype):
    p, w = alias_params(dtype)
    out = case_fn(VARIANTS[vname] if dtype != torch.float64 else VARIANTS["eager_tanh"], p, w)
    wrt = [p[k] for k in out["wrt"]]
    snaps = [(n, t, t.detach().clone()) for n, t in out.get("stash", [])]
    first = None
    if out.get("twice"):
        first = [g.clone() for g in torch.autograd.grad(out["loss"], wrt, retain_graph=True)]
    grads = torch.autograd.grad(out["loss"], wrt)
    res = {"grads": dict(zip(out["wrt"], grads)), "first": dict(zip(out["wrt"], first)) if first else None,
           "stash_changed": {n: float((t.detach() != s).float().mean()) for n, t, s in snaps}}
    if "a" in out["wrt"]:
        res["grad_aliases_input"] = grads[0].data_ptr() == p["a"].data_ptr()
    return res


def alias():
    rows = []
    for cname, desc, case_fn, variants in ALIAS_CASES:
        ref = run_alias(case_fn, "eager_tanh", torch.float64)["grads"]
        for vname in variants:
            row = {"case": cname, "description": desc, "variant": vname}
            try:
                res = run_alias(case_fn, vname, torch.bfloat16)
                errs = {k: rel(g, ref[k]) for k, g in res["grads"].items()}
                row["grad_rel_err_max"] = max(errs.values())
                row["grad_rel_err"] = errs
                if res["first"]:
                    row["first_call_grad_rel_err_max"] = max(rel(g, ref[k]) for k, g in res["first"].items())
                row["stash_changed_fraction"] = res["stash_changed"]
                if "grad_aliases_input" in res:
                    row["grad_aliases_input"] = res["grad_aliases_input"]
                row["error"] = None
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
            rows.append(row)
            print("alias", row, flush=True)
    for row in rows:
        base = next(r for r in rows if r["case"] == row["case"] and r["variant"] == "eager_tanh")
        if row.get("grad_rel_err_max") is not None and base.get("grad_rel_err_max"):
            row["err_vs_eager_bf16"] = round(row["grad_rel_err_max"] / base["grad_rel_err_max"], 2)
    return rows


# ---------------------------------------------------------------- 2. numerics

def numerics():
    rows = []
    compiled_tanh = torch.compile(eager_tanh, dynamic=False)
    compiled_exact = torch.compile(eager_exact, dynamic=False)
    for dname, dtype in DTYPES.items():
        for scale in GATE_SCALES:
            g = cuda_gen()
            a = (scale * torch.randn(NUM_T, NUM_I, device="cuda", generator=g)).to(dtype)
            b = torch.randn(NUM_T, NUM_I, device="cuda", generator=g).to(dtype)
            dc = torch.randn(NUM_T, NUM_I, device="cuda", generator=g).to(dtype)

            a64, b64 = a.double().requires_grad_(), b.double().requires_grad_()
            c64_exact = F.gelu(a64, approximate="none") * b64
            da64_e, db64_e = torch.autograd.grad(c64_exact, (a64, b64), dc.double())
            c64_exact = c64_exact.detach()

            a64t, b64t = a.double().requires_grad_(), b.double().requires_grad_()
            c64_tanh = gelu_tanh_fp64(a64t) * b64t
            da64_t, db64_t = torch.autograd.grad(c64_tanh, (a64t, b64t), dc.double())
            c64_tanh = c64_tanh.detach()

            row = {
                "dtype": dname, "gate_std": scale, "T": NUM_T, "I": NUM_I,
                "floor_tanh": {
                    "c": rel(c64_tanh.to(dtype), c64_tanh),
                    "da": rel(da64_t.to(dtype), da64_t),
                    "db": rel(db64_t.to(dtype), db64_t),
                },
                "tanh_vs_exact_fp64": {
                    "c": rel(c64_tanh, c64_exact),
                    "da": rel(da64_t, da64_e),
                    "db": rel(db64_t, db64_e),
                },
            }

            def grads_of(fn, ref_c, ref_da, ref_db):
                aa, bb = a.clone().requires_grad_(), b.clone().requires_grad_()
                c = fn(aa, bb)
                da, db = torch.autograd.grad(c, (aa, bb), dc)
                return {"c": rel(c, ref_c), "da": rel(da, ref_da), "db": rel(db, ref_db)}

            row["eager_tanh_vs_tanh"] = grads_of(eager_tanh, c64_tanh, da64_t, db64_t)
            row["eager_exact_vs_exact"] = grads_of(eager_exact, c64_exact, da64_e, db64_e)
            row["eager_exact_vs_tanh"] = grads_of(eager_exact, c64_tanh, da64_t, db64_t)
            row["torch_compile_tanh"] = grads_of(compiled_tanh, c64_tanh, da64_t, db64_t)
            row["torch_compile_exact"] = grads_of(compiled_exact, c64_exact, da64_e, db64_e)
            row["liger_vs_tanh"] = grads_of(liger_act, c64_tanh, da64_t, db64_t)
            row["liger_vs_exact"] = grads_of(liger_act, c64_exact, da64_e, db64_e)
            row["liger_oop_vs_tanh"] = grads_of(oop_act, c64_tanh, da64_t, db64_t)
            row["fp32_mul_fwd_c_vs_tanh"] = rel(geglu_forward_fp32(a, b), c64_tanh)

            ua, ub = a.clone(), b.clone()
            dao, dbo = geglu_backward_oop(ua, ub, dc)
            dau, dbu = geglu_backward(ua, ub, dc)
            row["oop_bit_identical_to_upstream"] = bool(torch.equal(dau, dao) and torch.equal(dbu, dbo))

            # Match against F.gelu(approximate='tanh') in the same dtype (HF Gemma path)
            aa, bb = a.clone().requires_grad_(), b.clone().requires_grad_()
            c_hf = eager_tanh(aa, bb)
            da_hf, db_hf = torch.autograd.grad(c_hf, (aa, bb), dc)
            aa2, bb2 = a.clone().requires_grad_(), b.clone().requires_grad_()
            c_l = liger_act(aa2, bb2)
            da_l, db_l = torch.autograd.grad(c_l, (aa2, bb2), dc)
            row["liger_vs_eager_tanh_same_dtype"] = {
                "c": rel(c_l, c_hf), "da": rel(da_l, da_hf), "db": rel(db_l, db_hf),
            }

            rows.append(row)
            print("numerics", {k: row[k] for k in ("dtype", "gate_std", "floor_tanh", "tanh_vs_exact_fp64",
                                                   "liger_vs_tanh", "liger_vs_exact",
                                                   "liger_vs_eager_tanh_same_dtype",
                                                   "oop_bit_identical_to_upstream")}, flush=True)
            del a, b, dc, a64, b64, a64t, b64t, c64_exact, c64_tanh, da64_e, db64_e, da64_t, db64_t
            gc.collect()
            torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------- 3. full MLP

def mlp_fns():
    def sep(act):
        return lambda x, wg, wu, wd: F.linear(act(F.linear(x, wg), F.linear(x, wu)), wd)

    return {
        "eager_tanh": ("sep", sep(eager_tanh)),
        "eager_exact": ("sep", sep(eager_exact)),
        "torch_compile": ("sep", torch.compile(sep(eager_tanh), dynamic=False)),
        "liger": ("sep", sep(liger_act)),
        "liger_oop": ("sep", sep(oop_act)),
    }


def mlp_step_memory(fn, args, dy):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    y = fn(*args)
    torch.cuda.synchronize()
    held = torch.cuda.memory_allocated() - base
    grads = torch.autograd.grad(y, args, dy)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del y, grads
    return held, peak


def mlp_bench():
    rows = []
    fns = mlp_fns()
    for sname, h, i in MLP_SHAPES:
        g = cuda_gen()
        wg = (torch.randn(i, h, device="cuda", generator=g) / h ** 0.5).to(torch.bfloat16)
        wu = (torch.randn(i, h, device="cuda", generator=g) / h ** 0.5).to(torch.bfloat16)
        wd = (torch.randn(h, i, device="cuda", generator=g) / i ** 0.5).to(torch.bfloat16)
        for t in MLP_TOKENS:
            x = torch.randn(t, h, device="cuda", generator=g).to(torch.bfloat16)
            dy = torch.randn(t, h, device="cuda", generator=g).to(torch.bfloat16)
            row = {"shape": sname, "H": h, "I": i, "T": t,
                   "act_MiB": round(t * i * 2 / 2**20, 1),
                   "weight_grads_MiB": round(3 * h * i * 2 / 2**20, 1)}
            for vname, (kind, fn) in fns.items():
                args = [x.clone().requires_grad_()] + [w.clone().requires_grad_() for w in (wg, wu, wd)]
                try:
                    for _ in range(2):
                        torch.autograd.grad(fn(*args), args, dy)
                    held, peak = mlp_step_memory(fn, args, dy)
                    ms = do_bench(lambda: torch.autograd.grad(fn(*args), args, dy), return_mode="median")
                    row[vname] = {"held_after_fwd_MiB": round(held / 2**20, 1),
                                  "extra_peak_MiB": round(peak / 2**20, 1),
                                  "fwd_bwd_ms": round(ms, 3)}
                except torch.OutOfMemoryError:
                    row[vname] = {"error": "OOM"}
                del args
                gc.collect()
                torch.cuda.empty_cache()
            rows.append(row)
            print("mlp", row, flush=True)
            del x, dy
        del wg, wu, wd
        gc.collect()
        torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------- 4. bandwidth

def kernel_resources(a, n):
    block, warps = calculate_settings(n)
    ks = {
        "fwd": _geglu_tanh_forward_kernel.warmup(a, a, a, a.stride(0), n_cols=n, BLOCK_SIZE=block,
                                                 num_warps=warps, grid=(1,)),
        "bwd": _geglu_tanh_backward_kernel.warmup(a, a, a, a.stride(0), n_cols=n, BLOCK_SIZE=block,
                                                  num_warps=warps, grid=(1,)),
        "bwd_oop": _geglu_bwd_oop_kernel.warmup(a, a, a, a, a, a.stride(0), n_cols=n, BLOCK_SIZE=block,
                                                num_warps=warps, grid=(1,)),
    }
    out = []
    for kname, ck in ks.items():
        ck._init_handles()
        out.append({"I": n, "kernel": kname, "BLOCK_SIZE": block, "num_warps": ck.metadata.num_warps,
                    "n_regs": ck.n_regs, "n_spills": ck.n_spills})
    return out


def bandwidth():
    rows, kernels = [], []
    compiled = torch.compile(eager_tanh, dynamic=False)
    for n in BW_IS:
        g = cuda_gen()
        a = torch.randn(BW_T, n, device="cuda", generator=g).to(torch.bfloat16)
        b = torch.randn(BW_T, n, device="cuda", generator=g).to(torch.bfloat16)
        dc = torch.randn(BW_T, n, device="cuda", generator=g).to(torch.bfloat16)
        aw, bw = a.clone(), b.clone()
        ag, bg = a.clone().requires_grad_(), b.clone().requires_grad_()
        c_eager = eager_tanh(ag, bg)
        ac, bc = a.clone().requires_grad_(), b.clone().requires_grad_()
        c_comp = compiled(ac, bc)
        torch.autograd.grad(c_comp, (ac, bc), dc, retain_graph=True)
        out = torch.empty_like(a)
        block, _ = calculate_settings(n)
        nbytes = BW_T * n * a.element_size()
        fns = {
            "copy": (lambda: out.copy_(a), 2),
            "fwd_liger": (lambda: geglu_forward(a, b), 3),
            "fwd_fp32_mul": (lambda: geglu_forward_fp32(a, b), 3),
            "fwd_eager_tanh": (lambda: eager_tanh(a, b), 3),
            "fwd_eager_exact": (lambda: eager_exact(a, b), 3),
            "fwd_compile": (lambda: compiled(a, b), 3),
            "bwd_liger": (lambda: geglu_backward(aw, bw, dc), 5),
            "bwd_liger_oop": (lambda: geglu_backward_oop(a, b, dc), 5),
            "bwd_eager": (lambda: torch.autograd.grad(c_eager, (ag, bg), dc, retain_graph=True), 5),
            "bwd_compile": (lambda: torch.autograd.grad(c_comp, (ac, bc), dc, retain_graph=True), 5),
        }
        do_bench(fns["copy"][0], rep=300)
        times = {k: [] for k in fns}
        for _ in range(BW_ROUNDS):
            for k, (fn, _) in fns.items():
                times[k].append(do_bench(fn, return_mode="median"))
        row = {"T": BW_T, "I": n, "BLOCK_SIZE": block, "lane_util": round(n / block, 3),
               "MiB_per_tensor": round(nbytes / 2**20, 1)}
        for k, (_, mult) in fns.items():
            ms = sorted(times[k])[len(times[k]) // 2]
            row[k + "_ms"] = round(ms, 4)
            row[k + "_gbps"] = round(mult * nbytes * 1e-9 / (ms * 1e-3), 1)
        for k in ("fwd_liger", "fwd_fp32_mul", "fwd_compile", "fwd_eager_exact"):
            row[k + "_vs_eager"] = round(row["fwd_eager_tanh_ms"] / row[k + "_ms"], 2)
        for k in ("bwd_liger", "bwd_liger_oop", "bwd_compile"):
            row[k + "_vs_eager"] = round(row["bwd_eager_ms"] / row[k + "_ms"], 2)
        rows.append(row)
        print("bw", row, flush=True)
        kernels += kernel_resources(a, n)
        del a, b, dc, aw, bw, ag, bg, c_eager, ac, bc, c_comp, out
        gc.collect()
        torch.cuda.empty_cache()
    return rows, kernels


# ---------------------------------------------------------------- report

def upstream_commit() -> str:
    return subprocess.run(["git", "-C", str(LIGER_DIR), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def write_md(result):
    gpu = result["gpu"]
    lines = [
        f"Liger-Kernel {result['upstream_commit'][:10]}, triton-windows {triton.__version__}, torch {torch.__version__}, "
        f"{gpu.get('name', 'cuda')} (memory cap {gpu.get('memory_fraction_cap')}), runtime {result['runtime_s']} s, "
        f"peak allocated {result['peak_memory_mib']} MiB",
        "",
        f"## 1. Aliasing (bf16, T = {ALIAS_T}, H = {ALIAS_H}, I = {ALIAS_I}; max rel L2 grad error over inputs vs fp64 "
        f"eager tanh-approx autograd; `x eager` = error / eager_tanh bf16 error; stash = fraction of elements "
        f"changed by backward)",
        "",
        "| case | variant | grad rel err | x eager | 1st call err | stash changed | grad aliases input | error |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in result["alias"]:
        stash = ", ".join(f"{k} {v:.3f}" for k, v in (r.get("stash_changed_fraction") or {}).items())
        lines.append(
            f"| {r['case']} | {r['variant']} | {fmt(r.get('grad_rel_err_max', ''))} | "
            f"{r.get('err_vs_eager_bf16', '')} | {fmt(r.get('first_call_grad_rel_err_max', ''))} | {stash} | "
            f"{r.get('grad_aliases_input', '')} | {r.get('error') or ''} |"
        )
    lines += ["", "Cases:", ""] + [f"- `{c}`: {d}" for c, d, _, _ in ALIAS_CASES]
    lines += [
        "",
        f"## 2. Error vs fp64 (T = {NUM_T}, I = {NUM_I}; a = std * N(0,1), b, dc ~ N(0,1); rel L2)",
        "",
        "### vs fp64 tanh-approx (Liger / HF Gemma target)",
        "",
        "| dtype | gate std | output | floor | eager_tanh | compile_tanh | liger | liger_oop | fp32-mul fwd |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in result["numerics"]:
        for o in ("c", "da", "db"):
            lines.append(
                f"| {r['dtype']} | {r['gate_std']} | {o} | {fmt(r['floor_tanh'][o])} | "
                f"{fmt(r['eager_tanh_vs_tanh'][o])} | {fmt(r['torch_compile_tanh'][o])} | "
                f"{fmt(r['liger_vs_tanh'][o])} | {fmt(r['liger_oop_vs_tanh'][o])} | "
                f"{fmt(r['fp32_mul_fwd_c_vs_tanh']) if o == 'c' else ''} |"
            )
    lines += [
        "",
        "### vs fp64 exact-erf GELU (and tanh-vs-exact gap)",
        "",
        "| dtype | gate std | output | tanh_vs_exact fp64 | eager_exact | compile_exact | liger_vs_exact | "
        "eager_exact_vs_tanh |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in result["numerics"]:
        for o in ("c", "da", "db"):
            lines.append(
                f"| {r['dtype']} | {r['gate_std']} | {o} | {fmt(r['tanh_vs_exact_fp64'][o])} | "
                f"{fmt(r['eager_exact_vs_exact'][o])} | {fmt(r['torch_compile_exact'][o])} | "
                f"{fmt(r['liger_vs_exact'][o])} | {fmt(r['eager_exact_vs_tanh'][o])} |"
            )
    lines += [
        "",
        "Out-of-place backward bit-identical to upstream: "
        + ", ".join(f"{r['dtype']}/{r['gate_std']} {r['oop_bit_identical_to_upstream']}"
                    for r in result["numerics"]),
        "",
        "Liger vs same-dtype eager `F.gelu(approximate='tanh')`: "
        + "; ".join(
            f"{r['dtype']}/std{r['gate_std']}: c {fmt(r['liger_vs_eager_tanh_same_dtype']['c'])}, "
            f"da {fmt(r['liger_vs_eager_tanh_same_dtype']['da'])}, "
            f"db {fmt(r['liger_vs_eager_tanh_same_dtype']['db'])}"
            for r in result["numerics"]
        ),
    ]
    names = MLP_VARIANTS
    lines += [
        "",
        "## 3. Full Gemma MLP fwd + bwd (bf16; held = memory still allocated after forward incl. output; "
        "extra peak = max allocated during fwd + bwd above the inputs, incl. weight grads; do_bench median)",
        "",
    ]
    for metric, title in (("held_after_fwd_MiB", "held after forward (MiB)"),
                          ("extra_peak_MiB", "extra peak (MiB)"),
                          ("fwd_bwd_ms", "fwd + bwd (ms)")):
        lines += [f"### {title}", "", "| shape | T | [T, I] MiB | " + " | ".join(names) + " |",
                  "|---|---|---|" + "---|" * len(names)]
        for r in result["mlp"]:
            cells = [str(r[n].get(metric, r[n].get("error"))) for n in names]
            lines.append(f"| {r['shape']} | {r['T']} | {r['act_MiB']} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        f"## 4. Elementwise bandwidth (bf16, T = {BW_T}; fwd 3 * T * I * 2 bytes, bwd 5 * T * I * 2 bytes; "
        "do_bench median of 3 rounds; `xE` = speedup vs eager_tanh)",
        "",
        "| I | BLOCK | lanes used | copy | fwd liger | fwd fp32-mul | fwd eager_tanh | fwd eager_exact | "
        "fwd compile | bwd liger | bwd oop | bwd eager | bwd compile | fwd liger xE | bwd liger xE | "
        "bwd compile xE |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in result["bandwidth"]:
        lines.append(
            f"| {r['I']} | {r['BLOCK_SIZE']} | {r['lane_util']} | {r['copy_gbps']} | {r['fwd_liger_gbps']} | "
            f"{r['fwd_fp32_mul_gbps']} | {r['fwd_eager_tanh_gbps']} | {r['fwd_eager_exact_gbps']} | "
            f"{r['fwd_compile_gbps']} | {r['bwd_liger_gbps']} | {r['bwd_liger_oop_gbps']} | "
            f"{r['bwd_eager_gbps']} | {r['bwd_compile_gbps']} | {r['fwd_liger_vs_eager']} | "
            f"{r['bwd_liger_vs_eager']} | {r['bwd_compile_vs_eager']} |"
        )
    lines += ["", "### Kernel resources", "",
              "| I | kernel | BLOCK | warps | regs/thread | spills (bytes) |",
              "|---|---|---|---|---|---|"]
    lines += [
        f"| {k['I']} | {k['kernel']} | {k['BLOCK_SIZE']} | {k['num_warps']} | {k['n_regs']} | {k['n_spills']} |"
        for k in result["kernels"]
    ]
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    t_start = time.time()
    torch.manual_seed(SEED)
    alias_rows = alias()
    numeric_rows = numerics()
    mlp_rows = mlp_bench()
    bw_rows, kernel_rows = bandwidth()
    result = {
        "experiment": HERE.name,
        "upstream": "linkedin/Liger-Kernel",
        "upstream_commit": upstream_commit(),
        "component": "src/liger_kernel/ops/geglu.py (LigerGELUMulFunction)",
        "triton": triton.__version__,
        "gpu": GPU,
        "alias": alias_rows,
        "numerics": numeric_rows,
        "mlp": mlp_rows,
        "bandwidth": bw_rows,
        "kernels": kernel_rows,
        "peak_memory_mib": gpu_guard.peak_memory_mib(),
        "runtime_s": round(time.time() - t_start, 1),
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_md(result)


if __name__ == "__main__":
    main()
