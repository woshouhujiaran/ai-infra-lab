"""Liger-Kernel Triton RoPE (`LigerRopeFunction`) on an RTX 4060 Laptop.

`rope_forward` / `rope_backward` transpose Q/K to (B, S, H, D), make them
contiguous, then run an in-place Triton kernel that rotates the leading
`rotary_dim` entries of every head (Llama split-half layout). Cos/sin may be
duplicated (`cat(freqs, freqs)`, width == rotary_dim) or half-width (GPT-OSS).

1. API / layout footguns: `position_ids` is accepted but never used; in-place
   writes mutate the caller's storage when transpose+contiguous is a no-op
   (BSHD contiguous → BHSD view); GPT-OSS half-width cos without `rotary_dim`
   is silently treated as partial RoPE over the first half of each head.
2. Partial RoPE: the non-rotary tail `[rd:hd]` stays untouched; GQA (n_q != n_kv).
3. Numerics: Q/K and dQ/dK vs fp64 and HF `apply_rotary_pos_emb` for full /
   partial / GPT-OSS-style half-width in fp32 / fp16 / bf16.
4. Autograd: `retain_graph` twice with fresh vs reused `dQ`/`dK` (backward is
   also in-place on the contiguous dq/dk tiles).
5. Memory / latency / bandwidth vs eager HF and `torch.compile` at Llama-ish
   and Qwen GQA shapes.

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
from transformers.models.llama.modeling_llama import (  # noqa: E402
    apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
)
from liger_kernel.ops.rope import (  # noqa: E402
    LigerRopeFunction,
    _triton_rope,
    rope_backward,
    rope_forward,
)
from liger_kernel.transformers.rope import (  # noqa: E402
    liger_rotary_pos_emb,
    liger_rotary_pos_emb_gpt_oss,
)

SEED = 0
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
ALIAS_B, ALIAS_S, ALIAS_HQ, ALIAS_HK, ALIAS_D = 2, 128, 8, 2, 64
NUM_B, NUM_S, NUM_HQ, NUM_HK, NUM_D = 1, 512, 16, 4, 128
NUM_PARTIAL_RD = 64
MEM_SHAPES = [
    ("Llama-3.2-1B", 1, 32, 8, 64),   # B omitted; T varies
    ("Qwen3-0.6B", 1, 16, 8, 128),
]
MEM_SEQS = [512, 2048, 8192]
BW_B, BW_HQ, BW_HK, BW_D = 1, 16, 4, 128
BW_SEQS = [256, 1024, 4096, 8192]
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
    if isinstance(x, (list, tuple)):
        return list(x)
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


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def make_cos_sin(B, S, rotary_dim, device, dtype, *, duplicated=True, batch_cos=False):
    """Build Llama-style (duplicated) or GPT-OSS-style (half-width) cos/sin."""
    half = rotary_dim // 2
    # Use deterministic angles so fp64 / dtype floors are stable across runs.
    pos = torch.arange(S, device=device, dtype=torch.float64)
    inv = 1.0 / (10000 ** (torch.arange(0, half, device=device, dtype=torch.float64) / half))
    freqs = pos[:, None] * inv[None, :]  # (S, half)
    if duplicated:
        emb = torch.cat([freqs, freqs], dim=-1)  # (S, rotary_dim)
    else:
        emb = freqs  # (S, rotary_dim/2)
    cos = emb.cos()
    sin = emb.sin()
    if batch_cos:
        cos = cos.unsqueeze(0).expand(B, -1, -1).contiguous()
        sin = sin.unsqueeze(0).expand(B, -1, -1).contiguous()
    else:
        cos = cos.unsqueeze(0)  # (1, S, W)
        sin = sin.unsqueeze(0)
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def hf_rope(q, k, cos, sin, unsqueeze_dim=1):
    return hf_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)


def hf_partial_rope(q, k, cos, sin):
    """HF-style partial RoPE: rotate leading cos.width, leave the tail alone."""
    rd = cos.shape[-1]
    q_rot, q_pass = q[..., :rd], q[..., rd:]
    k_rot, k_pass = k[..., :rd], k[..., rd:]
    cos_u = cos.unsqueeze(1)
    sin_u = sin.unsqueeze(1)
    q_out = torch.cat(((q_rot * cos_u) + (rotate_half(q_rot) * sin_u), q_pass), dim=-1)
    k_out = torch.cat(((k_rot * cos_u) + (rotate_half(k_rot) * sin_u), k_pass), dim=-1)
    return q_out, k_out


def hf_gpt_oss_style(q, k, cos_half, sin_half):
    """GPT-OSS: cos/sin width = head_dim/2, rotate the whole head via broadcast."""
    # Match transformers gpt_oss: unsqueeze then broadcast over the split halves.
    cos = torch.cat([cos_half, cos_half], dim=-1)
    sin = torch.cat([sin_half, sin_half], dim=-1)
    return hf_rope(q, k, cos, sin)


def fp64_rope(q, k, cos, sin, *, mode="full", rotary_dim=None):
    q64 = q.detach().float().double()
    k64 = k.detach().float().double()
    c64 = cos.detach().float().double()
    s64 = sin.detach().float().double()
    if mode == "full":
        return hf_rope(q64, k64, c64, s64)
    if mode == "partial":
        return hf_partial_rope(q64, k64, c64, s64)
    if mode == "gpt_oss":
        return hf_gpt_oss_style(q64, k64, c64, s64)
    raise ValueError(mode)


def rand_qk(B, Hq, Hk, S, D, device, dtype, requires_grad=False):
    q = torch.randn(B, Hq, S, D, device=device, dtype=dtype)
    k = torch.randn(B, Hk, S, D, device=device, dtype=dtype)
    if requires_grad:
        q = q.requires_grad_(True)
        k = k.requires_grad_(True)
    return q, k


# ---------------------------------------------------------------- sections

def section_api_footguns(device):
    """position_ids unused; GPT-OSS half-width without rotary_dim; inspect signature."""
    rows = []
    B, S, Hq, Hk, D = ALIAS_B, ALIAS_S, ALIAS_HQ, ALIAS_HK, ALIAS_D
    dtype = torch.bfloat16

    # 1) position_ids is in the signature but never consumed by the kernel.
    import inspect
    sig = inspect.signature(LigerRopeFunction.forward)
    has_pos = "position_ids" in sig.parameters
    q, k = rand_qk(B, Hq, Hk, S, D, device, dtype)
    cos, sin = make_cos_sin(B, S, D, device, dtype, duplicated=True)
    # Build cos/sin for sequential positions; pass bogus position_ids to Liger.
    q_ref, k_ref = q.clone(), k.clone()
    q_bad, k_bad = q.clone(), k.clone()
    # Cos already indexed by arange(S). Fake position_ids = reversed.
    pos_rev = torch.arange(S - 1, -1, -1, device=device).unsqueeze(0).expand(B, -1)
    out_ref = liger_rotary_pos_emb(q_ref, k_ref, cos, sin, position_ids=None)
    out_bad = liger_rotary_pos_emb(q_bad, k_bad, cos, sin, position_ids=pos_rev)
    # HF with reversed indexing of the same table would differ; Liger ignores ids.
    cos_rev = cos[:, pos_rev[0]]  # (1, S, D) gathered by reversed positions
    sin_rev = sin[:, pos_rev[0]]
    q_hf, k_hf = q.clone(), k.clone()
    hf_rev = hf_rope(q_hf, k_hf, cos_rev, sin_rev)
    rows.append({
        "case": "position_ids_ignored",
        "has_position_ids_param": has_pos,
        "liger_with_rev_ids_vs_none_q": rel_l2(out_bad[0], out_ref[0]),
        "liger_with_rev_ids_vs_none_k": rel_l2(out_bad[1], out_ref[1]),
        "hf_rev_indexed_vs_liger_q": rel_l2(hf_rev[0], out_ref[0]),
        "hf_rev_indexed_vs_liger_k": rel_l2(hf_rev[1], out_ref[1]),
        "same_as_ignoring_ids": (
            rel_l2(out_bad[0], out_ref[0]) == 0.0 and rel_l2(out_bad[1], out_ref[1]) == 0.0
        ),
    })

    # 2) GPT-OSS half-width cos without rotary_dim → partial RoPE on first D/2.
    cos_h, sin_h = make_cos_sin(B, S, D, device, dtype, duplicated=False)
    q0, k0 = rand_qk(B, Hq, Hk, S, D, device, dtype)
    q_mis, k_mis = q0.clone(), k0.clone()
    q_ok, k_ok = q0.clone(), k0.clone()
    q_part_ref, k_part_ref = q0.clone(), k0.clone()
    out_mis = liger_rotary_pos_emb(q_mis, k_mis, cos_h, sin_h)  # no rotary_dim
    out_ok = liger_rotary_pos_emb_gpt_oss(q_ok, k_ok, cos_h, sin_h)
    # Explicit partial over first D/2 with duplicated half? Kernel uses width of cos
    # as rotary_dim when rotary_dim is None → rd = D/2, and loads cos[0:rd/2]=cos[0:D/4].
    # So mis-inferred is NOT the same as "partial with duplicated cos of width D/2".
    # Compare against: correct GPT-OSS and against true partial with duplicated cos.
    cos_dup_half, sin_dup_half = make_cos_sin(B, S, D // 2, device, dtype, duplicated=True)
    out_partial = liger_rotary_pos_emb(q_part_ref, k_part_ref, cos_dup_half, sin_dup_half)
    # Tail of mis-inferred: positions [D/2:] should be unchanged (partial behaviour).
    tail_unchanged = torch.equal(out_mis[0][..., D // 2 :], q0[..., D // 2 :])
    rows.append({
        "case": "gpt_oss_half_width_without_rotary_dim",
        "cos_width": int(cos_h.shape[-1]),
        "head_dim": D,
        "inferred_vs_gpt_oss_q": rel_l2(out_mis[0], out_ok[0]),
        "inferred_vs_gpt_oss_k": rel_l2(out_mis[1], out_ok[1]),
        "inferred_vs_true_partial_dup_q": rel_l2(out_mis[0], out_partial[0]),
        "tail_D_over_2_unchanged": bool(tail_unchanged),
        "gpt_oss_rotates_whole_head": bool(
            not torch.equal(out_ok[0][..., D // 2 :], q0[..., D // 2 :])
        ),
    })
    return rows


def section_inplace_alias(device):
    """When does the in-place kernel mutate the caller's Q/K storage?"""
    rows = []
    B, S, Hq, Hk, D = ALIAS_B, ALIAS_S, ALIAS_HQ, ALIAS_HK, ALIAS_D
    dtype = torch.bfloat16
    cos, sin = make_cos_sin(B, S, D, device, dtype, duplicated=True)

    # Case A: standard contiguous BHSD — transpose makes non-contig → copy.
    q = torch.randn(B, Hq, S, D, device=device, dtype=dtype)
    k = torch.randn(B, Hk, S, D, device=device, dtype=dtype)
    q_before, k_before = q.clone(), k.clone()
    q_ptr, k_ptr = q.data_ptr(), k.data_ptr()
    oq, ok = liger_rotary_pos_emb(q, k, cos, sin)
    rows.append({
        "case": "contiguous_BHSD",
        "input_q_mutated": not torch.equal(q, q_before),
        "input_k_mutated": not torch.equal(k, k_before),
        "out_q_same_ptr_as_in": oq.data_ptr() == q_ptr,
        "out_k_same_ptr_as_in": ok.data_ptr() == k_ptr,
        "out_q_is_input_q": oq.data_ptr() == q.data_ptr(),
    })

    # Case B: BHSD view of contiguous BSHD — transpose+contiguous is no-op → mutate.
    q_bshd = torch.randn(B, S, Hq, D, device=device, dtype=dtype)
    k_bshd = torch.randn(B, S, Hk, D, device=device, dtype=dtype)
    q_view = q_bshd.transpose(1, 2)  # (B, Hq, S, D)
    k_view = k_bshd.transpose(1, 2)
    q_bshd_before, k_bshd_before = q_bshd.clone(), k_bshd.clone()
    oq2, ok2 = liger_rotary_pos_emb(q_view, k_view, cos, sin)
    rows.append({
        "case": "transpose_view_of_BSHD",
        "storage_q_mutated": not torch.equal(q_bshd, q_bshd_before),
        "storage_k_mutated": not torch.equal(k_bshd, k_bshd_before),
        "out_q_same_storage_as_bshd": oq2.transpose(1, 2).data_ptr() == q_bshd.data_ptr()
        or oq2.data_ptr() == q_view.data_ptr(),
        "view_and_out_share_ptr": oq2.data_ptr() == q_view.data_ptr(),
    })

    # Case C: rope_forward returns transposed contiguous tiles; check clone path.
    q3 = torch.randn(B, Hq, S, D, device=device, dtype=dtype)
    k3 = torch.randn(B, Hk, S, D, device=device, dtype=dtype)
    q3_before = q3.clone()
    oq3, ok3, _, _ = rope_forward(q3, k3, cos, sin)
    rows.append({
        "case": "rope_forward_contiguous_BHSD",
        "input_mutated": not torch.equal(q3, q3_before),
        "out_contiguous_BSHD": bool(oq3.transpose(1, 2).is_contiguous()),
    })

    # Case D: retain_graph + reuse dQ/dK (backward also in-place).
    q4, k4 = rand_qk(B, Hq, Hk, S, D, device, dtype, requires_grad=True)
    cos4, sin4 = make_cos_sin(B, S, D, device, dtype, duplicated=True)
    oq4, ok4 = LigerRopeFunction.apply(q4, k4, cos4, sin4)
    dq = torch.randn_like(oq4)
    dk = torch.randn_like(ok4)
    rows.append({
        "case": "grad_like_stride",
        "out_q_contiguous": bool(oq4.is_contiguous()),
        "dQ_contiguous": bool(dq.is_contiguous()),
        "dQ_stride": list(dq.stride()),
        "out_q_stride": list(oq4.stride()),
        "same_stride_as_out": list(dq.stride()) == list(oq4.stride()),
    })
    # Clean calls with clones (backward is in-place on non-contiguous dQ/dK).
    g1 = torch.autograd.grad((oq4, ok4), (q4, k4), (dq.clone(), dk.clone()), retain_graph=True)
    g2 = torch.autograd.grad((oq4, ok4), (q4, k4), (dq.clone(), dk.clone()), retain_graph=True)
    rows.append({
        "case": "retain_graph_fresh_grads",
        "second_vs_first_q": rel_l2(g2[0], g1[0]),
        "second_vs_first_k": rel_l2(g2[1], g1[1]),
    })
    # Reuse same grad buffers: first call is still correct but overwrites the
    # caller's dQ/dK (and the returned grad tensors alias those buffers). A
    # second call then double-applies the inverse rotation.
    dq_r, dk_r = dq.clone(), dk.clone()
    dQ_before, dK_before = dq_r.clone(), dk_r.clone()
    g_reuse1 = torch.autograd.grad((oq4, ok4), (q4, k4), (dq_r, dk_r), retain_graph=True)
    reuse_first_q = rel_l2(g_reuse1[0], g1[0])
    reuse_first_k = rel_l2(g_reuse1[1], g1[1])
    g_reuse1_q = g_reuse1[0].clone()
    g_reuse1_k = g_reuse1[1].clone()
    aliased = g_reuse1[0].data_ptr() == dq_r.data_ptr()
    g_reuse2 = torch.autograd.grad((oq4, ok4), (q4, k4), (dq_r, dk_r), retain_graph=True)
    rows.append({
        "case": "retain_graph_reuse_dQ_dK",
        "reuse_first_vs_clean_q": reuse_first_q,
        "reuse_first_vs_clean_k": reuse_first_k,
        "reuse_second_vs_first_q": rel_l2(g_reuse2[0], g_reuse1_q),
        "reuse_second_vs_first_k": rel_l2(g_reuse2[1], g_reuse1_k),
        "caller_dQ_overwritten": not torch.equal(dq_r, dQ_before),
        "caller_dK_overwritten": not torch.equal(dk_r, dK_before),
        "returned_grad_aliases_dQ": bool(aliased),
    })
    return rows


def section_partial_and_gqa(device):
    """Partial rotary_dim tail untouched; GQA head counts."""
    rows = []
    B, S, D = 2, 64, 128
    dtype = torch.bfloat16

    for Hq, Hk, rd, label in [
        (16, 16, 128, "mha_full"),
        (16, 4, 128, "gqa_full"),
        (16, 4, 64, "gqa_partial_64"),
        (8, 2, 96, "gqa_partial_96"),
    ]:
        q0, k0 = rand_qk(B, Hq, Hk, S, D, device, dtype)
        cos, sin = make_cos_sin(B, S, rd, device, dtype, duplicated=True)
        q, k = q0.clone(), k0.clone()
        oq, ok = liger_rotary_pos_emb(q, k, cos, sin)
        tail_q = torch.equal(oq[..., rd:], q0[..., rd:]) if rd < D else True
        tail_k = torch.equal(ok[..., rd:], k0[..., rd:]) if rd < D else True
        # vs HF partial / full
        if rd == D:
            ref_q, ref_k = hf_rope(q0.clone(), k0.clone(), cos, sin)
        else:
            ref_q, ref_k = hf_partial_rope(q0.clone(), k0.clone(), cos, sin)
        rows.append({
            "label": label,
            "Hq": Hq, "Hk": Hk, "D": D, "rotary_dim": rd,
            "tail_q_unchanged": bool(tail_q),
            "tail_k_unchanged": bool(tail_k),
            "q_rel_vs_hf": rel_l2(oq, ref_q),
            "k_rel_vs_hf": rel_l2(ok, ref_k),
        })
    return rows


def section_numerics(device):
    rows = []
    B, S, Hq, Hk, D = NUM_B, NUM_S, NUM_HQ, NUM_HK, NUM_D
    configs = [
        ("full", D, True, None),
        ("partial", NUM_PARTIAL_RD, True, None),
        ("gpt_oss", D, False, D),  # half-width cos, explicit rotary_dim=D
    ]
    for dtype_name, dtype in DTYPES.items():
        for mode, rd, duplicated, rotary_dim_arg in configs:
            torch.manual_seed(SEED)
            q0, k0 = rand_qk(B, Hq, Hk, S, D, device, dtype)
            cos, sin = make_cos_sin(B, S, rd if duplicated else D, device, dtype,
                                    duplicated=duplicated)
            # fp64 reference
            ref_q, ref_k = fp64_rope(q0, k0, cos, sin, mode=mode)
            floor_q = ref_q.to(dtype).float().double()
            floor_k = ref_k.to(dtype).float().double()
            floor_q_err = rel_l2(floor_q, ref_q)
            floor_k_err = rel_l2(floor_k, ref_k)

            # Liger
            ql, kl = q0.clone().requires_grad_(True), k0.clone().requires_grad_(True)
            if mode == "gpt_oss":
                oql, okl = liger_rotary_pos_emb(ql, kl, cos, sin, rotary_dim=rotary_dim_arg)
            else:
                oql, okl = liger_rotary_pos_emb(ql, kl, cos, sin)
            # randn_like(Liger out) inherits the non-contiguous BHSD-of-BSHD stride,
            # so rope_backward's transpose+contiguous is a no-op and mutates dQ/dK
            # in place — always clone before each backward.
            dq = torch.randn_like(oql)
            dk = torch.randn_like(okl)
            gql, gkl = torch.autograd.grad(
                (oql, okl), (ql, kl), (dq.clone(), dk.clone()), retain_graph=False,
            )

            # HF reference in dtype
            qh, kh = q0.clone().requires_grad_(True), k0.clone().requires_grad_(True)
            if mode == "full":
                oqh, okh = hf_rope(qh, kh, cos, sin)
            elif mode == "partial":
                oqh, okh = hf_partial_rope(qh, kh, cos, sin)
            else:
                oqh, okh = hf_gpt_oss_style(qh, kh, cos, sin)
            gqh, gkh = torch.autograd.grad((oqh, okh), (qh, kh), (dq.clone(), dk.clone()))

            # fp64 grads
            q64 = q0.detach().float().double().requires_grad_(True)
            k64 = k0.detach().float().double().requires_grad_(True)
            c64, s64 = cos.float().double(), sin.float().double()
            if mode == "full":
                o64q, o64k = hf_rope(q64, k64, c64, s64)
            elif mode == "partial":
                o64q, o64k = hf_partial_rope(q64, k64, c64, s64)
            else:
                o64q, o64k = hf_gpt_oss_style(q64, k64, c64, s64)
            g64q, g64k = torch.autograd.grad(
                (o64q, o64k), (q64, k64),
                (dq.float().double(), dk.float().double()),
            )

            # torch.compile of HF body
            compile_err = None
            compile_q = compile_k = compile_dq = compile_dk = None
            try:
                if mode == "full":
                    compiled = torch.compile(hf_rope, fullgraph=False)
                    qc, kc = q0.clone().requires_grad_(True), k0.clone().requires_grad_(True)
                    ocq, ock = compiled(qc, kc, cos, sin)
                elif mode == "partial":
                    compiled = torch.compile(hf_partial_rope, fullgraph=False)
                    qc, kc = q0.clone().requires_grad_(True), k0.clone().requires_grad_(True)
                    ocq, ock = compiled(qc, kc, cos, sin)
                else:
                    compiled = torch.compile(hf_gpt_oss_style, fullgraph=False)
                    qc, kc = q0.clone().requires_grad_(True), k0.clone().requires_grad_(True)
                    ocq, ock = compiled(qc, kc, cos, sin)
                gcq, gck = torch.autograd.grad((ocq, ock), (qc, kc), (dq.clone(), dk.clone()))
                compile_q = rel_l2(ocq, ref_q)
                compile_k = rel_l2(ock, ref_k)
                compile_dq = rel_l2(gcq, g64q)
                compile_dk = rel_l2(gck, g64k)
            except Exception as e:
                compile_err = f"{type(e).__name__}: {e}"

            row = {
                "dtype": dtype_name,
                "mode": mode,
                "rotary_dim": rd if mode != "gpt_oss" else D,
                "cos_width": int(cos.shape[-1]),
                "floor": {"q": floor_q_err, "k": floor_k_err},
                "liger": {
                    "q": rel_l2(oql, ref_q), "k": rel_l2(okl, ref_k),
                    "dq": rel_l2(gql, g64q), "dk": rel_l2(gkl, g64k),
                },
                "eager_hf": {
                    "q": rel_l2(oqh, ref_q), "k": rel_l2(okh, ref_k),
                    "dq": rel_l2(gqh, g64q), "dk": rel_l2(gkh, g64k),
                },
                "liger_vs_hf": {
                    "q": rel_l2(oql, oqh), "k": rel_l2(okl, okh),
                    "dq": rel_l2(gql, gqh), "dk": rel_l2(gkl, gkh),
                },
                "torch_compile": (
                    {"q": compile_q, "k": compile_k, "dq": compile_dq, "dk": compile_dk}
                    if compile_err is None else {"error": compile_err}
                ),
            }
            rows.append(row)
            print("num", dtype_name, mode, {k: fmt(v) for k, v in row["liger"].items()}, flush=True)
    return rows


def _peak_delta(fn, inputs):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    held = torch.cuda.memory_allocated()
    return out, (held - base) / 2**20, (peak - base) / 2**20


def section_memory(device):
    rows = []
    dtype = torch.bfloat16
    for shape_name, B, Hq, Hk, D in MEM_SHAPES:
        for S in MEM_SEQS:
            torch.manual_seed(SEED)
            cos, sin = make_cos_sin(B, S, D, device, dtype, duplicated=True)
            q0, k0 = rand_qk(B, Hq, Hk, S, D, device, dtype)
            nbytes = (B * Hq * S * D + B * Hk * S * D) * 2  # bf16 Q+K

            def measure(name, fwd_fn):
                # Held after forward
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                base = torch.cuda.memory_allocated()
                q = q0.clone().requires_grad_(True)
                k = k0.clone().requires_grad_(True)
                try:
                    oq, ok = fwd_fn(q, k, cos, sin)
                    torch.cuda.synchronize()
                    held = (torch.cuda.memory_allocated() - base) / 2**20
                    dq, dk = torch.randn_like(oq), torch.randn_like(ok)
                    torch.cuda.reset_peak_memory_stats()
                    before = torch.cuda.memory_allocated()
                    torch.autograd.grad((oq, ok), (q, k), (dq, dk))
                    torch.cuda.synchronize()
                    extra = (torch.cuda.max_memory_allocated() - before) / 2**20
                    # latency
                    def once():
                        qq = q0.clone().requires_grad_(True)
                        kk = k0.clone().requires_grad_(True)
                        ooq, ook = fwd_fn(qq, kk, cos, sin)
                        torch.autograd.grad((ooq, ook), (qq, kk), (dq, dk))
                    ms = do_bench(once, return_mode="median")
                    return {
                        "held_after_fwd_MiB": round(held, 1),
                        "extra_peak_MiB": round(extra, 1),
                        "fwd_bwd_ms": round(ms, 4),
                    }
                except Exception as e:
                    return {"error": f"{type(e).__name__}: {e}"}

            cell = {
                "shape": shape_name, "B": B, "S": S, "Hq": Hq, "Hk": Hk, "D": D,
                "QK_MiB": round(nbytes / 2**20, 1),
            }
            cell["liger"] = measure("liger", liger_rotary_pos_emb)
            cell["eager_hf"] = measure("eager", hf_rope)
            try:
                compiled = torch.compile(hf_rope, fullgraph=False)
                # warmup
                _ = compiled(q0.clone(), k0.clone(), cos, sin)
                torch.cuda.synchronize()
                cell["torch_compile"] = measure("compile", compiled)
            except Exception as e:
                cell["torch_compile"] = {"error": f"{type(e).__name__}: {e}"}
            rows.append(cell)
            print("mem", shape_name, S, cell["liger"], cell["eager_hf"], flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    return rows


def section_bandwidth(device):
    rows, kernels = [], []
    dtype = torch.bfloat16
    B, Hq, Hk, D = BW_B, BW_HQ, BW_HK, BW_D
    for S in BW_SEQS:
        cos, sin = make_cos_sin(B, S, D, device, dtype, duplicated=True)
        q = torch.randn(B, Hq, S, D, device=device, dtype=dtype)
        k = torch.randn(B, Hk, S, D, device=device, dtype=dtype)
        nbytes = (B * Hq * S * D + B * Hk * S * D) * 2  # read+write roughly 2x for in-place

        # Forward bandwidth: in-place reads Q/K/cos/sin and writes Q/K.
        # Count 2*(Q+K) as the payload (cos/sin are small: 2*S*D).
        payload = 2 * nbytes  # read + write Q/K
        cos_bytes = 2 * B * S * D * 4  # fp32-ish; actually bf16 here: *2
        cos_bytes = 2 * cos.shape[0] * S * cos.shape[-1] * cos.element_size() * 2  # cos+sin read

        qg = q.clone().requires_grad_(True)
        kg = k.clone().requires_grad_(True)
        oqg, okg = liger_rotary_pos_emb(qg, kg, cos, sin)
        dq, dkv = torch.randn_like(oqg), torch.randn_like(okg)

        qh = q.clone().requires_grad_(True)
        kh = k.clone().requires_grad_(True)
        oqh, okh = hf_rope(qh, kh, cos, sin)

        compile_ok, compile_err = True, None
        try:
            compiled = torch.compile(hf_rope, fullgraph=False)
            qc = q.clone().requires_grad_(True)
            kc = k.clone().requires_grad_(True)
            ocq, ock = compiled(qc, kc, cos, sin)
            torch.cuda.synchronize()
        except Exception as e:
            compile_ok, compile_err = False, f"{type(e).__name__}: {e}"

        def copy_qk():
            return q.clone(), k.clone()

        fns = {
            "copy": (copy_qk, 1.0),  # 1× nbytes write (clone reads+writes ≈ 2; report vs copy)
            "fwd_liger": (lambda: liger_rotary_pos_emb(q.clone(), k.clone(), cos, sin), 2.0),
            "fwd_eager": (lambda: hf_rope(q.clone(), k.clone(), cos, sin), 2.0),
            "bwd_liger": (
                lambda: torch.autograd.grad((oqg, okg), (qg, kg), (dq.clone(), dkv.clone()),
                                            retain_graph=True),
                2.0,
            ),
            "bwd_eager": (
                lambda: torch.autograd.grad((oqh, okh), (qh, kh), (dq.clone(), dkv.clone()),
                                            retain_graph=True),
                2.0,
            ),
        }
        if compile_ok:
            fns["fwd_compile"] = (lambda: compiled(q.clone(), k.clone(), cos, sin), 2.0)
            fns["bwd_compile"] = (
                lambda: torch.autograd.grad((ocq, ock), (qc, kc), (dq.clone(), dkv.clone()),
                                            retain_graph=True),
                2.0,
            )

        do_bench(fns["copy"][0], rep=50)
        times = {name: [] for name in fns}
        for _ in range(BW_ROUNDS):
            for name, (fn, _) in fns.items():
                times[name].append(do_bench(fn, return_mode="median"))

        row = {
            "B": B, "S": S, "Hq": Hq, "Hk": Hk, "D": D,
            "QK_MiB": round(nbytes / 2**20, 2),
            "compile_ok": compile_ok,
        }
        if not compile_ok:
            row["compile_error"] = compile_err
        for name, (_, mult) in fns.items():
            ms = sorted(times[name])[len(times[name]) // 2]
            row[name + "_ms"] = round(ms, 4)
            # GB/s relative to Q+K bytes * mult
            row[name + "_gbps"] = round(mult * nbytes * 1e-9 / (ms * 1e-3), 1)
        for name in ("fwd_liger", "fwd_eager"):
            row[name + "_vs_copy"] = round(row["copy_ms"] / max(row[name + "_ms"], 1e-9), 2)
        row["fwd_liger_vs_eager"] = round(row["fwd_eager_ms"] / max(row["fwd_liger_ms"], 1e-9), 2)
        row["bwd_liger_vs_eager"] = round(row["bwd_eager_ms"] / max(row["bwd_liger_ms"], 1e-9), 2)
        if compile_ok:
            row["fwd_compile_vs_eager"] = round(
                row["fwd_eager_ms"] / max(row["fwd_compile_ms"], 1e-9), 2
            )
        rows.append(row)
        print("bw", row, flush=True)

        # Kernel metadata via warmup
        try:
            qt = q.transpose(1, 2).contiguous()
            kt = k.transpose(1, 2).contiguous()
            cos_c, sin_c = cos.contiguous(), sin.contiguous()
            pad_hd = triton.next_power_of_2(D)
            pad_rd = triton.next_power_of_2(D)
            pad_nq = triton.next_power_of_2(Hq)
            pad_nk = triton.next_power_of_2(Hk)
            block = max(pad_nq, pad_nk)
            kern = _triton_rope.warmup(
                qt, qt.stride(1), kt, kt.stride(1),
                cos_c, cos_c.stride(-2), sin_c, sin_c.stride(-2),
                S, B, cos_c.shape[0], Hq, Hk, D,
                pad_nq, pad_nk, pad_hd, pad_rd, D,
                BLOCK_SIZE=block, BACKWARD_PASS=False, grid=(1,),
            )
            kern._init_handles()
            kernels.append({
                "S": S, "Hq": Hq, "Hk": Hk, "D": D, "BLOCK_SIZE": block,
                "n_regs": kern.n_regs, "n_spills": kern.n_spills,
                "shared": kern.metadata.shared,
            })
        except Exception as e:
            kernels.append({"S": S, "error": f"{type(e).__name__}: {e}"})

        del q, k, qg, kg, oqg, okg, qh, kh, oqh, okh, dq, dkv
        if compile_ok:
            del qc, kc, ocq, ock
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
        "## 0. API footguns",
        "",
        "| case | detail |",
        "|---|---|",
    ]
    for r in result["api"]:
        if r["case"] == "position_ids_ignored":
            lines.append(
                f"| position_ids_ignored | has_param={r['has_position_ids_param']}; "
                f"liger(rev ids)==none q/k rel={fmt(r['liger_with_rev_ids_vs_none_q'])}/"
                f"{fmt(r['liger_with_rev_ids_vs_none_k'])}; "
                f"HF rev-indexed vs liger q/k={fmt(r['hf_rev_indexed_vs_liger_q'])}/"
                f"{fmt(r['hf_rev_indexed_vs_liger_k'])}; "
                f"same_as_ignoring={r['same_as_ignoring_ids']} |"
            )
        else:
            lines.append(
                f"| gpt_oss_half_width_without_rotary_dim | cos_width={r['cos_width']} "
                f"head_dim={r['head_dim']}; vs gpt_oss q/k={fmt(r['inferred_vs_gpt_oss_q'])}/"
                f"{fmt(r['inferred_vs_gpt_oss_k'])}; vs true partial="
                f"{fmt(r['inferred_vs_true_partial_dup_q'])}; "
                f"tail_unchanged={r['tail_D_over_2_unchanged']}; "
                f"gpt_oss_rotates_whole_head={r['gpt_oss_rotates_whole_head']} |"
            )

    lines += [
        "",
        f"## 1. In-place aliasing (bf16, B={ALIAS_B}, S={ALIAS_S}, "
        f"Hq/Hk={ALIAS_HQ}/{ALIAS_HK}, D={ALIAS_D})",
        "",
        "| case | fields |",
        "|---|---|",
    ]
    for r in result["alias"]:
        fields = ", ".join(f"{k}={fmt(v)}" for k, v in r.items() if k != "case")
        lines.append(f"| {r['case']} | {fields} |")

    lines += [
        "",
        "## 2. Partial RoPE + GQA (bf16)",
        "",
        "| label | Hq | Hk | D | rd | tail_q | tail_k | q vs hf | k vs hf |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in result["partial_gqa"]:
        lines.append(
            f"| {r['label']} | {r['Hq']} | {r['Hk']} | {r['D']} | {r['rotary_dim']} | "
            f"{r['tail_q_unchanged']} | {r['tail_k_unchanged']} | "
            f"{fmt(r['q_rel_vs_hf'])} | {fmt(r['k_rel_vs_hf'])} |"
        )

    lines += [
        "",
        f"## 3. Error vs fp64 (B={NUM_B}, S={NUM_S}, Hq/Hk={NUM_HQ}/{NUM_HK}, D={NUM_D}; "
        "floor is dtype-rounded fp64 for q/k only)",
        "",
        "| dtype | mode | out | floor | liger | eager_hf | compile | liger vs hf |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in result["numerics"]:
        comp = r["torch_compile"]
        for o in ("q", "k", "dq", "dk"):
            floor_v = r["floor"][o] if o in r["floor"] else None
            cr = comp.get(o, comp.get("error", ""))
            cr_s = fmt(cr) if not isinstance(cr, str) else short_err(cr)
            lines.append(
                f"| {r['dtype']} | {r['mode']} | {o} | {fmt(floor_v)} | "
                f"{fmt(r['liger'][o])} | {fmt(r['eager_hf'][o])} | {cr_s} | "
                f"{fmt(r['liger_vs_hf'][o])} |"
            )

    names = ["liger", "eager_hf", "torch_compile"]
    lines += ["", "## 4. Memory + latency (bf16; do_bench median)", ""]
    for metric, title in (
        ("held_after_fwd_MiB", "held after forward (MiB)"),
        ("extra_peak_MiB", "extra peak during bwd (MiB)"),
        ("fwd_bwd_ms", "fwd + bwd (ms)"),
    ):
        lines += [
            f"### {title}",
            "",
            "| shape | S | Hq | Hk | D | " + " | ".join(names) + " |",
            "|---|---|---|---|---|" + "---|" * len(names),
        ]
        for r in result["memory"]:
            cells = [str(r[n].get(metric, r[n].get("error", ""))) for n in names]
            lines.append(
                f"| {r['shape']} | {r['S']} | {r['Hq']} | {r['Hk']} | {r['D']} | "
                + " | ".join(cells) + " |"
            )
        lines.append("")

    lines += [
        "## 5. Bandwidth (bf16 GQA 16/4 × D=128; payload = 2·(Q+K) bytes)",
        "",
        "| S | copy ms | fwd liger ms/GBps | fwd eager | bwd liger | bwd eager | "
        "liger/eager fwd | liger/eager bwd |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in result["bandwidth"]:
        lines.append(
            f"| {r['S']} | {r['copy_ms']} | {r['fwd_liger_ms']}/{r['fwd_liger_gbps']} | "
            f"{r['fwd_eager_ms']}/{r['fwd_eager_gbps']} | "
            f"{r['bwd_liger_ms']}/{r['bwd_liger_gbps']} | "
            f"{r['bwd_eager_ms']}/{r['bwd_eager_gbps']} | "
            f"{r['fwd_liger_vs_eager']} | {r['bwd_liger_vs_eager']} |"
        )

    lines += [
        "",
        "## 6. Kernel resources",
        "",
        "| S | BLOCK | regs | spills | shared |",
        "|---|---|---|---|---|",
    ]
    for r in result["kernels"]:
        if "error" in r:
            lines.append(f"| {r.get('S')} |  |  |  | {short_err(r['error'])} |")
        else:
            lines.append(
                f"| {r['S']} | {r['BLOCK_SIZE']} | {r['n_regs']} | {r['n_spills']} | {r['shared']} |"
            )

    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    t0 = time.perf_counter()
    print("gpu", GPU, flush=True)

    api = section_api_footguns(device)
    print("api", api, flush=True)
    alias = section_inplace_alias(device)
    print("alias", alias, flush=True)
    partial_gqa = section_partial_and_gqa(device)
    print("partial_gqa", partial_gqa, flush=True)
    numerics = section_numerics(device)
    memory = section_memory(device)
    bandwidth, kernels = section_bandwidth(device)

    torch.cuda.synchronize()
    peak = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    result = {
        "upstream_commit": upstream_commit(),
        "gpu": GPU,
        "runtime_s": round(time.perf_counter() - t0, 1),
        "peak_memory_mib": peak,
        "seed": SEED,
        "api": api,
        "alias": alias,
        "partial_gqa": partial_gqa,
        "numerics": numerics,
        "memory": memory,
        "bandwidth": bandwidth,
        "kernels": kernels,
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    write_md(result)
    print("done", result["runtime_s"], "s, peak", peak, "MiB", flush=True)


if __name__ == "__main__":
    main()
