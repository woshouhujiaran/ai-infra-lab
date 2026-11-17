# 0024 — Liger-Kernel RoPE (`LigerRopeFunction` / `rope_forward`)

- **Upstream**: [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) (submodule `upstream/Liger-Kernel`, commit `1ab779f`), imported unmodified from `src/`.
- **Component**: `src/liger_kernel/ops/rope.py`. Triton in-place Llama/Mistral split-half RoPE on Q and K (after transpose to `(B, S, H, D)` + `contiguous`). Supports full and partial `rotary_dim`, duplicated cos/sin or GPT-OSS half-width cos/sin (via explicit `rotary_dim`), and GQA (`n_q ≠ n_kv`).
- **Hardware**: RTX 4060 Laptop 8 GB, torch 2.14.1+cu126, triton-windows 3.8.0, memory capped at 60% by `lab/gpu_guard.py`. Peak allocated memory was ~0.75 GB.
- **Run**: `.venv\Scripts\python.exe experiments\0024-liger-rope\run.py` (about 30–60 s).

## Question

Does the in-place Triton RoPE stay safe under common layout / autograd patterns (transpose views, `randn_like` grads, `retain_graph`)? Is `position_ids` honored? What happens if GPT-OSS half-width cos/sin are passed without `rotary_dim`? How do numerics and memory/bandwidth compare to HF `apply_rotary_pos_emb` and `torch.compile`?

## Setup

- **API footguns**: pass reversed `position_ids` vs `None`; pass GPT-OSS half-width cos without `rotary_dim` vs `liger_rotary_pos_emb_gpt_oss`.
- **In-place aliasing** (bf16, B=2, S=128, GQA 8/2, D=64): contiguous BHSD inputs; BHSD view of contiguous BSHD storage; `randn_like` grad strides; `retain_graph` with fresh vs reused `dQ`/`dK`.
- **Partial + GQA**: MHA/GQA full RoPE and partial `rotary_dim` ∈ {64, 96} on D=128; check non-rotary tail unchanged vs HF.
- **Numerics**: S=512, GQA 16/4, D=128; full / partial / GPT-OSS; fp32 / fp16 / bf16; rel L2 vs fp64 and vs HF / `torch.compile`.
- **Memory / latency**: bf16 at Llama-3.2-1B (32/8 × 64) and Qwen3-0.6B (16/8 × 128), S ∈ {512, 2048, 8192}.
- **Bandwidth**: bf16 GQA 16/4 × D=128, S ∈ {256, 1024, 4096, 8192}; payload counted as 2·(Q+K).

## Findings

1. **`position_ids` is accepted and silently ignored.**
   - `LigerRopeFunction.forward` still takes `position_ids`, but the kernel indexes cos/sin by `pid % seq_len`.
   - Reversed ids match `position_ids=None` bit-exactly; HF-style gather of the same table with reversed positions disagrees at rel ≈ 1.0.
   - Safe only when cos/sin are already gathered for the true positions (the modern HF Llama path).

2. **GPT-OSS half-width cos without `rotary_dim` is silently treated as partial RoPE over the first `D/2`.**
   - Inferred path leaves the upper half of each head untouched and disagrees with `liger_rotary_pos_emb_gpt_oss` at rel ≈ 1.1.
   - Must pass `rotary_dim=head_dim` (or use the GPT-OSS wrapper).

3. **In-place writes mutate caller storage when transpose+contiguous is a no-op; backward does the same to `dQ`/`dK`.**
   - Contiguous BHSD inputs are copied first — caller Q/K unchanged.
   - A BHSD view of contiguous BSHD storage is mutated in place (common after `hidden.view(...).transpose(1, 2)`).
   - Liger's output is a non-contiguous BHSD view of contiguous BSHD; `randn_like` inherits that stride, so backward's transpose+contiguous is a no-op, overwrites the caller's `dQ`/`dK`, and returns aliases of those buffers. First `retain_graph` reuse is still correct; a second reuse double-applies the inverse rotation (rel ≈ 1.0).

4. **Partial RoPE and GQA match HF within bf16 noise; the non-rotary tail is untouched.**
   - Full and partial (rd 64 / 96) GQA cells: q/k vs HF ~1.3–1.8e-3; tails bit-identical to the inputs.

5. **Numerics sit near the dtype floor and track HF; `torch.compile` of HF is often on the floor.**
   - fp32: liger q/dq ~3e-8 / 2.8e-8 (HF ~4e-8).
   - bf16 full: liger q ~0.002 (floor 0.0017, HF 0.0023); liger vs HF ~0.002.
   - GPT-OSS with explicit `rotary_dim` matches the full-RoPE cells.

6. **In-place RoPE needs no extra backward peak; 2.0–3.7× faster than eager HF, with little latency win over `torch.compile` at long S.**
   - Held after forward equals 2·(Q+K) for liger / eager / compile.
   - Extra peak during backward: liger **0** MiB (in-place on the grad tiles) vs eager 6.5–112 MiB vs compile 2.5–48 MiB at the shapes above.
   - Fwd+bwd at Llama-3.2-1B × 8k: liger 1.43 ms vs eager 5.55 ms (3.9×) vs compile 1.28 ms.
   - Bandwidth: fwd liger 2.0–3.7× eager (65–110 GB/s); bwd liger 4.7–7.1× eager (102–177 GB/s). No register spills (32 regs, 0 shared).

Takeaway: use Liger RoPE when cos/sin are pre-indexed and you want zero extra backward peak; never rely on `position_ids`, always pass `rotary_dim` for half-width cos, and do not reuse `dQ`/`dK` buffers across `retain_graph` backward calls when they came from `randn_like(liger_out)` (or any non-contiguous BHSD-of-BSHD layout).

Full numbers: [results.md](results.md) / [results.json](results.json).
