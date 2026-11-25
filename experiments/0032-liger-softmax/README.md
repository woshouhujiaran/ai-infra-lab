# 0032 — Liger-Kernel Softmax (`LigerSoftmaxFunction`)

- **Upstream**: [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) (submodule `upstream/Liger-Kernel`), imported unmodified from `src/`.
- **Component**: `src/liger_kernel/ops/softmax.py`. One Triton program per row; `calculate_settings(n)` sets `BLOCK_SIZE = next_power_of_2(n)` (hard-fails above 65536) and picks warps; single-block kernel when `n_cols <= BLOCK_SIZE`, else multi-block online max/sum. Backward saves `y` and applies `y * (dy - sum(dy*y))`. `@ensure_contiguous` on both forward and backward. Distinct from [0010](../0010-triton-fused-softmax) (triton tutorial fused softmax, forward bandwidth only).
- **Hardware**: RTX 4060 Laptop 8 GB, torch 2.14.1+cu126, triton-windows, memory capped at 60% by `lab/gpu_guard.py`. Peak allocated ~3.3 GB.
- **Run**: `.venv\Scripts\python.exe experiments\0032-liger-softmax\run.py` (~30 s).

## Question

Is Liger's Softmax a usable drop-in for attention / vocab-sized rows on an Ada laptop GPU — and does the multi-block path that was written for wide rows ever run through the public API?

## Setup

- **Numerics**: T = 2048, V = 4096; fp32 / fp16 / bf16; logit scale 1 and 10. Relative L2 of Y / dX vs fp64 textbook softmax and vs `F.softmax` / `torch.compile`.
- **Edges**: causal-style half-row `-inf`, all `-inf`, NaN/`+inf` cells, large logits (×1e3), all-equal zeros, one-hotish peaks.
- **Limits**: public `_softmax_forward` over V ∈ {1…65536}; over-limit 65537 / 128256; force multi-block with `BLOCK_SIZE = 1024` vs public single-block.
- **Layouts**: row-strided / transposed / 3D; `LigerSoftmax` Module vs Function.
- **Memory / latency**: bf16 attention shapes (S = 512–16384) and vocab-ish rows (32k / 64k).
- **Bandwidth**: bf16, T = 4096, V from 512 to 65536 (incl. V = 2049 past a power of two). Forward counted as 2·T·V·2 bytes.

## Findings

1. **Numerics sit on the dtype floor and match `F.softmax`.**
   - bf16 Y ≈ 1.66e-3 vs fp64 (equal to the rounding floor); liger vs F is ~1.5e-5 on Y. fp16 ≈ 2.1e-4; fp32 ≈ 9e-8. `torch.compile` matches. Causal `-inf` masks keep masked dX at 0; all `-inf` / NaN rows NaN like torch.

2. **The multi-block path is dead behind `calculate_settings` — and does not compile.**
   - For every legal V, `BLOCK_SIZE = next_power_of_2(V) ≥ V`, so `multi_block_launch` is always `False`. V ≥ 65537 raises before any kernel launch (128k / 152k vocab rows and attention S > 64k are unsupported). Forcing the multi-block kernels with a capped `BLOCK_SIZE = 1024` fails to compile on this triton: `m = tl.float32(-float("inf"))` → `TypeError: 'dtype' object is not callable`. The online wide-row kernels are unreachable *and* broken on current triton-windows.

3. **`@ensure_contiguous` makes strided / 3D inputs safe.**
   - Row-strided x leaves the neighbouring canary untouched; transposed and 3D match contiguous `F.softmax` within ~2e-5 / ~3e-3 on Y / dX. Module ≡ Function bit-exactly.

4. **Forward is competitive past ~1k columns; small rows pay launch cost.**
   - After warm-up, Liger forward hits 164–234 GB/s for V ≥ 1024 (1.0–2.2× `F.softmax` at mid widths; tied at 16k–64k ≈ 230 GB/s copy bandwidth). V = 2049 (BLOCK 4096, lane util 0.5) still 1.9× F. At attention S = 512 the first measured call is launch-heavy (0.17 vs 0.011 ms); once warm, bandwidth at V = 512 is only ~63 GB/s for both. End-to-end fwd+bwd at 32k vocab: Liger 0.71 vs F 1.16 ms; at 64k Liger and F are within ~20%.

Takeaway: Liger Softmax is floor-accurate and fine for attention / mid-width rows that fit in one Triton block, but the multi-block implementation is dead code (gated out by `calculate_settings`) and currently uncompilable, so anything wider than 65536 columns hard-fails. Prefer `F.softmax` / `torch.compile` unless you already standardize on Liger kernels; do not rely on this op for 128k-vocab LM heads or >64k context softmax without a working multi-block rewrite.

Full numbers: [results.md](results.md) / [results.json](results.json).
