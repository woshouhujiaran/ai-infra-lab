# 0027 — Triton tutorial seeded dropout vs torch and a stored mask

- **Upstream**: [triton-lang/triton](https://github.com/triton-lang/triton) (shallow submodule `upstream/triton`, tag `v3.8.0`, matching the installed `triton-windows` 3.8.0). The tutorial file is imported unmodified; `run.py` executes only its kernels and wrappers and skips the demo code.
- **Component**: `python/tutorials/04-low-memory-dropout.py`: `_seeded_dropout` / `seeded_dropout` (`random = tl.rand(seed, offsets)`, `keep = random > p`, output `x / (1 - p)`), and the mask-based `_dropout` / `dropout` it is compared against.
- **Hardware**: RTX 4060 Laptop 8 GB (memory capped at 60% by `lab/gpu_guard.py`), torch 2.14.1+cu126, triton-windows 3.8.0.
- **Run**: `.venv\Scripts\python.exe experiments\0027-triton-tutorial-dropout\run.py` (about 80 s).

## Question

The tutorial claims a one-int32-seed dropout has a smaller memory footprint, less data movement and simpler randomness management than a stored mask. Is the mask statistically sound, how does it behave at the edges (p, dtype, NaN, huge tensors, stacked layers, checkpointing), and does it keep up with `torch.nn.functional.dropout` on a laptop GPU?

## Setup

- Statistics use fp32 ones so that `out != 0` is the keep mask. Variants: `tutorial` (as written), `x4` (rewrite: one Philox call per four elements via `tl.rand4x`, 64-bit offsets) and `torch` (`F.dropout`, `manual_seed` per call). The `int64 offsets` variant is the tutorial kernel with `pid` cast to int64 and nothing else changed.
- Keep rate: p ∈ {0.01, 0.1, 0.5, 0.9, 0.99}, 4 seeds × 2^25 elements. Lag-k agreement: 2^26 elements, lags 1–2^20. Seed agreement: 2^24 elements for 8 seed pairs, including pairs that differ only above bit 32.
- p = 0 test: uint8 ones with 2^31 − 4096 elements (the largest size with safe int32 offsets), 32 seeds, tutorial kernel as written.
- Overflow test: uint8, 2^31 + 2^20 elements, each kernel in its own child process.
- Throughput: GB/s counts `2 · n · bytes`, median of three `do_bench` runs (L2 flushed), sizes 2^20 / 2^24 / 2^27 in fp32 and fp16. The tutorial's BLOCK_SIZE / `num_warps` sweep (and the rewrite's) runs at 2^26 elements first; the "best config" columns use the winner.

## Findings

1. **The mask is statistically fine.** Keep rate matches 1 − p for all five p (largest |z| 1.4 for the tutorial, 2.5 for the rewrite, 1.1 for torch). Lag-k and seed-to-seed agreement stay at 0.5 (largest |z| 2.0 tutorial, 1.8 torch; the rewrite's 3.0 at lag 8 is one of 27 lag tests and does not repeat at neighbouring lags). Seeds that differ only above bit 32 (0 vs 2^32, 7 vs 2^32 + 7) and −1 vs 2^32 − 1 give independent masks.
2. **The mask does not depend on the launch.** BLOCK_SIZE 128–8192 × 1–8 warps and the int64 rewrite are bit-identical to the default for n = 2^24 + 37. fp16 / bf16 outputs equal the fp64 reference rounded to the dtype (max error 0, fp32: 1.2e-7).
3. **`p = 0` is not the identity.** `rand` folds a signed int32 onto [0, 2^31), so exactly 0.0 occurs with probability 2^-31, and `random > 0` is false there. 37 of 6.9e10 elements were dropped (32 expected), and re-evaluating `tl.rand` at those offsets gives exactly 0.0. A single dropout call on a 2.1e9-element tensor zeroes about one element even with p = 0; `F.dropout(p=0)` is a no-op.
4. **No input validation.** p = 1.5 returns all zeros and p = −0.5 keeps everything scaled by 0.667, where torch raises `ValueError`. Non-contiguous x trips an assert; n = 0 works.
5. **NaN / inf at dropped positions disappear.** With p = 0.5, 50.6% of 3156 non-finite inputs stay non-finite (those that were kept). Torch computes `x * mask * scale`, so 100% stay non-finite (`inf * 0 = NaN`). That hides overflow from a dropped activation, so a loss-scaler or NaN check downstream sees fewer events.
6. **A seed is a mask, not a stream.** Two stacked layers with the same seed keep 50% instead of 25% and the output mean is 2.0 instead of 1.0; four layers give 8.0 (bias 2^(L−1)). Torch's advancing RNG gives 1.0 with 6.25% kept. The seed must change per layer and per step; the tutorial does not do this for you.
7. **The memory and recompute claims hold.** Between forward and backward, a seed-based `autograd.Function` holds 0 bytes per element; `F.dropout` holds a 1 byte/element mask (128 MiB at 2^27 fp16), and `torch.compile(F.dropout)` also holds 128 MiB. The backward `seeded_dropout(dy, p, seed)` equals the forward mask applied to dy. Under non-reentrant checkpointing with `preserve_rng_state=False`, the torch dropout gradient is wrong (relative error 0.81) while the seeded one is exact, like torch with `preserve_rng_state=True`.
8. **Past 2^31 elements it crashes.** `pid * BLOCK_SIZE` is int32. At 2^31 + 2^20 uint8 elements the tutorial kernel raises an illegal memory access (wrapped negative offsets). The int64 variant completes with tail keep rate 0.5003, a tail that is not a copy of the head, and head samples equal to the tutorial's output below the limit. The int64 rewrite costs nothing (0.99–1.01× at 2^24 and above in the throughput table).
9. **It runs at copy bandwidth and nothing beats it.** At 2^24–2^27 elements the tutorial kernel does 226–231 GB/s, equal to `copy_` for both dtypes. Philox is not the limit: `rand` alone does 83 G calls/s (1.4× what fp16 dropout needs at 2^27, 2.9× for fp32), so one-call-per-four-elements (`rand4x`, 267 G numbers/s) gains nothing (0.99–1.03× of the tutorial for n ≥ 2^24). Tuning BLOCK / warps gains at most +5% (fp16 at 2^27, 1024 × 16 warps); the other fp32 and fp16 configurations sit within about 2% of each other, and a few lose a lot (fp16 BLOCK 256 with 16 warps: 127 GB/s).
10. **Versus torch.** `F.dropout` reaches 0.85–0.89× of the seeded kernel in fp32 and 0.76–0.79× in fp16 at 2^24+, since it also writes a mask. `torch.compile(F.dropout)` matches the seeded kernel in fp32 (1.0×) but reaches only 0.83–0.87× in fp16, and is 0.67× at 2^20 in fp16 where CPU overhead dominates. The tutorial's own mask-based path is the slow one: with a precomputed int32 mask it runs at 0.67–0.68× (fp32) and 0.49–0.53× (fp16); building the mask with `torch.rand` first makes it 3.1× (fp32) to 5.2× (fp16) slower than the seeded kernel.

Takeaway: the tutorial's seeded kernel is a good dropout. It is statistically sound, deterministic across launch configs, keeps no state and runs at copy bandwidth, which is 13–32% faster than torch's eager dropout at 2^24+ elements and keeps a checkpointed backward exact. The sharp edges are all outside the Philox call: no input validation, `p = 0` drops about 1 in 2^31 elements, dropped NaN / inf vanish, int32 offsets crash at 2^31 elements (cast `pid` to int64, free), and seed management is up to the caller (reuse a seed across layers and you get a 2^(L−1) bias).

Full numbers: [results.md](results.md) / [results.json](results.json).
