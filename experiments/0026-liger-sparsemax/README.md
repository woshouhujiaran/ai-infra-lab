# 0026 — Liger-Kernel sparsemax (`LigerSparsemaxFunction`)

- **Upstream**: [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) (submodule `upstream/Liger-Kernel`, commit `1ab779f`), imported unmodified from `src/`.
- **Component**: `src/liger_kernel/ops/sparsemax.py`. Forward runs `torch.sort(x.float())` (values + int64 indices), then `_sparsemax_forward_kernel`: one program per row, the **whole row in one `next_power_of_2(n_cols)` block**, fp32 `cumsum` of the sorted row, support `z_k > (cumsum_k − 1) / k`, `tau = (Σ support − 1) / k`, `y = max(x − tau, 0)`. Backward recomputes the support from `y > 0` (in the input dtype) and subtracts the support mean of `grad_output`, cast to that dtype.
- **Hardware**: RTX 4060 Laptop 8 GB, torch 2.14.1+cu126, triton-windows 3.8.0, memory capped at 60% (4.8 GiB) by `lab/gpu_guard.py`. Peak allocated 3.5 GiB.
- **Run**: `.venv\Scripts\python.exe experiments\0026-liger-sparsemax\run.py` (about 1 minute; `run.py numerics edge` runs only some sections).

## Question

Upstream's tests only check fp32 `randn` rows up to 512 wide against a torch sort + cumsum reference. Where does the kernel stop matching an exact fp64 sparsemax: large vocabularies, large logit offsets, half precision, ties, non-finite logits? What does the sort + whole-row-in-a-block design cost in memory and time compared with eager torch and with a sort-free rewrite?

## Setup

- **References**: exact fp64 sparsemax (sort + cumsum, `1 + k z_k > cumsum_k`) and its analytic backward `dX = g − mean(g | support)` on the support `y > 0`. "Floor" is the rounding of the fp64 result to the input dtype. Also compared with the upstream test's eager torch implementation (fp32 sort + cumsum, autograd through it; `torch` in the tables).
- **Rewrite (`RowsSparsemax`, in `run.py`)**: no sort. Row max shift, then Michelot's fixed-point iteration (`tau ← (Σ_{z > tau} z − 1) / |{z > tau}|` from `tau = max − 1` until the support count stops changing) over 4096-column chunks, fp32 throughout; backward recomputes the support from the saved input and the stored `(max, tau)` and subtracts an fp32 support mean. 2–13 passes per row in all runs (cap 1024 never reached).
- **Numerics**: BT 64, V ∈ {128, 4096, 32768, 65536}, logits `randn * scale` with scale ∈ {0.01, 0.1, 1, 10} (support from ~320 to ~1.5 elements), fp32 / fp16 / bf16.
- **Shift**: fp32, 64 × 32768, `randn + C` for C up to 1e6. **Ties**: 2048 (V = 4096) / 512 (V = 32768) rows of `randn`, rows whose support differs from fp64. **Edge cases**: ties, `n_cols` 1 / 2, `-inf` padding, `+inf`, NaN, all `-inf`, V = 65536 / 65537 / 151936.
- **API**: `dim` ∈ {0, 1, 2, −2} on a 3D tensor, strided input, in-place edits of the output, `retain_graph`, `grad_output` with a common offset in half precision.
- **Memory / latency**: bf16 and fp32, V 128–65536 at 2^22 elements; bf16 V = 65536 and 151936 with BT up to 4096; held / peak memory over the live input, `do_bench` fwd+bwd and `no_grad` forward, share of `torch.sort`, registers / spills / shared memory per kernel.

## Findings

1. **Hard limit of 65536 columns.** `calculate_settings` raises `RuntimeError: Cannot launch Triton kernel since n = 65537 exceeds the recommended Triton blocksize = 65536` for V = 65537 and for the 151936-token Qwen / 128256-token Llama-3 vocabularies; eager torch (8e-7 off fp64 at 151936) and `RowsSparsemax` (7e-8) run them. Below the limit the whole row sits in one block: the forward kernel uses 125 registers at 16384 columns, then spills (18 at 32768, 152 at 65536 with 32 warps; backward 6 / 66).

2. **The sort is the cost: ~93% of the forward, 18× the input in peak memory, and an earlier OOM.**
   - `torch.sort(x.float())` takes 3.0 ms of the 3.2 ms forward at 128 × 32768 (bf16; 2.9 of 3.2 ms at 64 × 65536). It materializes an fp32 copy, fp32 sorted values and int64 indices: peak forward memory is 18× the bf16 input (2304 MiB at 1024 × 65536; 3–11× the fp32 input) while only the output (1×) is held afterwards. At 2048 × 65536 bf16 (256 MiB of input) it OOMs under the 4.8 GiB cap; eager torch holds 7.5× the input after forward (960 vs 128 MiB at 1024 × 65536) and OOMs at the same size.
   - `RowsSparsemax` has peak = held = the output (128 MiB at 1024 × 65536; backward 2×), and runs 4096 × 65536 (peak 512 MiB forward, 1 GiB backward) and 2048 × 151936 (594 MiB) where upstream OOMs or refuses.
   - fwd+bwd latency at 2^22 elements, upstream / `RowsSparsemax`: bf16 2.1–17× (e.g. 1.1 vs 0.52 ms at V = 2048, 3.3 vs 0.19 ms at V = 32768, 58 vs 12 ms at 1024 × 65536), fp32 2.4–8.7×. Upstream is only ~1.5–2.5× faster than eager torch, since both pay for the same sort. Cells with V ≤ 8192 vary up to 2× between runs on this laptop.

3. **fp32 error is dominated by the cumsum, not by rounding.** Against fp64, y of upstream (and of the torch reference) is 16–42× the rounding floor at scale 1 (2.4e-7–8.4e-7 rel L2) and 280–600× at scale 10 (7e-7–1.6e-6), with `|sum(y) − 1|` up to 1.1e-5 at 65536 columns; the sort-free rewrite is at 1–3× the floor (4e-8 / 2e-8) and `|sum − 1|` ≤ 2.4e-7 at scales 1–10. dX is at the floor for all of them (3e-8–4e-8) because it only needs the support. In fp16 / bf16, y is at the floor for all three.

4. **Not shift-invariant.** `sparsemax(x + C) = sparsemax(x)`, but the fp32 cumsum swallows the `− 1` once `C·k` grows: y error 1.8e-5 at C = 100, 1.2e-4 at 1e3, 1.9e-3 at 1e4, 1.5e-2 at 1e5 and 0.14 at 1e6, with the wrong support in 9 / 22 of 64 rows at C = 1e5 / 1e6 and dX 17% / 50% wrong (the torch reference behaves the same). Subtracting the row max first (`liger_sm(x − x.amax(-1, keepdim=True))`, or the rewrite's built-in shift) brings it back to 5e-8 at every C.

5. **Half-precision backward casts the support mean to the dtype.** The exact dX does not depend on a common offset in `grad_output`, but upstream computes `g − cast(mean(g))`: with `g = noise + c`, the dX error vs fp64 is 1.5× the floor at c = 1, 1.7% (bf16) / 0.2% (fp16) at c = 10, 11% / 1.7% at c = 100 and 16% / 11% at c = 1000, against 0.16% / 0.02% for an fp32 mean. In fp32 it is the same as the rewrite (5e-5 at c = 1000). dX is 1.0–1.9× the floor in half precision even with zero offset.

6. **Exact ties at tau flip the support in the rewrite, not in upstream.** Elements with `x == tau` have `y = 0` either way but different gradients (the Jacobian is discontinuous there). bf16 hits them constantly (100 of 2048 rows at V = 4096; 11 for fp16; 0 for fp32). Upstream and the torch reference decide the support exactly on the representable values (0 mismatching rows); Michelot's fp32 `tau` is rounded, so `RowsSparsemax` includes the tied element in 8 / 2048 bf16 rows (1 / 512 at 32768) and its dX is 2.1% / 0.21% off fp64's support (0.16–0.17% on its own support). Nothing else in the rewrite is less exact, but a production version would need a tie rule.

7. **Non-finite logits give silently wrong output, not NaN.** One NaN: upstream returns `[2, 0, 1.5, 1]` for `[1, nan, 0.5, 0]` (sums to 4.5; the NaN is dropped from the support and the row is not normalized); one `+inf`: `[2, inf, 1.5, 1]`; `NaN, −inf, 2, 1.5`: `[0, 0, 3, 2.5]`; all `-inf` / all NaN: zeros. The eager torch reference returns NaN (also for a harmless `-inf` padding, where `0·inf` poisons it), and the rewrite also swallows NaN / `+inf`. `-inf` padding itself is handled exactly by upstream (1.2e-7 off fp64) and the rewrite. Ties, `n_cols = 1`, `2` and uniform rows are right (uniform V = 1000: `sum(y) = 0.99993`).

8. **API.** `dim ≠ −1` is correct (y 4.5e-8–2.4e-7, dX 3e-8–4e-8) but returns a non-contiguous transposed view for `dim` 0 / 1 / −2 (strides `[1, 4, 132]` for dim 0 of a 4 × 33 × 129 tensor); strided input works. In-place edits of the output (`mul_`, `clamp_`, `zero_`) raise autograd's "view of a custom Function output" error at the edit (loud, not silent). A second `retain_graph` backward is exact.

Takeaway: use upstream sparsemax for ≤ 65536 columns, logits centred near 0 and a gradient without a large common offset in half precision — subtract the row max before calling it, keep NaN / `inf` out of the input (nothing flags them), and expect an 18× input-size peak and a sort-bound forward. For LLM vocabularies, large logit offsets, or half-precision gradients with a common offset, a sort-free kernel needs ~70 lines, uses no extra memory and is 2–17× faster, but needs an explicit tie rule.

Full numbers: [results.md](results.md) / [results.json](results.json).
