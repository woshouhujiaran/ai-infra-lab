# 0016 — Triton tutorial matmul vs cuBLAS: autotune, grouped ordering, fp16 / fp8 accumulation

- **Upstream**: [triton-lang/triton](https://github.com/triton-lang/triton) (shallow submodule `upstream/triton`, already pinned for 0010 at tag `v3.8.0`, matching the installed `triton-windows` 3.8.0). The tutorial file is imported unmodified; `run.py` executes it only up to its unit test, so the tutorial's own benchmark and plot are skipped.
- **Component**: `python/tutorials/03-matrix-multiplication.py`. This covers `matmul_kernel` (one C tile per program, grouped `GROUP_SIZE_M` program ordering "to promote L2 data reuse", fp32 accumulator, optional fused `leaky_relu`, int32 pointer offsets), its 15-entry CUDA autotune list keyed on `(M, N, K)`, and the `matmul()` wrapper, which also takes fp8 inputs. The fp16-accumulator, int64-offset and fp8-promotion variants are copies of `matmul_kernel` in `run.py` (`variant_kernel`).
- **Hardware**: RTX 4060 Laptop 8 GB (sm_89, 24 SMs, 99 KiB shared memory per block, 32 MB L2, 256 GB/s theoretical DRAM), torch 2.14.1+cu126, triton-windows 3.8.0, memory capped at 60% by `lab/gpu_guard.py`.
- **Run**: `.venv\Scripts\python.exe experiments\0016-triton-tutorial-matmul\run.py` (about 13 min; it uses a fresh `TRITON_CACHE_DIR`, so the first autotune of each kernel includes cold compilation).

## Question

The tutorial claims its kernel reaches cuBLAS performance, that grouped tile ordering is worth more than 10% (220 → 245 TFLOPS on A100), and that fp8 inputs "just work". On a laptop Ada GPU: which of the 15 autotune configs even fit, and what does autotuning cost when shapes change, as with LLM batch sizes? Does grouping matter, and why? What do fp16 accumulation and fp8 buy in speed, and what do they cost in accuracy? Where does the kernel break: unaligned shapes, or operands with 2^31 or more elements?

## Setup

- **Configs**: each of the 15 CUDA configs is compiled for 4096³ fp16 and fp8 e4m3. The run records shared memory, registers and spills, and whether the kernel launches (99 KiB limit).
- **Square sweep**: M = N = K ∈ {256 … 8192}. Variants:
  - `cublas` (`torch.matmul`) and the `tutorial` as-is;
  - `acc16`: fp16 inputs with an fp16 MMA accumulator;
  - `int64`: int64 row and column offsets;
  - the tutorial on fp8 `e5m2` / `e4m3` (B column-major, as in the tutorial);
  - `promote_e4m3`: each K block's `tl.dot` is added into the fp32 accumulator via a runtime scale;
  - `torch._scaled_mm` e4m3 with unit scales, with and without `use_fast_accum`.

  Every Triton variant has its own autotuner over the tutorial's config list.
- **`GROUP_SIZE_M` sweep**: the tile is fixed to the autotuned winner, and GROUP_SIZE_M ∈ {1 … 64} (1 = plain row-major order). Shapes are 4096³, 8192³ and 4096×14336×4096. Outputs are checked bit-identical.
- **LLM linear**: `x[M, K] @ W.T` with W of shape [N, K] like `nn.Linear`, for (N, K) ∈ {4096×4096, 14336×4096} and M ∈ {1, 16, 128, 1024, 4096}. Variants are `F.linear`, the tutorial given the column-major `W.T` view, and the tutorial on a pre-transposed row-major copy.
- **Re-autotuning**: a fresh autotuner runs `x[M, 4096] @ W.T` for every M = 1..32.
- **Epilogue**: fused `leaky_relu` vs tutorial + `F.leaky_relu` vs cuBLAS + `F.leaky_relu`.
- **Numerics**: a 512×512 output with K ∈ {256 … 65536} and N(0, 1) inputs, compared with fp64. fp8 inputs are unscaled casts, as in the tutorial. Accumulation error is measured against fp64 of the already-quantized inputs.
- **Unaligned shapes**: K or N ∈ {4095, 4097, 4104}, M = 4097, and 4095³.
- **int32 offsets**: A [M, 4096] fp16 with M·K = 2^31 exactly vs 2^31 + 2^20 elements (4.0 GiB), B [4096, 64]. Each case runs in a fresh process, and every row is compared with cuBLAS.
- **Timing**: `triton.testing.do_bench` medians with L2 flushed before each rep. Each figure is the best of 4 rounds whose order alternates, after 2 s of GPU warm-up; medians are kept in results.json.
- **Clock state**: the laptop GPU's clock varies with CPU load, which suggests power sharing. A cuBLAS 4096³ reference measured 22.3–27.4 TFLOPS at the start of different sections. In a first run without warm-up and alternating order, whole sections ran at half speed. Compare variants within a table, not across tables.

## Findings

1. **One third of the autotune list cannot run on this GPU, and the fp8 configs spill.** All five fp16 configs with `BLOCK_SIZE_K = 128` need 192–240 KiB of shared memory against a 99 KiB limit. The autotuner silently scores them as `inf`. `128x256x64` with 3 stages fits at exactly 96 KiB. In fp8, two configs do not fit. The three remaining "good config for fp8 inputs" tiles (which look tuned for Hopper) and the large fp16 tiles hit 255 registers with 62–162 spills. The autotuner never picks them and settles on `128x64x64`. The 10 fp16 configs that fit span only 4.9–5.9 ms at 4096³ (20%). The winner changed between runs (`128x256x64` vs `64x128x32`), so the choice is partly noise.
2. **fp16 performance matches cuBLAS, including at the small end.** From 2048 to 8192 the tutorial runs at 0.91–1.02× cuBLAS (25–30 TFLOPS). At 256–1024 it is 1.14–1.18× faster than cuBLAS. The `int64` copy runs at 0.95–1.11× the tutorial, so int64 offsets cost nothing measurable.
3. **Autotuning per shape costs seconds, and the cache key includes M.** With compiled kernels already cached, each new (M, N, K) takes 1.2 s to tune at small sizes and 4.8 s at 8192³. It takes 13–17 s when a new integer specialization has to compile, for example the first M that is not a multiple of 16. Serving batch sizes 1..32 with one weight cost 54.5 s of first calls, against 0.25 ms for a warm call.
4. **Grouped ordering matters, but only once B or A no longer fits in L2.** At 4096³ (B = 32 MiB, the size of L2) all group sizes are within 3%. At 8192³ with 64×128 tiles, plain row-major order (g = 1) runs at 15.8 TFLOPS vs 25.9–27.5 for any g from 2 to 64 (1.74×). At 4096×14336×4096 the gap is 15.5 vs 25.0–26.0 TFLOPS (1.68×).

   A DRAM model explains the g = 1 numbers exactly. Each row of tiles re-streams all of B: 128 × 128 MiB = 17.2 GB in 69.7 ms, and 64 × 112 MiB = 7.5 GB in 31.0 ms. Both are 242–247 GB/s, about the 256 GB/s DRAM peak.

   An earlier run whose autotuner picked the transposed tile (128×64) showed the mirror image at 8192³. There g = 1 ran at full speed and g = 64 (column-major order, re-streaming A 128 times) fell to 15.4 TFLOPS.

   So the tutorial's default of g = 8 is safe here, and the claimed gain is real: more than 1.7× rather than 10%. It appears only when an operand exceeds L2.
5. **GeForce Ada runs fp16-accumulate and fp8 at twice the fp16/fp32-accumulate rate.** Switching only the accumulator to fp16 (`acc16`) gives 1.77–1.94× cuBLAS from 1024 up (46–56 TFLOPS from 2048). From 2048 up, the tutorial's fp8 path gets 42–51 TFLOPS (1.67–1.82× cuBLAS fp16), and `torch._scaled_mm` e4m3 gets 50–58 TFLOPS, still 1.11–1.19× faster than the tutorial's fp8.

   On accuracy, `acc16` error grows with K: 4× the fp16 output-rounding floor at K = 256, 16× at 4096 and 64× at 65536 (relative RMS 0.0133). Even so, at K ≤ 16384 that is 6–45× smaller than the 3.8% (e4m3) or 7.5% (e5m2) error from casting the inputs to fp8 alone. So for fp16 weights on this GPU, `acc16` gets fp8 speed with far less error.
6. **The tutorial's fp8 path accumulates inside the tensor core at reduced precision; neither an add nor `max_num_imprecise_acc` fixes it.** Against fp64 of the quantized inputs, its error grows from 1.5× the floor at K = 256 to 34× at K = 65536 (0.0071 for e4m3, 0.0052 for e5m2). `torch._scaled_mm` with `use_fast_accum=True` gives identical numbers at every K. The default `_scaled_mm`, which promotes partial sums to fp32, stays at the floor.

   Writing `acc += tl.dot(a, b)` changes nothing, because Triton folds the add back into the MMA accumulator. `tl.dot(..., max_num_imprecise_acc=BLOCK_K)` is also ignored on sm_89.

   What works is `acc += tl.dot(a, b) * scale` with a runtime `scale` (1.0 here; blockwise-scaled fp8 GEMMs have this form anyway). It holds the floor (0.000225) up to K = 65536 and costs 0–7% throughput (0.93–1.02× the tutorial's fp8).
7. **cuBLAS's default fp16 reductions also lose precision at large K.** With the default `allow_fp16_reduced_precision_reduction = True`, cuBLAS error at this 512×512 output grows to 1.5–3.9× the floor from K = 1024 to 65536. Setting the flag to False, or using the tutorial's fp32 accumulator, stays at 1.0–1.1×.
8. **For LLM linears, B layout and the fused epilogue barely matter, and cuBLAS wins only at decode.** At M = 1, `F.linear` is 10–13% faster (189–231 GB/s of weight traffic vs 171–206). At M = 16–128 the tutorial is up to 10% faster. At M ≥ 1024 they are within 5%. Passing the column-major `W.T` view costs at most 2%, except at 4096×14336×4096, where it was 8% slower than a row-major copy (12% in an earlier run).

   Fusing `leaky_relu` makes it essentially free (4.15 vs 4.10 ms without activation). The separate `F.leaky_relu` pass adds about 4% at N = 14336 (one extra read and write of C). Fused and separate outputs differ in about 13% of elements by at most 3e-5, with the same error vs fp64.
9. **Unaligned leading dimensions cost both libraries about a third.** K = 4095 or N = 4097 drops cuBLAS to 18.3–18.4 TFLOPS (0.66× of aligned) and the tutorial to 19.2–19.7 (0.69–0.71×). The tutorial is 5–7% ahead of cuBLAS there. K = 4104 (a multiple of 8 but not 16) is back to near full speed: 1.02× for cuBLAS and 0.94× for the tutorial. 4095³ is the tutorial's worst case: 15.6 TFLOPS, 0.83× cuBLAS. All results are correct.
10. **The kernel crashes once an operand reaches 2^31 elements.** `offs_am * stride_am` is computed in int32. With A = [524288, 4096] (largest offset exactly 2^31 − 1) every row is correct. With 256 more rows (4.0 GiB, which still fits on this 8 GB GPU), the launch fails with `cudaErrorIllegalAddress` and the CUDA context is lost. Casting the row and column offsets to int64 makes the same case correct, at no measurable cost (finding 2). The B and C offsets use the same int32 pattern. For example, logits for 16k tokens × a 152k vocabulary would exceed 2^31 elements of C.

Takeaway: on an RTX 4060 Laptop the tutorial kernel is a credible cuBLAS replacement for fp16 (0.91–1.18×), and its grouped ordering is essential once an operand exceeds L2. Before copying it into real code:

- Drop the five configs with `BLOCK_SIZE_K = 128` on consumer GPUs.
- Do not autotune on raw M in a server; bucket M or pin configs.
- Cast offsets to int64.
- For fp8, add each K block's partial product into an fp32 accumulator outside the tensor core (`acc += dot * scale`) to avoid in-MMA accumulation error that grows with K.

On GeForce, an fp16 accumulator gives the same 2× speedup as fp8 with about 10× less error, if 0.3% relative error at K = 4096 is acceptable.

Full numbers: [results.md](results.md) / [results.json](results.json).
