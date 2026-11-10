# 0017 — gpt-fast weight-only int8 / int4 quantization

- **Upstream**: [meta-pytorch/gpt-fast](https://github.com/meta-pytorch/gpt-fast) (submodule `upstream/gpt-fast` at `6ecad9b`). `quantize.py` and `model.py` are imported unmodified. `run.py` replaces only `quantize.prepare_int4_weight_and_scales_and_zeros` with a repacking copy (see finding 1). It also swaps `model.flex_attention` for causal SDPA, which is exact for the teacher-forced forwards from position 0 used here.
- **Component**: gpt-fast's weight-only quantizers in `quantize.py`:
  - `WeightOnlyInt8QuantHandler` + `WeightOnlyInt8Linear`: symmetric per-output-channel int8. The forward is `F.linear(x, w_int8.to(x.dtype)) * scales`, with the scales stored in bf16.
  - `WeightOnlyInt4QuantHandler` + `WeightOnlyInt4Linear`: asymmetric groupwise int4, with bf16 scales and zeros per group of 32–256 input columns. Weights are packed for torch's tinygemm kernel `aten._weight_int4pack_mm`. Layers whose `in_features` doesn't fit are padded up to a multiple of 1024.
- **Models**:
  - [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas) stories110M (12 layers, dim 768, hidden 2048, 32k vocab, shared embedding / classifier), loaded into gpt-fast's `Transformer` as in [0014](../0014-gptfast-speculative-decode). stories15M (dim 288) is used only for the padding check.
  - For speed, synthetic bf16 weights with the stories110M and Llama-2-7B linear shapes.
- **Hardware**: RTX 4060 Laptop 8 GB (through `lab/gpu_guard`, peak 2.9 GiB allocated), i9-13900H, torch 2.14.1+cu126, triton-windows.
- **Run**:

  ```powershell
  $d = "$env:USERPROFILE\.cache\ai-infra-lab\tinyllamas"; mkdir $d -Force
  foreach ($n in "stories15M","stories110M") { curl.exe -L -o "$d\$n.bin" "https://huggingface.co/karpathy/tinyllamas/resolve/main/$n.bin" }
  .venv\Scripts\python.exe experiments\0017-gptfast-weight-only-quant\run.py
  ```

  The download is 499 MB. The run takes about 8 minutes.

## Question

Do gpt-fast's int8 and int4 weight-only paths still work at HEAD? How much accuracy does each cost on a real (small) Llama, layer by layer and end to end? And where on a laptop GPU do they actually beat bf16: at what token count M, and with or without `torch.compile` / CUDA graphs?

## Setup

1. **Upstream as-is**:
   - Run `WeightOnlyInt8QuantHandler` and `WeightOnlyInt4QuantHandler` (g128, `inner_k_tiles` 8) on stories110M, plus the helper round trip `group_dequantize_tensor(*group_quantize_tensor(w))`.
   - The int4 failure is fixed by repacking. The fix and two plausible wrong fixes are checked against an fp64 matmul with the dequantized weight.
   - Also checked: torch's own `aten._weight_int8pack_mm` on CUDA, and int4 storage for stories15M, whose 288-wide layers hit the padding path.
2. **Per-layer error**:
   - Data: the fp32 model samples 8 × 256 tokens at T = 1. The real fp32 inputs of all 61 linears are recorded on 4 × 256 of those tokens.
   - Metric: relative Frobenius error of the layer output against `x @ W.T` in fp64. Methods: bf16 `F.linear`, upstream int8, and int4 g32 / g64 / g128 / g256 through tinygemm.
   - To split the error apart:
     - *int8 ideal* uses the same codes with fp32 scales and fp64 math;
     - *bf16 scales* rounds the scales to bf16 and keeps fp64 math;
     - *int4 ideal* is the dequantized int4 weight in fp64;
     - *nearest codes* computes `(w - min) / scale` in fp32 instead of the bf16 that upstream inherits from the bf16 weight.
3. **Whole model**: the same 8 × 256 tokens, teacher-forced through bf16, int8 and int4 g32–g256 models built by the upstream handlers. Metrics against the fp32 model: mean / p99 KL(fp32 ‖ quant), top-1 agreement, and perplexity of the sampled tokens.
4. **Speed**:
   - Grid: one linear at M = 1 … 2048 tokens, for the stories110M shapes (wqkv, wo, w1/w3, w2, 32k output head) and the Llama-2-7B shapes (12288×4096, 4096×4096, 11008×4096, 4096×11008).
   - Methods: bf16 `F.linear`; upstream int8 forward, eager and `torch.compile` (`fullgraph`, static shapes, `coordinate_descent_tuning` as in gpt-fast's `generate.py`); `_weight_int8pack_mm`; int4 g32 / g128 tinygemm, with an `inner_k_tiles` 2 / 4 / 8 sweep at M = 1 and 16.
   - Each method rotates through enough weight copies (≥ 96 MiB, 3× L2) that every call streams its weight from DRAM, as in a real decode step.
   - Timed both as an eager launch loop and as CUDA-graph replay (GPU time only, which is what gpt-fast's `reduce-overhead` decode gets). Inductor's generated code is checked for an `extern_kernels` matmul call.

## Findings

1. **At HEAD, int4 quantization fails before producing a single weight; int8 works.**
   - `WeightOnlyInt4QuantHandler` raises `Expected in.dtype() == at::kByte` at `quantize.py:363`. Since torch 2.5, `_convert_weight_to_int4pack` takes `[N, K/2]` uint8 with two codes per byte, not `[N, K]` int32.
   - Packing the even column into the high nibble is correct: the error against the fp64 dequantized weight is 2.3e-3 relative, which is bf16 output rounding. The whole int4 model then runs.
   - The other nibble order is accepted without complaint and returns garbage (relative error 1.40).
   - The naive `.to(torch.uint8)` without packing fails later, in `_weight_int4pack_mm`, with a shape check.
   - Separately, the GPTQ helper round trip `group_dequantize_tensor(*group_quantize_tensor(w))` trips its own `assert scales_and_zeros.dtype == torch.float` (`quantize.py:98`): the quantizer emits bf16 scales and zeros, but the dequantizer insists on fp32.
2. **Padding to a multiple of 1024 can make int4 bigger than bf16.**
   - stories15M's 288-wide inputs are a multiple of 32 but not of `inner_k_tiles · 16` = 128, so 25 of 31 linears are padded to 1024 columns.
   - With g32, the int4 linears take 30.2 MiB against 29.0 MiB in bf16. Padding only to lcm(groupsize, 128) = 384 would give 11.8 MiB (10.0 MiB at g128).
   - The padded forward is correct (2.3e-3 against the dequantized weight).
3. **Per layer, int8 costs 2.6–3.4× the bf16 error in transformer blocks, and much more on the output head.**
   - Transformer blocks: bf16 gives 2.2–2.9e-3 and upstream int8 5.8–9.9e-3. Most of that is the int8 rounding itself (ideal: 4.9–9.4e-3). Rounding the scales to bf16 adds about 7% (wqkv: 4.85 → 5.18e-3), and the rest comes from bf16 activations and outputs.
   - The shared embedding / output head has rows with max/RMS 18 (3.3–3.6 elsewhere), so per-row int8 steps are coarse there: 1.8e-2, 7× bf16.
   - int4 costs 4.6–7.8e-2 at g32 and 5.9e-2–1.0e-1 at g128. Group size matters most on the outlier-heavy head: 4.7e-2 at g32, 9.3e-2 at g128, 0.15 at g256.
   - Upstream's bf16 quantization arithmetic gives 2.0–2.3% of int4 weights a non-nearest code. Fixing it changes the layer error by < 1% (g128 wqkv 5.92 → 5.90e-2), so it is harmless.
4. **End to end, int8 is essentially free and int4 costs about 1–1.5% perplexity.**
   - int8: mean KL to fp32 is 3.6e-4 (bf16: 9.8e-5), top-1 agreement 99.1% (bf16 99.6%), and ppl 2.066 against 2.065 for both fp32 and bf16.
   - int4: KL 1.0e-2 (g32) to 2.1e-2 (g256), top-1 95–96%, ppl +1.0% at g32 and +1.5% at g128.
   - Bits per linear weight: 8.02 for int8; 5.00, 4.50, 4.25 and 4.12 for int4 g32, g64, g128 and g256.
5. **At M = 1 under CUDA graphs, every fused path runs at DRAM bandwidth, so the speedup is just the bytes ratio.**
   - On the 7B shapes, bf16, compiled int8 and tinygemm int4 all reach 241–252 GB/s.
   - One 7B layer's linears: bf16 1611 µs; int8 compiled 808 µs (1.99×); int4 g128 434 µs (3.71×); int4 g32 510 µs (3.16×, its scales add 18% bytes).
   - stories110M per token (12 layers + head): 1006 → 509 µs (int8, 1.97×) and → 316 µs (int4 g128, 3.19×). Its small block layers only reach 120–225 GB/s.
   - Upstream's int8 forward *without* compile is 0.36× bf16: it materializes the whole bf16 weight on every call (45 GB/s effective).
6. **Compiled int8 is fused only at M = 1; from M = 2 it is 2.4× slower than bf16.**
   - At M = 1, Inductor emits one Triton reduction kernel that reads int8. At M ≥ 2 the generated code dequantizes into a temporary bf16 weight and calls the extern cuBLAS mm.
   - One 7B layer then takes 4044–4272 µs at M = 2–64, against 1639–1756 µs for bf16 (0.41×). It only recovers to 0.89× at M = 2048.
   - So gpt-fast int8 pays off only for batch-1 decoding. A batch of 2, or a speculative verify of k + 1 tokens, is slower than plain bf16.
7. **tinygemm int4 wins up to about 16 tokens and loses 4× at prefill sizes.**
   - On the 7B layer: flat time up to M = 4 (3.7× faster than bf16), 3.2× at M = 8, 1.9× at M = 16, break-even at M = 32 (0.96×), 0.50× at M = 64, and 0.24–0.25× from M = 256.
   - At M = 2048 a 7B layer takes 111 ms against 27 ms for bf16, so prefill should not run through tinygemm.
   - `inner_k_tiles` 2 / 4 / 8 are within 10% of each other.
8. **torch's `_weight_int8pack_mm` runs on CUDA but is no alternative.** It is 0.33× bf16 at M = 1, and its time grows linearly with M: a 7B layer takes 3.85 s at M = 2048, 140× bf16.
9. **Without CUDA graphs, compiling a single quantized linear doesn't pay off at small sizes.** Each call into the compiled function adds about 50–150 µs of host overhead on Windows (bf16 `F.linear`: about 10 µs). In an eager loop, compiled int8 is 0.17–0.29× bf16 for stories110M at M ≤ 16. The 7B shapes are large enough to hide this (1.93× at M = 1).

Takeaway:
- At HEAD, gpt-fast's int4 path is broken by torch's int4 packing change. The one-line repack is easy to get silently wrong (nibble order), so it needs a numeric check against the dequantized weight.
- With the repack, both quantizers are accurate: int8 is within noise of bf16 end to end, and int4 costs about 1–1.5% perplexity on stories110M.
- Their speed is narrow, though:
  - int8 is 2× only for batch-1 decoding with `torch.compile` plus CUDA graphs, and 2.4× *slower* than bf16 from two tokens on;
  - int4 tinygemm is 3–3.7× for up to 8 tokens, even at 16, and 4× slower for prefill.
- Use the quantized layers for single-stream decoding, and keep a bf16 (or dequantize-once) path for prefill and batched or speculative verification.

Repeatability: parts 1–3 gave identical numbers in all four runs. For part 4, the last two runs agree to about 1% in nearly every cell. The first runs had no GPU warm-up and measured the first cell cold (44.9 vs 16.5 µs), so `run.py` now warms the GPU for 3 s first. An earlier run also showed single M = 2048 outliers (bf16 7B wo 2582 vs 2272 µs) that did not recur.

Full numbers: [results.md](results.md) / [results.json](results.json).
