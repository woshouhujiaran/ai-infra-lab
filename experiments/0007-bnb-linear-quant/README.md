# 0007 — bitsandbytes Linear8bitLt / Linear4bit vs fp16 Linear

- **Upstream**: [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) (submodule `upstream/bitsandbytes` pinned at tag `0.50.2`; the same version is installed from PyPI and is what runs)
- **Component**: `bitsandbytes/nn/modules.py` — `Linear8bitLt` (LLM.int8(), inference mode `has_fp16_weights=False`) and `Linear4bit` (FP4 / NF4, optional double-quantized statistics)
- **Hardware**: RTX 4060 Laptop 8 GB, torch 2.14.1+cu126, memory capped at 60% by `lab/gpu_guard.py`
- **Run**: `.venv\Scripts\python.exe experiments\0007-bnb-linear-quant\run.py` (about 45 s; peak GPU memory 550 MiB)

## Question

How much weight memory do bitsandbytes' 8-bit and 4-bit linear layers actually save on real LLM layer shapes, how much output error do they add (and how does LLM.int8()'s outlier threshold and NF4 vs FP4 change that), and what do they cost or gain in latency from single-token decode to 2048-token prefill on a laptop GPU?

## Setup

- Layer shapes (no bias): Qwen3-0.6B `up_proj` 1024→3072, Llama-7B `q_proj` 4096→4096, Llama-7B `up_proj` 4096→11008.
- Weights: Gaussian, std `1/sqrt(in_features)`, stored as fp16. For 4096→4096 also a heavy-tailed Student-t (df=3) matrix rescaled to the same std.
- Variants (all fp16 activations / compute dtype):
  - `fp16`: `torch.nn.Linear`.
  - `int8`: `Linear8bitLt(threshold=0)` — row-wise int8 weights, row-wise int8 activations, int8 matmul.
  - `int8_t6`: `Linear8bitLt(threshold=6.0)` — activation columns with any |x| > 6 go through an fp16 matmul instead.
  - `fp4`, `nf4`: `Linear4bit`, blocksize 64, fp32 absmax per block.
  - `nf4_dq`: NF4 with `compress_statistics=True` (absmax themselves quantized to 8 bit).
- Resident memory: `torch.cuda.memory_allocated()` delta after the layer is moved to the GPU and the fp16 staging copy is freed (includes the caching allocator's rounding of large blocks to 2 MiB).
- Numerics: 256 tokens; reference is `x @ W.T` in fp64 with the original fp16 weights. "Outlier" activations: Gaussian input with `in_features/512` random channels scaled 25× (the emergent-feature pattern LLM.int8() targets). Weight error uses the layer's own dequantization.
- Latency: 1 / 16 / 256 / 2048 tokens, warmup 10, 100 calls timed with CUDA events, `torch.inference_mode()`. Int8 variants are also timed on outlier inputs. Numbers below are from the committed run; a second run agreed within ~5% on the 4096-wide shapes.

## Findings

1. **Memory matches the format math.** int8 is 8.0 bits/weight (0.50× fp16), FP4/NF4 4.5 bits/weight (0.28×: 4 bits + one fp32 absmax per 64 weights), and double quant brings NF4 to 4.13 bits/weight (0.26×). The 11008-wide shape shows ~0.1–0.2 bits more only because of 2 MiB allocator rounding.
2. **int8 is ~1% error until activations have outliers.** With Gaussian inputs, `int8` output error is 1.1–1.2% (weight error alone 0.8–0.9%). With 25× outlier channels, threshold 0 jumps to 4.9–7.2% because each row's activation scale is set by the outlier. `threshold=6.0` brings it back to 1.0–1.1% (slightly better than the no-outlier case, since the large columns are computed in fp16).
3. **NF4 beats FP4, and the gap grows with heavy tails.** On Gaussian weights NF4 has 9.2% output error vs FP4's 12.2%. On Student-t weights: NF4 12.2%, FP4 18.7% (int8 4.4%). 4-bit error does not change with activation outliers (activations stay fp16). Double quant adds at most 0.0004 relative error for a 0.37 bits/weight saving.
4. **Decode (1 token): 4-bit is the only clear win.** On the 4096-wide shapes NF4/FP4 run 1.2–1.5× faster than fp16 (bandwidth-bound GEMV on a 3.6× smaller weight). `int8` is about parity (0.84–1.11×), and `int8_t6` is slower because of the extra outlier scan (0.6–0.8× on normal inputs, 0.24–0.53× when outlier columns are present).
5. **Small batches (16 tokens) are the worst case for every quantized layer.** On the 4096-wide shapes int8 runs 0.34–0.41× of fp16 speed and 4-bit 0.28–0.43×. int8 is even slower at 16 tokens than at 256 tokens (q_proj 0.45 vs 0.37 ms), i.e. the int8 matmul path is poorly tuned for small M.
6. **Prefill (256–2048 tokens): int8 is fastest, 4-bit is about fp16 speed.** `int8` is 2.4–3.0× faster than fp16 (Ada int8 tensor cores). `int8_t6` keeps 1.3–2.0× on normal inputs and 1.0–1.7× with outliers (extra fp16 matmul on the outlier columns). NF4/FP4 are 0.64–0.75× at 256 tokens and 0.95–1.04× at 2048.
7. **4-bit prefill materializes the fp16 weight.** At 2048 tokens the 4-bit layers' temporary peak equals fp16's output buffer plus a full fp16 copy of the weight (q_proj 48 vs 16 MiB, up_proj 129 vs 43 MiB), so 4-bit saves resident memory but not peak activation-time memory per layer. int8 needs ~3× fp16's temp at 2048 tokens (int8 input copy + int32 accumulator + fp16 output: 137 vs 43 MiB for up_proj).
8. **The Qwen3-0.6B shape is too small to benefit.** Sub-0.3 ms timings are launch-bound and varied up to 2× between runs; only `int8` at 2048 tokens reliably beat fp16 (1.5×).

Takeaway: on this GPU, NF4 (with double quant) is the right choice for memory-bound single-stream decode on 4096-wide layers — 3.8–3.9× less weight memory, 1.2–1.5× faster than fp16 at 1 token, and lower error than FP4. LLM.int8() only pays off in speed for prefill-sized batches, and only keeps its ~1% error when the outlier threshold is on, which gives up 20–45% of that speedup.

Full numbers: [results.md](results.md) / [results.json](results.json).
