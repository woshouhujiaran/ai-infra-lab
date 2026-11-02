# 0009 — Liger-Kernel cross entropy / fused linear cross entropy vs PyTorch CE

- **Upstream**: [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) (submodule `upstream/Liger-Kernel` pinned at tag `v0.8.5`; imported straight from `upstream/Liger-Kernel/src`, nothing installed)
- **Component**: `liger_kernel/ops/cross_entropy.py` (`LigerCrossEntropyFunction`, Triton online-softmax CE that writes the gradient into the logits buffer) and `liger_kernel/ops/fused_linear_cross_entropy.py` (`LigerFusedLinearCrossEntropyFunction`, LM head matmul + CE in token chunks so the full logits are never materialized)
- **Hardware**: RTX 4060 Laptop 8 GB, torch 2.14.1+cu126, triton-windows 3.8.0, memory capped at 60% (4.9 GB) by `lab/gpu_guard.py`
- **Run**: `.venv\Scripts\python.exe experiments\0009-liger-fused-ce\run.py` (about 4 min including `torch.compile`)

## Question

How much memory do Liger's CE kernels actually save on a laptop GPU for one LM-head + loss training step, what do they cost in time, and does chunking the fused linear CE (and accumulating dW in bf16) cost accuracy? At what token count does each approach stop fitting in 4.9 GB?

## Setup

- LM head shapes (hidden, vocab), no bias: Qwen3-0.6B (1024, 151936), Llama-3.2-1B (2048, 128256), and 1024 / 32000 (Llama-2 / Mistral-sized vocab).
- Tokens per step (B×T): 1024, 4096, 16384. Hidden states `x ~ N(0,1)`, weights `N(0, 1/H) × 3` (logit std ≈ 3), all bf16, both `x` and `W` require grad. Mean reduction, no ignored tokens.
- Variants:
  - `torch_bf16`: `F.cross_entropy(x @ W.T)` on bf16 logits.
  - `torch_fp32`: `F.cross_entropy((x @ W.T).float())`, the Hugging Face upcast path (baseline for speedups).
  - `torch_compile`: `torch.compile(torch_fp32, dynamic=False)`.
  - `liger_ce`: bf16 `x @ W.T`, then `LigerCrossEntropyFunction`.
  - `liger_flce`: `LigerFusedLinearCrossEntropyFunction` with defaults (`chunk_mem_const=1`, dW accumulated in bf16).
  - `liger_flce_c8`: same with `chunk_mem_const=8` (chunks 8× larger).
  - `liger_flce_fp32acc`: defaults plus `accum_dtype=torch.float32`.
- Numerics: 1024 tokens, targets sampled from the model's own softmax. Reference = loss, dX, dW computed in fp32 from the same bf16 inputs. Rerun once with `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` ("strict"; PyTorch's default is `True`).
- Memory: `max_memory_allocated` during one forward + backward minus what was allocated before (x, W, targets). It includes the dX and dW the step returns (the "dX+dW" column is that floor). OOM = `torch.OutOfMemoryError` under the 60% cap.
- Latency: 2 warmup + 10 timed steps with CUDA events. A second full run agreed within ~8%.

## Findings

1. **Peak memory is set by how many logits-sized buffers are live.** With L = bf16 logits size and G = dX+dW, the measured extra peak is ≈ 6L for `torch_fp32` (bf16 logits + fp32 copy + fp32 softmax/grad temporaries) and ≈ 3L for `torch_bf16`, on every shape and token count that fit. `torch_compile` and `liger_ce` need L + G (the gradient reuses the logits buffer), and `liger_flce` G + one chunk. On the Qwen3 head at 1024 tokens that is 1782 / 890 / 596 / 301 MiB.
2. **Where each approach stops fitting in 4.9 GB.** `torch_fp32` OOMs at 4096 tokens on both large-vocab heads (and at 16384 on the 32k vocab). `torch_bf16`, `torch_compile` and `liger_ce` all OOM at 16384 tokens on the large-vocab heads (the 4.0–4.7 GB of logits alone). Only the fused linear CE runs there: 366 MiB (Qwen3) and 692 MiB (Llama-3.2-1B), of which 329 / 565 MiB are the gradients themselves.
3. **`liger_ce` ≈ `torch.compile`, both clearly better than the HF upcast.** Same memory to the MiB, same speed within 5%, 1.3–1.7× faster than `torch_fp32` with 2–4.7× less extra memory. Gradients are identical to `torch_fp32` to 4 digits. The one difference: `liger_ce` stores per-token losses in the input dtype, so the returned loss is bf16-rounded (abs error 1.1e-3–4.4e-3 vs 1e-4–2e-4 for fp32/FLCE). That only affects the logged loss value, not the gradients.
4. **Default FLCE chunking costs a fixed ~`num_chunks` × (dW read-modify-write) per step, which makes it very slow on large-vocab / small-hidden heads.** `chunk_mem_const=1` sizes chunks so logits ≈ BT×H, which gives ⌈V/H⌉ → 128 chunks for Qwen3 (V/H = 148) and 32 for the others. Each chunk does an `addmm` into the full V×H dW. On Qwen3 that is ~0.67 s/step regardless of tokens (675 / 699 / 776 ms at 1k / 4k / 16k), versus 35 ms for `liger_ce` at 1024 tokens (19× slower). The overhead is amortized only once the matmuls dominate: at 16384 tokens default FLCE runs at the same speed as `chunk_mem_const=8` on Llama-3.2-1B (973 vs 978 ms ≈ 26 TFLOP/s) and as `liger_ce` on the 32k vocab (129 vs 125 ms).
5. **`chunk_mem_const=8` is the better setting on this GPU.** It cuts chunks 8× (Qwen3: 16) for +14–376 MiB of memory and is 3.7–7.3× faster than the default at 1024 tokens, 1.3–4.6× at 4096, and 1.0–1.3× at 16384. Wherever `liger_ce` fits, it is within 1.0–1.1× of `liger_ce` speed, except at 1024 tokens on the large-vocab heads (2.6× slower on Qwen3, 1.3× on Llama-3.2-1B).
6. **The one large error was cuBLAS, not Liger.** With PyTorch's default `allow_bf16_reduced_precision_reduction=True`, `liger_flce_c8` on Llama-3.2-1B had 1.18% dX error (7× the 0.17% bf16 rounding floor). Its 128-token chunks produce an `M=128, K=128256` bf16 GEMM that cuBLAS runs with split-K reductions in bf16. A plain `torch.mm` of that shape shows the same 1.1%. With the flag off it drops to 0.26%, like every other variant. The same effect inflates the full-batch PyTorch dX (`torch_*`, `liger_ce`: 0.35–0.39% default vs 0.26–0.31% strict). Default FLCE's smaller chunks mostly escape it: identical either way on the large-vocab heads, 0.35 vs 0.31% on the 32k head.
7. **bf16 dW accumulation across chunks is cheap in accuracy, and fp32 accumulation is not worth its cost.** Accumulating dW in bf16 over 32–128 chunks raises dW error from 0.243–0.312% (`accum_dtype=fp32`, identical to PyTorch) to 0.262–0.384%. That is +8% relative on the large-vocab heads and +23% on the 32k head: 1.9–2.3× the 0.14–0.16% bf16 rounding floor, vs 1.7–1.9× for PyTorch. `accum_dtype=fp32` adds a full fp32 dW (+590 MiB / +1 GiB, as much memory as `torch_fp32` at 1024 tokens on Llama-3.2-1B) and runs 1.5× slower when chunk-bound.

Takeaway: on an 8 GB laptop, when the logits fit (≲ 4k tokens with a 128k–152k vocab), use `liger_ce` or simply `torch.compile` the CE: 2–4.7× less memory than the HF fp32 upcast and 1.3–1.7× faster. Beyond that, Liger's fused linear CE is the only option that runs (16k tokens in < 0.7 GB). Raise `chunk_mem_const` to ~8 there, because the default's per-chunk dW updates make it up to 19× slower on Qwen-style heads. Leave `accum_dtype` at bf16. If you care about dX precision, turn off `allow_bf16_reduced_precision_reduction` rather than relying on chunk sizes.

Full numbers: [results.md](results.md) / [results.json](results.json).
