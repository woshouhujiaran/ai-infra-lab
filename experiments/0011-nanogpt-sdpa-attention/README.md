# 0011 — nanoGPT `CausalSelfAttention`: SDPA ("flash") path vs manual attention

- **Upstream**: [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) (submodule `upstream/nanoGPT` at `3adf61e`). `model.py` is loaded unmodified. To get the manual path, `run.py` hides `F.scaled_dot_product_attention` while the module is built, so nanoGPT's own `__init__` sets `self.flash = False` and registers its causal-mask buffer.
- **Component**: `model.py` `CausalSelfAttention`. With `self.flash = True` it calls `F.scaled_dot_product_attention(..., is_causal=True)`, and the code comment says this means "Flash Attention CUDA kernels". The manual path computes `q @ k^T`, applies `masked_fill` with a `block_size × block_size` tril buffer, then softmax, dropout and `@ v`.
- **Hardware**: RTX 4060 Laptop 8 GB, torch 2.14.1+cu126 (Windows wheel), cuDNN 9.10.2, triton-windows for `torch.compile`, memory capped at 60% (4.8 GB) by `lab/gpu_guard.py`.
- **Run**: `.venv\Scripts\python.exe experiments\0011-nanogpt-sdpa-attention\run.py`. It takes about 4 minutes on a cold `torch.compile` cache and about 2 minutes warm.

## Question

nanoGPT assumes that having SDPA means having FlashAttention. On a Windows laptop GPU, which kernel actually runs? How much latency and memory does it save over the manual path as the sequence length grows, in nanoGPT's two training precisions (bf16 autocast, and fp32 with TF32)? Can `torch.compile` close the gap for the manual path? What changes when attention dropout is on?

## Setup

- GPT-2 small attention: `n_embd = 768`, `n_head = 12` (head dim 64), `bias = True`, `block_size = T`, identical weights (seed 0) for every variant.
- Each batch has 16,384 tokens, with T ∈ {256, 512, …, 16384} and B = 16384 / T. The linear layers therefore cost the same at every T, and attention FLOPs grow linearly with T.
- Modes, following nanoGPT's `train.py`, which turns TF32 on in both: `bf16_autocast` (fp32 weights, `torch.autocast(bfloat16)`) and `fp32_tf32`.
- Variants:
  - `sdpa_default`: nanoGPT as-is.
  - `sdpa_cudnn` and `sdpa_math`: the same call inside `sdpa_kernel(CUDNN_ATTENTION)` or `sdpa_kernel(MATH)`.
  - `manual`: nanoGPT's manual path.
  - `manual_compile`: `torch.compile(manual, dynamic=False)`, bf16 only.
  - `linears_only`: just `c_attn` and `c_proj`, as a lower bound.
- Measurements:
  - Latency: `triton.testing.do_bench` median for forward under `no_grad` and for forward + backward.
  - Memory: extra peak allocation above inputs and weights.
  - Backend: CUDA kernel names from `torch.profiler` for one forward + backward.
  - Error: relative L2 error of y, dx and dW_qkv against an fp64 manual reference at T = 1024, B = 2.
- Repeatability: two full runs agreed within 1% for every SDPA and manual cell. The exceptions were three `linears_only` cells in the first run (up to 3× slower, which looks like transient clock behavior) and `manual_compile` at T ≤ 512, which differed by 8%. The committed numbers are from the second run.

## Findings

1. **No FlashAttention on this build, and nanoGPT's "flash" path silently uses the CUTLASS memory-efficient kernel.** `torch.backends.cuda.is_flash_attention_available()` is `False`, and forcing `FLASH_ATTENTION` raises "Torch was not compiled with flash attention". In both bf16 and fp32, the default dispatch runs `fmha_cutlassF/B_*_sm80` (PyTorch's mem-efficient attention). The faster cuDNN kernel (`cudnn_generated_fort_native_sdpa_sm80_flash_*`) runs only when forced, and only in bf16 or fp16. With fp32 inputs it raises "No available kernel".
2. **The SDPA path still wins easily.** In bf16, forward + backward of the whole module at T = 1024 takes 15.6 ms against 90.7 ms for manual (5.8×), and the forward pass alone is 8.5× faster. Subtracting the 9.9 ms that the linear layers take at every T, the attention part is 5.7 ms vs 80.8 ms (14×). In fp32 + TF32 the gap is smaller: 2.4× at T = 1024 and 1.7× at T = 256.
3. **The manual path's memory grows with T, and it runs out of memory at T = 2048.** SDPA's extra memory for forward + backward stays flat at 295 MiB (bf16) or 485 MiB (fp32) at every T up to 16384. Manual uses 894, 1662 and 3199 MiB at T = 256, 512 and 1024, and runs out of memory under the 4.8 GB cap at T = 2048 for training and T = 4096 for the forward pass alone. bf16 autocast barely helps here (3199 MiB vs 3267 MiB in fp32), because autocast runs the softmax in fp32 and then casts the scores back to bf16 for `@ v`. The bf16-to-fp32 copy kernels are the manual path's top kernels. nanoGPT's mask buffer is minor by comparison: 4 MiB at the default `block_size = 1024`, 1 GiB at 16384.
4. **The default dispatch leaves 1.3–1.5× on the table at long T. cuDNN is faster from T = 1024 up.** Forced cuDNN vs the default mem-efficient kernel, bf16 forward + backward: 6% slower at T = 256 and 512, then 4%, 15%, 29%, 40% and 52% faster at T = 1024, 2048, 4096, 8192 and 16384 (67.9 ms vs 103.3 ms at 16k). The forward pass alone is 4–16% faster from T = 1024 up. The cost is a constant 48 MiB of extra memory (343 vs 295 MiB).
5. **`torch.compile` helps the manual path, but does not turn it into SDPA.** Inductor does not pattern-match nanoGPT's `masked_fill(tril buffer)` attention into SDPA. Instead it fuses mask, softmax and the cast into one Triton kernel (`triton_per_fused__softmax_...`). That makes manual 2.0–3.2× faster and cuts its memory 2.4–3.4× (943 MiB at T = 1024). Training now works at T = 2048, but compiled manual is still 1.4×, 1.6×, 1.9× and 3.2× slower than `sdpa_default` at T = 256 to 2048, and runs out of memory at T = 4096.
6. **The math fallback is no better than nanoGPT's manual code.** `sdpa_math` upcasts bf16 inputs to an fp32 softmax. It runs at 0.72–0.78× of manual's forward speed, 0.88–1.01× for forward + backward, and uses about the same memory. So the gain comes from the fused kernels, not from calling SDPA itself.
7. **Dropout 0.1 rules out cuDNN and costs the mem-efficient kernel 37–57%.** In this build, cuDNN rejects `dropout_p > 0` with "No available kernel", so nanoGPT configured with dropout always gets the mem-efficient kernel. That kernel's forward + backward goes from 15.6 to 21.4 ms at T = 1024 (+37%) and from 33.1 to 52.1 ms at T = 4096 (+57%). Its memory stays at 295 MiB, because the dropout mask is regenerated from the RNG instead of being stored. Manual pays +18% (107 ms) and compiled manual +7% (32 ms, but memory grows to 1158 MiB).
8. **Numerics are equivalent.** In bf16, every variant has relative error of 3.7–3.9e-3 on y and 4.5–5.0e-3 on dx and dW_qkv against fp64. This is bf16 rounding of the projections, and SDPA and cuDNN match the manual path. In fp32 + TF32, the mem-efficient kernel (3.6e-4 on y, 4.7e-4 on gradients) is slightly more accurate than manual or math (4.2e-4 and 5.6e-4).

Takeaway: on Windows PyTorch wheels, nanoGPT's `flash` flag gets you the CUTLASS mem-efficient kernel, not FlashAttention. That is still the right default: it is 5.8× faster than manual for bf16 training at T = 1024, and its memory stays flat while manual runs out at T = 2048. For bf16 runs with `dropout = 0` and T ≥ 1024, wrapping the forward pass in `sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION], set_priority=True)` gives another 1.04–1.5× at a cost of 48 MiB. With dropout on, or in fp32, there is nothing faster to switch to. Don't count on `torch.compile` to rescue the manual path: it closes about 80% of the gap to SDPA but still materializes T × T scores.

Full numbers: [results.md](results.md) / [results.json](results.json).
