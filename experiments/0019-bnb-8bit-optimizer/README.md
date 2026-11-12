# 0019 — bitsandbytes 8-bit blockwise Adam / AdamW optimizer states

- **Upstream**: [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) (submodule `upstream/bitsandbytes` at tag `0.50.2`, the installed version). The training harness uses nanoGPT's `model.py` from `upstream/nanoGPT`.
- **Component**: `bitsandbytes/optim/optimizer.py` (`Optimizer2State`, `Optimizer8bit.step`) and the CUDA kernel `kOptimizerStatic8bit2StateBlockwise` in `csrc/kernels.cu`. Adam's m and v are stored as uint8 codes into `create_dynamic_map` code books, with one fp32 absmax per 256 elements. Each step dequantizes them, updates the parameter with fp32 m / v, then re-quantizes them to the nearest code.
- **Hardware**: RTX 4060 Laptop 8 GB, torch 2.14.1+cu126, memory capped at 60% by `lab/gpu_guard.py`.
- **Run**: `.venv\Scripts\python.exe experiments\0019-bnb-8bit-optimizer\run.py` (about 11 min; peak torch-allocated GPU memory 1983 MiB, on the GPT-2-124M shapes).

## Question

How much memory do bnb's 8-bit optimizer states save on a real parameter list, and where does that memory live for the paged variant? Is `step()` faster or slower than fused torch Adam? When does the blockwise dynamic quantization of m and v actually change the update, and why? Does it matter for a small real training run, and what would a cheap fix look like?

## Setup

- **Code books**: `create_dynamic_map(signed=True)` for m and `(signed=False)` for v.
- **Reference implementation**: the same step re-implemented in torch: fp32 update, nearest code via `searchsorted`, and the kernel's "keep the sign of m" code nudge. It was checked against the CUDA kernel over 50 steps on 1,048,676 elements, which is not a multiple of 256.
  - **Teacher-forced check**: each step starts from bnb's own codes.
  - **Free-running check**: the reference keeps its own state.
- **Ablations**, all built on the reference:
  - tensor-wise dynamic (one absmax for the whole tensor);
  - blockwise / tensor-wise linear int8 (`round(x / absmax * 127)` for m, `* 255` for v);
  - **sqrt(v) fix**: same blockwise dynamic map, but stores `sqrt(v)` instead of `v`.
- **weight_decay semantics**: bnb `Adam` / `AdamW` / `Adam8bit` with `weight_decay=0.1` vs `torch.optim.Adam` (L2) and `AdamW` (decoupled), 20 steps on the same gradients.
- **Memory and step time on GPT-2-124M** (`nanoGPT GPTConfig()`: 148 tensors, 124.5 M params):
  - state bytes;
  - the `torch.cuda.memory_allocated` delta;
  - the `nvidia-smi` delta, before and after deleting the optimizer;
  - `torch.save(state_dict)` size;
  - wall-clock `step()` time, also with bnb's per-parameter `sync_gpu` patched to a no-op from the experiment script (bitwise-identical parameters checked).
- **Latency vs size**: one tensor of 2^18–2^26 elements with fp32 and bf16 parameters, 20 steps after 5 warmup. "Effective GB/s" counts parameter read+write, gradient read and state read+write.
- **Gradient dynamic range sweep**: synthetic gradient streams `g_t = 10^(-r·u) · (mu + noise_t)`, with `u ~ U(0,1)` per element, so every 256-element block spans r decades. Also `mu ~ N(0, 0.25)` fixed and `noise ~ N(0, 1)`. 1 M elements, 300 steps, betas (0.9, 0.999). The last 50 per-step updates are compared with fp32 Adam on the same stream.
- **Training**: byte-level nanoGPT, 6 layers × 384, block 256, 10.7 M params, on the first 8 MB of torch's own `.py` sources (local, no download).
  - 1200 steps × 32 × 256 tokens, bf16 autocast, warmup 100, cosine to 10%, betas (0.9, 0.95), wd 0.1 on 2-D params, grad clip 1.0.
  - Peak lr 1e-3 and 3e-3, the same init and batch order for seed 0, plus a torch seed-1 run as the noise floor.
  - At the end, the fp32 run's `exp_avg_sq` is scanned for entries that 8-bit would round to code 0 (below 1.63e-7 × block absmax).

## Findings

1. **The torch re-implementation reproduces the kernel.** In teacher-forced steps only 1.3e-6 of m codes and 1.5e-5 of v codes differ, always by one code, from fp32 rounding at bin edges. Parameters agree to 4.5e-5 of the update size. After 50 free-running steps the drift is 6.8e-5 of the distance travelled, so the ablations below isolate the quantizer, not implementation noise.
2. **The code books are not symmetric.** The signed map has 1.0 but no −1 (its minimum is −0.993), so a block whose absmax comes from a negative m stores it 0.7% short. Both maps put 129 of their 256 codes above 0.1 and 30 below 1e-3. The smallest non-zero codes are 5.5e-7 (m) and 3.25e-7 (v) of the block absmax.
3. **`bnb.optim.Adam(weight_decay=...)` is AdamW, not Adam.** All bnb 2-state kernels apply decay as `p *= 1 - lr·wd` after the update. With wd 0.1, bnb `Adam` 32-bit lands 0.1% from `torch.optim.AdamW` and 11% from `torch.optim.Adam`, identical to bnb `AdamW`. Swapping `torch.optim.Adam` for `bnb.optim.Adam8bit` therefore silently changes L2 regularization into decoupled decay.
4. **Memory: 8.0 → 2.04 bytes per parameter on GPT-2-124M** (950 → 242 MiB, checkpoint 950 → 242 MiB).
   - The 98 tensors under `min_8bit_size=4096` (LayerNorms and biases, 121 k elements) keep fp32 states and cost nothing.
   - **Paged states are invisible to PyTorch.** `PagedAdamW8bit` puts 237 MiB in CUDA managed memory (`torch.frombuffer` over `cudaMallocManaged`, so torch sees CPU tensors). `memory_allocated` grows by only 4.6 MiB, `nvidia-smi` by 338 MiB, and `set_per_process_memory_fraction` (our 60% cap) does not apply.
   - **Paged memory leaks.** After `del optimizer` the 334 MiB stays allocated, because `GlobalPageManager.paged_tensors` still holds all 100 buffers.
5. **`step()` is slowed down by a `torch.cuda.synchronize()` after every parameter.** On GPT-2's 148 tensors:
   - bnb AdamW 32-bit takes 38.0 ms per step vs 20.1 ms for torch fused AdamW;
   - AdamW8bit takes 25.3 ms;
   - with `sync_gpu` patched out they drop to 20.2 ms and 14.6 ms, with bitwise-identical parameters;
   - so 8-bit is only faster than torch fused once the sync is gone.
6. **The 8-bit kernel is not bandwidth-bound.** It reaches 80–140 GB/s of traffic vs 200–235 GB/s for fp32 Adam (the ~250 GB/s DRAM limit). It moves 16 instead of 28 bytes per element, but the per-element binary search over the code book costs more than the saved traffic.
   - With fp32 params, 8-bit Adam runs at 0.84× torch fused speed at 4 M elements, 0.68× at 16 M and 1.05× at 64 M; an earlier shorter run measured 1.98 ms vs 2.01 ms at 16 M, so mid-size points vary by up to ~1.7× between runs on this laptop.
   - With bf16 params, the 8-bit kernel is 1.8× slower than torch fused bf16 Adam at 16–64 M (90 vs 230 GB/s). torch then keeps bf16 states, so it moves fewer bytes.
   - Paged and non-paged run at the same speed once warm, because the states fit in VRAM.
7. **8-bit v underflows once gradients within a 256-element block span ~3 decades.** v ∝ g² needs twice the decades m does, but both use the same 7-decade code book.
   - Once a v entry rounds to code 0, its history is gone and the next denominator is just `sqrt(1-β2)·|g_t|`. Meanwhile m still remembers past gradients.
   - Whenever the current gradient is near 0, the update explodes. The largest update grows from 1.5·lr (fp32) to 3.7e4·lr at a 3-decade span and 4.7e4–5.5e4·lr at 4–6 decades.
   - In numbers: at 2 decades bnb's update error is 11%, with no v underflow. At 3 decades it is 2,200%, with only 0.09% of v at code 0. At 4 / 5 / 6 decades 19% / 35% / 45% of v is code 0, and the smallest-gradient decile is moved 27–69× too far.
   - Tensor-wise dynamic quantization is 1.3–7.3× worse than blockwise. Linear int8 is the most precise at 0 decades (2.2% vs 3.9% error) but explodes already at 2 decades (42% of v at 0, updates up to 3.4e6·lr). Its v floor is 1/510 of the absmax, and m survives longer than v.
8. **Storing sqrt(v) instead of v removes the explosion** at the same 8 bits and block size: 0 v underflow at every range, update error 9.6% / 13% / 20% / 29% / 46% at 2–6 decades, and the largest update 2.0–7.8·lr.
9. **Real training stays clear of that regime, and bnb 8-bit matches fp32.**
   - In the fp32 nanoGPT run the median block of sqrt(v) spans 0.68 decades and the 99th percentile 1.45. Only 5.5e-5 of v entries would round to 0 (bnb's actual code-0 fraction: 4.7e-5).
   - At lr 3e-3 the 99th percentile widens to 4.6 decades and 9.0e-4 would round to 0 (bnb: 2.0e-4). A few blocks reach the unstable range, but too few to matter here.
   - Final val loss: torch AdamW 1.0125, torch seed 1 1.0204, bnb AdamW 32-bit 1.0109, **bnb AdamW8bit 1.0194** (within seed noise), sqrt(v) reference 1.0134.
   - At lr 3e-3: torch 1.0609, bnb 8-bit 1.0509, sqrt(v) 1.0628.
   - Blockwise linear int8 states diverge at both lrs (val loss 11.8 / 8.9, hundreds of loss spikes). The dynamic code book is what makes 8-bit Adam work at all.

## Takeaway

On this laptop, `AdamW8bit` is a safe 4× cut of optimizer memory: training loss equals fp32 within seed noise, and the checkpoint shrinks by the same 4×. Its real risks are elsewhere:

- `Adam(weight_decay)` silently becomes AdamW.
- The per-parameter `synchronize()` makes `step()` slower than torch fused for many-tensor models.
- Paged states escape PyTorch's memory accounting and leak after the optimizer is deleted.
- v underflows, with unbounded updates, once a 256-element block's gradients span about 3 decades. Storing sqrt(v) is a drop-in fix in a pure-torch simulation; the CUDA kernel itself was not changed.

Full numbers: [results.md](results.md) / [results.json](results.json).
