# 0018 — gpt-fast Mixtral MoE per-token expert-weight gather

- **Upstream**: [meta-pytorch/gpt-fast](https://github.com/meta-pytorch/gpt-fast) (submodule `upstream/gpt-fast` at `6ecad9b`). `mixtral-moe/model.py` is imported unmodified.
- **Component**: `MOEFeedForward` + `ConditionalFeedForward` in `mixtral-moe/model.py`.
  - Router: a bf16 `nn.Linear(dim, E)`, then `softmax` in bf16, `topk(A)`, and a renormalization.
  - Expert FFN: every token gathers its own copies of its experts' weights, `w1[expert_indices]` → `[T, A, I, D]` (same for w3 / w2), and runs three einsums over them. The output is returned as `[T, D]` without being reshaped back.
- **Shapes**: one MoE layer with synthetic bf16 weights (w1 / w3 ~ N(0, 1/D), w2 ~ N(0, 1/I), x ~ N(0, 1)):
  - **Mixtral-8x7B**: D 4096, I 14336, 8 experts, top-2; 2.62 GiB per layer, 336 MiB per expert.
  - **OLMoE-1B-7B**: D 2048, I 1024, 64 experts, top-8; 0.75 GiB per layer, 12 MiB per expert.
- **Hardware**: RTX 4060 Laptop 8 GB (through `lab/gpu_guard`, capped at 60% = 4.8 GiB, so OOMs are real), i9-13900H, torch 2.14.1+cu126, triton-windows.
- **Run**: `.venv\Scripts\python.exe experiments\0018-gptfast-moe-expert-gather\run.py`. No downloads are needed. The run takes about 3–4 minutes, including about 20 s per `torch.compile` shape when the Inductor cache is cold.

## Question

gpt-fast's MoE layer avoids any routing bookkeeping by gathering every token's expert weights. What does that cost in memory and time as the token count T grows, eagerly and under `torch.compile` (static shapes as in decode, `dynamic=True` as with `--compile_prefill`)? Where do the usual alternatives (a per-expert loop, sort + `grouped_mm`, or dense all-expert bmm) win? Does the bf16 router change which experts are picked?

## Setup

1. **Correctness**:
   - Compare upstream eager, `torch.compile` and three rewrites against an fp32 per-expert reference that uses the *same* routing:
     - **loop**: HF-Mixtral style, one GEMM chain per expert over its tokens via `torch.where` + `index_add_`.
     - **grouped**: sort the T·A (token, expert) pairs and run three `F.grouped_mm` calls with device-side offsets.
     - **dense**: every expert on every token with `bmm`, then a dense combine.
   - Try to CUDA-graph capture each variant.
   - Run a tiny full `Transformer` (dim 64, 4 experts) at batch 2 against two batch-1 calls.
2. **Routing**:
   - Data: 65,536 tokens through a synthetic bf16 router whose logits have standard deviation 0.5–4.
   - Compare three ways of choosing the top-A set: upstream (bf16 logits, bf16 softmax), HF-style (bf16 logits, fp32 softmax), and fp32 logits. Each is checked against fp64 routing of the same bf16 inputs.
   - Metrics: set mismatch, exact ties between the A-th and (A+1)-th probability, and combine-weight error.
   - Output impact: the layer output with upstream routing vs fp64 routing, computed with the same bf16 FFN arithmetic, against an fp32 FFN with fp64 routing.
3. **Memory and speed**:
   - Grid: T = 1 … 1024 (Mixtral) and 1 … 4096 (OLMoE).
   - Memory: extra peak allocation per forward, with OOM under the cap.
   - Speed: eager launch loop (wall time including host syncs) and CUDA-graph replay (GPU time, what gpt-fast's `reduce-overhead` decode gets).
   - Compiled variants use gpt-fast's Inductor flags (`coordinate_descent_tuning`). Their generated code is checked for extern (cuBLAS) calls.
   - Bandwidth is measured against the bytes of the experts actually touched.

## Findings

1. **All variants compute the same thing; the compiled one is 2× more accurate.**
   - With identical routing, upstream eager, grouped and dense are all at 4.2–4.3e-3 relative error against fp32. The loop is at 4.7–5.6e-3, because `index_add_` accumulates in bf16.
   - `torch.compile` is at 2.2e-3. Inductor keeps `silu(x·w1) * (x·w3)` in fp32 inside its fused kernels, instead of rounding both intermediates to bf16.
2. **Only batch size 1 works.**
   - `MOEFeedForward` returns `[B·S, D]` and `TransformerBlock` adds it to `h` of shape `[B, S, D]`.
   - Batched decode (B = 2, S = 1) silently broadcasts the first layer's output to `[2, 2, D]`. It then crashes in the next layer's RoPE `view`.
   - Batched prefill (B = 2, S = 4) fails at the residual add.
   - B = 1 works for both, which matches the README's benchmarks: all at batch size 1, with a 5-token prompt.
3. **Eagerly, the gather materializes T·A full expert copies.**
   - That is exactly 672 MiB per token for a Mixtral layer and 96 MiB for OLMoE. Under the 4.8 GiB cap it hits OOM at T = 3 (Mixtral, on top of the 2.62 GiB layer) and T = 64 (OLMoE).
   - It is also slow: writing and re-reading the copies gives 79 GB/s effective, 3.2× slower than compiled at T = 1 (Mixtral 8.9 vs 2.8 ms).
   - `generate.py --compile` compiles only the decode step; prefill stays eager unless `--compile_prefill` is passed. Extrapolating from these numbers: full-size Mixtral with 8-way TP (the intermediate dimension is sharded) needs about 84 MiB per prompt token per GPU just for the transient copies. A 1000-token prompt would then need about 82 GiB, more than an A100-80GB has.
4. **`torch.compile` fuses the gather away, but every token re-reads its experts.**
   - Compiled, the extra memory is ≤ 32 MiB at every T. The generated code's only extern call is one `mm` for the router (none at T = 1). The three expert einsums are Triton kernels that read each token's experts in place.
   - At T = 1 this runs at DRAM bandwidth: 252 GB/s on Mixtral (2.79 ms), and 412 µs on OLMoE (3.0× eager). It is the fastest variant there.
   - There is no reuse across tokens, though, so time grows with T·A rather than with the number of distinct experts touched. Mixtral takes 5.6 / 11.1 / 22 / 44 / 178 ms at T = 2 / 4 / 8 / 16 / 64, all at about 253 GB/s of re-reads.
   - Static and `dynamic=True` compiles give the same times.
   - So `--compile_prefill` makes prefill memory-safe but slow. A 64-token Mixtral prefill is 14× slower than grouped (178 vs 12.4 ms), and a 256-token OLMoE prefill is also 14× slower (102 vs 7.1 ms).
5. **Which variant is fastest depends on T, and the crossover comes early.**

   | layer | T | fastest | runner-up |
   |---|---|---|---|
   | Mixtral | 1–2 | compiled gather (2.8 / 5.6 ms) | grouped (3.2 / 6.0 ms) |
   | Mixtral | 3–8 | grouped (6.1–10.5 ms, 230–236 GB/s of touched experts) | loop / dense (7.1–11.4 ms) |
   | Mixtral | 16–64 | dense under CUDA graph (11.5–12.0 ms, flat) | grouped (12.0–12.6 ms) |
   | Mixtral | 256–1024 | grouped (14.5 / 31.4 ms) | loop (15.8 / 33.1 ms) |
   | OLMoE | 1–4 | compiled gather (0.41 / 0.82 / 1.6 ms) | grouped (1.5–2.2 ms) |
   | OLMoE | 8 | tie: compiled gather 3.15, dense 3.30, grouped 3.1–4.5 ms (run to run) | |
   | OLMoE | 16–64 | dense under CUDA graph (3.35–3.66 ms) | grouped (4.3–6.7 ms) |
   | OLMoE | 256–4096 | grouped (7.1 / 7.9 / 22 ms) | dense (7.8 ms at 256; then 4–6× slower) |

   - **grouped**: on this sm89 GPU, `F.grouped_mm` takes a fallback that copies the group offsets to the host. It syncs on every call and cannot be CUDA-graph captured ("Cannot copy between CPU and CUDA tensors during CUDA graph capture").
   - **dense**: graph-capturable and flat until compute dominates. It reads all E experts on every call, so it is 4× (Mixtral) or 8× (OLMoE) slower than the compiled gather at T = 1, and does E/A times the FLOPs at large T.
   - **loop**: the HF loop syncs once per expert. On 64-expert OLMoE it takes 7–8 ms even at T = 1, 18–20× the compiled gather.
6. **The bf16 router flips the expert choice for 0.2–3.6% of tokens, mostly because bf16 logits tie.**
   - The bf16 gate output ties at the top-A boundary for 0.37% (Mixtral) and 4.3% (OLMoE) of tokens. About half of those tied tokens pick a different expert set than fp64 routing does (0.18% / 2.2%), even with HF's fp32 softmax.
   - Upstream's bf16 softmax adds ties only when the router is flat:
     - Mixtral, logit std 0.5: 1.41% ties and 0.72% mismatches; 0.54% of tokens differ from HF.
     - OLMoE, logit std 0.5: 6.9% ties and 3.6% mismatches; 1.5% differ from HF.
     - From logit std 2 on, upstream and HF differ on at most 0.04% of tokens.
   - A flipped token's output is effectively a different expert's: 62% (Mixtral) or 26% (OLMoE) relative difference. Those few tokens raise the layer's relative error from 0.50% to 5.3% (7 of 1024 tokens) and from 0.52% to 4.1% (95 of 4096 tokens).
   - Only fp32 logits (an fp32 gate) remove the ties. Which expert is "right" at an exact tie is arbitrary, though: what matters for a real checkpoint is matching the routing it was trained and evaluated with.

Takeaway:
- gpt-fast's gather is a good *decode* kernel once compiled: no bookkeeping, CUDA-graph friendly, at DRAM bandwidth, and the fastest variant while T·A is well below E (T ≤ 2 for Mixtral, T ≤ 4 for OLMoE).
- Run eagerly, as gpt-fast's default prefill does, it materializes 672 MiB per token per Mixtral layer. Compiled, it re-reads the experts for every token, which is 14× slower than a grouped GEMM at prefill sizes.
- It also only supports batch size 1.
- A practical MoE layer would keep the compiled gather for single-stream decode, and switch to sort + grouped GEMM for prefill or larger batches. Dense bmm is a graph-friendly middle ground for many small experts.
- Grouped GEMM on consumer Ada currently can't be graph-captured.

Repeatability:
- Parts 1–2 gave identical numbers in all runs (the first two runs had a run.py bug in the batch check, and a memory leak between configs that caused a spurious OOM; both are fixed).
- CUDA-graph timings agree within about 1% between the two complete runs.
- Eager paths dominated by host syncs (loop, and grouped at small OLMoE T) vary by up to 1.8× between runs (OLMoE loop at T = 2: 7.7 vs 13.1 ms). Their rankings above hold in both runs.

Full numbers: [results.md](results.md) / [results.json](results.json).
