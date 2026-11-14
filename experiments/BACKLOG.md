# Experiment backlog

Candidates for the daily run. Pick the first unchecked item that installs on Windows; check it off when done (or note why it was skipped).

- [x] nano-vllm `BlockManager`: prefix-cache hit rate vs prefix length / block size / arrival pattern (0001)
- [x] nano-vllm `Sampler`: exponential-race sampling vs `torch.multinomial` — distribution + throughput, eager vs compiled (0002)
- [x] nano-vllm `RMSNorm` with fused residual add vs separate ops — numerics (bf16) and speed (0003)
- [x] nano-vllm `RotaryEmbedding` cached cos/sin vs on-the-fly — error vs fp64 reference by position, speed (0004)
- [x] nano-vllm `Scheduler`: chunked prefill — time-to-first-token proxy vs `max_num_batched_tokens` (simulation) (0005)
- [x] karpathy/minbpe: BPE training/encoding throughput vs vocab size, compared with tiktoken (0006)
- [x] bitsandbytes: `Linear8bitLt` / `Linear4bit` (NF4) — weight memory, output error, latency vs fp16 Linear (0007)
- [x] mit-han-lab/streaming-llm: attention-sink KV eviction on a small model (≤ 500M) — perplexity vs cache size (0008)
- [x] linkedin/Liger-Kernel: fused cross-entropy (Triton, via `triton-windows`) — peak memory vs PyTorch CE (0009)
- [x] triton-lang/triton tutorial: fused softmax kernel vs `torch.softmax` bandwidth (via `triton-windows`) (0010)
- [x] karpathy/nanoGPT `CausalSelfAttention`: SDPA flash backend vs manual attention — memory/latency vs seq len (0011)
- [x] picotron-style tensor parallel `ColumnParallelLinear` simulated with gloo on CPU (2 processes) — correctness vs single-process (0012)
- [x] deepseek-ai/EPLB `rebalance_experts`: hierarchical vs global balance, redundancy, stale stats, runtime on synthetic DeepSeek-V3 routing (CPU) (0013)
- [x] meta-pytorch/gpt-fast `speculative_decode`: HEAD breakage, verify-mask correctness, rejection-sampling exactness, acceptance vs k and speedup with TinyStories 15M/42M drafts for 110M (0014)
- [x] nano-vllm `store_kvcache` (Triton KV-cache write): exactness with strided V / -1 padding, non-power-of-two D, int32 `slot * D` overflow, bandwidth vs torch scatter ops, eager vs CUDA graph (0015)
- [x] triton-lang/triton tutorial matmul: autotune configs vs 99 KiB shared memory, re-tune cost per M, `GROUP_SIZE_M` L2 effect, fp16-accumulate and fp8 speed / accumulation error, unaligned shapes, int32 offset overflow (0016)
- [x] meta-pytorch/gpt-fast `quantize.py` weight-only int8 / int4: HEAD breakage with current torch int4 packing, per-layer and whole-model error on stories110M, latency vs token count for bf16 / int8 eager / compiled / int4 tinygemm (0017)
- [x] meta-pytorch/gpt-fast `mixtral-moe` `MOEFeedForward` per-token expert-weight gather: memory / OOM point and latency vs token count for Mixtral-8x7B and OLMoE-1B-7B layer shapes, eager vs `torch.compile` vs per-expert loop / `grouped_mm` / dense, bf16 router softmax ties, batch > 1 shapes (0018)
- [x] bitsandbytes 8-bit optimizers (`Adam8bit` / `AdamW8bit` / `PagedAdamW8bit`): state memory and paged-memory accounting on GPT-2-124M, `step()` latency vs torch fused Adam, update error vs in-block gradient dynamic range vs linear / tensor-wise / sqrt(v) ablations, nanoGPT training loss (0019)
- [x] triton-lang/triton tutorial layer norm (`05-layer-norm.py`): input / grad layouts the wrapper silently mishandles (row-strided x, permuted x, stride-0 `dy` from `y.sum()`), 64 KB row limit, fp16/bf16 partial dW/db accumulation error vs token count, lock-based `GROUP_SIZE_M` reduction determinism and contention, fwd/bwd bandwidth vs torch / `torch.compile` / a lock-free fp32 rewrite (0020)
- [ ] linkedin/Liger-Kernel SwiGLU (`LigerSiLUMulFunction`, fused gate-up): in-place da/db writes into saved a/b (stashed activations, leaf inputs, shared saved tensors, `retain_graph`, checkpoint), bf16 error vs fp64 against eager / `torch.compile`, full-MLP memory and latency at Qwen3-0.6B / Llama-3.2-1B shapes, bandwidth vs intermediate size incl. non-power-of-two padding
