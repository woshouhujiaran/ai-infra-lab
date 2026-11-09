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
