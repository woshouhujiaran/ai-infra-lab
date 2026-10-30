# Experiment backlog

Candidates for the daily run. Pick the first unchecked item that installs on Windows; check it off when done (or note why it was skipped).

- [x] nano-vllm `BlockManager`: prefix-cache hit rate vs prefix length / block size / arrival pattern (0001)
- [x] nano-vllm `Sampler`: exponential-race sampling vs `torch.multinomial` — distribution + throughput, eager vs compiled (0002)
- [x] nano-vllm `RMSNorm` with fused residual add vs separate ops — numerics (bf16) and speed (0003)
- [x] nano-vllm `RotaryEmbedding` cached cos/sin vs on-the-fly — error vs fp64 reference by position, speed (0004)
- [x] nano-vllm `Scheduler`: chunked prefill — time-to-first-token proxy vs `max_num_batched_tokens` (simulation) (0005)
- [x] karpathy/minbpe: BPE training/encoding throughput vs vocab size, compared with tiktoken (0006)
- [ ] bitsandbytes: `Linear8bitLt` / `Linear4bit` (NF4) — weight memory, output error, latency vs fp16 Linear
- [ ] mit-han-lab/streaming-llm: attention-sink KV eviction on a small model (≤ 500M) — perplexity vs cache size
- [ ] linkedin/Liger-Kernel: fused cross-entropy (Triton, via `triton-windows`) — peak memory vs PyTorch CE
- [ ] triton-lang/triton tutorial: fused softmax kernel vs `torch.softmax` bandwidth (via `triton-windows`)
- [ ] karpathy/nanoGPT `CausalSelfAttention`: SDPA flash backend vs manual attention — memory/latency vs seq len
- [ ] picotron-style tensor parallel `ColumnParallelLinear` simulated with gloo on CPU (2 processes) — correctness vs single-process
