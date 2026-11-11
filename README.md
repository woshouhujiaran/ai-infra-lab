# ai-infra-lab

Small, locally reproducible experiments on individual AI-infra components (KV cache, schedulers, samplers, kernels, quantization, ...).

Rules:

- One experiment = one component of one upstream project, runnable on a single laptop (RTX 4060 Laptop 8 GB or CPU). No rented GPUs.
- Upstream projects are pinned as read-only git submodules under `upstream/` (push URL disabled). Nothing is pushed to upstream repositories.
- GPU experiments go through `lab/gpu_guard.py`, which refuses to run on a busy GPU and caps memory at 60%.
- `experiments/registry.json` lists every finished experiment; new experiments must not repeat an existing topic.

## Experiments

| ID | Upstream | Component | Device | Result |
|----|----------|-----------|--------|--------|
| [0001](experiments/0001-nanovllm-prefix-cache) | nano-vllm | BlockManager prefix cache | CPU | Burst arrival drops prefix-cache hit rate from 92.7% to 83.8% (2048-token shared prefix) |
| [0002](experiments/0002-nanovllm-sampler) | nano-vllm | Sampler (exponential race) | RTX 4060 | Exact like `torch.multinomial`; only `torch.compile` makes it 2.0× faster at batch 256 with 1/3 the temp memory of eager |
| [0003](experiments/0003-nanovllm-rmsnorm) | nano-vllm | RMSNorm fused residual-add | RTX 4060 | Fused matches fp32 ref (separate errs by 0.03125 abs); eager fused is slower; compile wins only at mid token counts |
| [0004](experiments/0004-nanovllm-rope) | nano-vllm | RotaryEmbedding cos/sin cache | RTX 4060 | fp32 table drifts with position (3.6% wrong bf16 outputs at 32k+); fp64-built table fixes it for free; cached beats on-the-fly once compiled |
| [0005](experiments/0005-nanovllm-chunked-prefill) | nano-vllm | Scheduler chunked prefill | CPU | Prefill-first chunking never shortens decode stalls (1.0 s → 1.5 s at budget 256); decode-first mixing cuts them to 44–155 ms at budgets 256–1024 for higher TTFT |
| [0006](experiments/0006-minbpe-vs-tiktoken) | minbpe | RegexTokenizer train/encode | CPU | Same ids as tiktoken on identical merges, but tiktoken encodes 12–29× faster (gap grows with vocab); a chunk cache recovers 2.5–3×; `train` crashes once the text runs out of pairs |
| [0007](experiments/0007-bnb-linear-quant) | bitsandbytes | Linear8bitLt / Linear4bit | RTX 4060 | NF4 cuts weights 3.8–3.9× and decodes 1.2–1.5× faster than fp16 at 9% error (FP4 12%); int8 prefill is 2.4–3.0× faster but needs threshold 6 to hold ~1% error with outlier activations; 16-token batches are 2–4× slower for all quantized layers |
| [0008](experiments/0008-streamingllm-attention-sink) | streaming-llm | StartRecentKVCache (attention sinks) | RTX 4060 | Past 2048 tokens dense ppl explodes to 6229 and window sits at 40–50; 4 sinks + position shift stay at 31–32 and match re-computation; without shift sinks blow up like dense; 1 sink is within 1.4% of 4; upstream's `torch.cat` eviction costs 12–23% of each step |
| [0009](experiments/0009-liger-fused-ce) | Liger-Kernel | CrossEntropy / FusedLinearCrossEntropy | RTX 4060 | Liger CE = `torch.compile` CE: 2–4.7× less memory and 1.3–1.7× faster than the HF fp32 upcast; only fused linear CE runs 16k tokens on 128k–152k vocabs (< 0.7 GB), but its default chunking is up to 19× slower on Qwen3's head — `chunk_mem_const=8` fixes most of it; bf16 dW accumulation costs +8–23% error |
| [0010](experiments/0010-triton-fused-softmax) | triton | Tutorial fused softmax kernel | RTX 4060 | Never beats `torch.softmax` as written (0.73–1.0×; torch already at copy bandwidth); per-row `num_warps` gives at most +4% at 2k–8k columns; rows ≥ 32k fp32 exceed the 99 KiB shared-memory limit; per-call `warmup()` makes small calls 4.6× slower than torch |
| [0011](experiments/0011-nanogpt-sdpa-attention) | nanoGPT | CausalSelfAttention SDPA vs manual | RTX 4060 | Windows torch has no FlashAttention, so the "flash" path runs the mem-efficient kernel: still 5.8× faster than manual for bf16 fwd+bwd at T=1024, with flat memory where manual OOMs at T=2048; forced cuDNN is 1.3–1.5× faster still at T ≥ 4096 but rejects dropout; `torch.compile` speeds manual up 2–3× but stays 1.4–3.2× behind SDPA |
| [0012](experiments/0012-picotron-column-parallel) | picotron | ColumnParallelLinear (TP via gloo) | CPU | Sync path matches `nn.Linear` (fp32 last-bit, any shape) but dim-0 shards are views of the full master, so every rank keeps and checkpoints the full 224 MiB weight at TP 2/4 on the CPU path (`.clone()` fixes it); async all-reduce crashes on 2D/4D inputs; bf16 dX error +19%/+40% at TP 2/4; one shared dX all-reduce for up+gate is 8–23% faster |
| [0013](experiments/0013-eplb-expert-balancer) | EPLB | `rebalance_experts` (hierarchical / global) | CPU | One redundant expert per GPU gets global EPLB to 1.03–1.05× max/mean GPU load (none: 1.26–3.78×; 1.004–1.017× with exact stats); hierarchical is stuck at 1.48–1.49× when group loads are uneven, but cuts nodes per token from 5.3 to 4.0 at 8 nodes; slightly stale stats (0.96 correlation) push skewed loads back to 1.39×; upstream packing takes 6 s at 256 GPUs, a heap version gives identical output 160× faster |
| [0014](experiments/0014-gptfast-speculative-decode) | gpt-fast | `speculative_decode` (draft + verify) | RTX 4060 | `generate(draft_model=...)` hits 3 bugs at HEAD, and the prefill-style verify mask is silently wrong under `torch.compile` past position 128 (logit err 2.7); fixed, the rejection sampler is exact and `(1-a^(k+1))/(1-a)` predicts 1.8–6.7 tokens per call within 3% in 21/24 cells, but launch-bound draft steps (55–67% of the target's cost) keep it at 0.66–0.98× plain decoding |
| [0015](experiments/0015-nanovllm-store-kvcache) | nano-vllm | `store_kvcache` Triton KV write | RTX 4060 | Bit-exact (strided V, -1 padding) and at the ~1 µs launch floor per decode call, 4.5–6.5× faster than `index_put_`, which also writes -1 padding into the last cache slot; fails to compile for non-power-of-two KV widths (5×64, 32×80, ...), and int32 `slot * D` past 2^31 elements per layer silently writes into the previous layer's cache; a masked int64 rewrite fixes both at no cost, and 4096-element tiles make D = 128 prefill 2.3× faster |
| [0016](experiments/0016-triton-tutorial-matmul) | triton | Tutorial matmul (autotune, grouped order, fp8) | RTX 4060 | fp16 at 0.91–1.18× cuBLAS, but 5 of 15 autotune configs exceed 99 KiB shared memory and every new batch size M costs a 1.2–17 s re-tune; row-major tile order (g = 1) is 1.7× slower once B exceeds L2 (DRAM-bound re-streaming), any g ≥ 2 fixes it; an fp16 accumulator matches fp8's 1.8× speedup over cuBLAS with ~10× less error than e4m3 inputs; the fp8 path accumulates inside the MMA (error 34× the floor at K = 64k, same as `_scaled_mm` fast accum), and only `acc += dot * scale` promotes it; int32 offsets crash past 2^31 elements of A, int64 fixes it for free |
| [0017](experiments/0017-gptfast-weight-only-quant) | gpt-fast | `quantize.py` weight-only int8 / int4 Linear | RTX 4060 | int4 fails at HEAD (torch now wants uint8-packed nibbles; the wrong nibble order runs silently with 140% error), and pad-to-1024 makes stories15M's int4 g32 bigger than bf16; once fixed, int8 is free end to end (ppl 2.066 vs 2.065) and int4 costs +1.0–1.5% ppl; at batch 1 under CUDA graphs int8 / int4 g128 are 2.0× / 3.7× faster than bf16 (all at ~250 GB/s), but compiled int8 is fused only at M = 1 and is 2.4× slower from M = 2, tinygemm int4 breaks even at M = 32 and is 4× slower at prefill, and torch's `_weight_int8pack_mm` scales linearly with M (140× bf16 at 2048) |
| [0018](experiments/0018-gptfast-moe-expert-gather) | gpt-fast | Mixtral `MOEFeedForward` per-token expert gather | RTX 4060 | Batch 1 only (B > 1 broadcasts / crashes); eager materializes T·A expert copies (672 MiB per token per Mixtral layer, OOM at 3 tokens); `torch.compile` fuses the gather away and is fastest at T ≤ 2–4 (DRAM bandwidth, 3× eager) but re-reads experts per token, so prefill is 14× slower than sort + `grouped_mm`, which on sm89 can't be CUDA-graph captured; dense bmm wins for OLMoE at 16–64 tokens; bf16 router logits tie and flip 0.2–2% of tokens' experts (bf16 softmax adds up to 1.5% more for flat routers) |

## Daily automation

A Windows scheduled task ("ai-infra-lab daily experiment", every day at 14:00) runs `scripts/daily_run.ps1`, which starts the Cursor CLI agent with [automation/daily_prompt.md](automation/daily_prompt.md). The agent picks a new topic from [experiments/BACKLOG.md](experiments/BACKLOG.md) that is not in the registry, runs it locally, commits step by step and pushes here. Logs go to `logs/` (not committed).

## Setup

```powershell
git clone --recurse-submodules <this repo>
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
