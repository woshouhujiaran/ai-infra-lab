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

## Daily automation

A Windows scheduled task ("ai-infra-lab daily experiment", every day at 14:00) runs `scripts/daily_run.ps1`, which starts the Cursor CLI agent with [automation/daily_prompt.md](automation/daily_prompt.md). The agent picks a new topic from [experiments/BACKLOG.md](experiments/BACKLOG.md) that is not in the registry, runs it locally, commits step by step and pushes here. Logs go to `logs/` (not committed).

## Setup

```powershell
git clone --recurse-submodules <this repo>
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
