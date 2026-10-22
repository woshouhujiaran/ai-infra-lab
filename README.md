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

## Setup

```powershell
git clone --recurse-submodules <this repo>
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
