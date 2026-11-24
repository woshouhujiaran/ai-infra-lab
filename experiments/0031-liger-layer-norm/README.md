# 0031 — Liger-Kernel LayerNorm (`LigerLayerNormFunction`)

- **Upstream**: [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) (submodule `upstream/Liger-Kernel`), imported unmodified from `src/`.
- **Component**: `src/liger_kernel/ops/layer_norm.py`. One Triton program per row (`BLOCK_SIZE = next_power_of_2(H)`, hard-fails above 65536); mean / rstd in fp32; affine fused; backward partitions rows across SMs into fp32 partial `dW` / `db` then `.sum(0)`. `@ensure_contiguous` copies every tensor arg. No in-place `dY→dX`, no casting modes (unlike Liger RMSNorm). Bias is required by the Triton kernel (`functional.layer_norm` invents `zeros_like(weight)` when `None`).
- **Hardware**: RTX 4060 Laptop 8 GB, torch 2.14.1+cu126, triton-windows, memory capped at 60% by `lab/gpu_guard.py`. Peak allocated ~1.5 GB.
- **Run**: `.venv\Scripts\python.exe experiments\0031-liger-layer-norm\run.py` (~2.5 min).

## Question

How does Liger's LayerNorm compare to `F.layer_norm` / `nn.LayerNorm` / `torch.compile` on numerics, strided layouts (the failure mode of the Triton tutorial in 0020), the 65536-column limit, SM-partitioned `dW`/`db` determinism, and memory / bandwidth — and is it worth using over cuDNN `F.layer_norm` on an Ada laptop GPU?

## Setup

- **Numerics**: T = 2048, H = 4096; fp32 / fp16 / bf16; with and without a real bias (zeros bias when off). Relative L2 of Y / dX / dW / db vs fp64 textbook LayerNorm and vs `F.layer_norm` / `nn.LayerNorm` / `torch.compile(F.layer_norm)`.
- **Layouts**: row-strided x (canary next to the view), transposed x, strided weight/bias, 3D `(B,S,H)`, and `y.sum().backward()` (stride-0 `dy`).
- **Edges**: H ∈ {65535, 65536, 65537}; freeze W / B via `requires_grad`; NaN / Inf rows; `LigerLayerNorm(..., bias=False)`; `functional.layer_norm(..., bias=None)`.
- **Determinism**: 8 repeated low-level backwards at T ∈ {1, 3, 23, 24, 25, 256, 4096} on 24 SMs; bit-identity of dX / dW / db and error vs `F.layer_norm`.
- **Memory / latency**: bf16 at GPT-2 / BERT (H = 768), Qwen3-0.6B (1024), Llama-3.2-1B (2048), T ∈ {2048, 8192, 16384}.
- **Bandwidth**: bf16, T = 4096, H from 512 to 16384 (incl. H = 2049 past a power of two). Forward counted as 2·T·H·2 bytes, backward as 3·T·H·2.

## Findings

1. **Numerics sit on the dtype floor and match `F.layer_norm`.**
   - bf16 Y / dX / dW ≈ 1.66e-3 relative to fp64 (equal to the rounding floor); liger vs F is 6–9e-6 on Y and similarly tiny on grads. fp16 ≈ 2.07e-4; fp32 ≈ 5e-8. `torch.compile` matches. Zero-bias path is fine (db abs max ≈ 0 when the bias leaf is a zeros tensor that still gets a grad).

2. **`@ensure_contiguous` closes the tutorial layernorm footguns.**
   - Row-strided x leaves the neighbouring canary untouched and matches contiguous (same Y / dX to numerical noise). Transposed x, strided W/B, 3D inputs, and stride-0 `dy` from `y.sum()` all stay finite and within ~1e-5 of `F.layer_norm`. Unlike 0020, there is no silent OOB write or NaN grad from these layouts — at the cost of an implicit copy on every non-contiguous arg.

3. **H = 65536 is the hard ceiling; Module `bias=False` still owns a grad-requiring zero bias.**
   - `calculate_settings(65537)` raises; 65535 / 65536 fwd+bwd succeed (bf16, T = 4). Freezing W and/or B via `needs_input_grad` correctly returns `None` grads. NaN / Inf rows produce NaN outputs like `F.layer_norm`. `LigerLayerNorm(bias=False)` still allocates a zero `bias` Parameter with `requires_grad=True` (so training will accumulate a useless `db`). `functional.layer_norm(..., bias=None)` is bit-identical to an explicit zeros bias.

4. **SM-partitioned `dW`/`db` reduction is bit-deterministic.**
   - Across 8 repeats at every tested T (including T ≪ 24 SMs, where most programs write zero partials into the `torch.empty` buffers), dX / dW / db are bit-identical. `db` matches `F.layer_norm` exactly; `dX`/`dW` drift at most ~2e-5 relative at T = 4096 (reduction-order noise), never a correctness failure.

5. **Memory ties `F.layer_norm`; end-to-end speed only helps at wider H.**
   - Held after forward and extra peak match cuDNN `F.layer_norm` (±0.4 MiB) and are 4–5× below an eager fp32-manual reference. At H = 768, Liger fwd+bwd is comparable to F once warm (0.75–1.47 vs 0.73–1.57 ms) and loses to `torch.compile` (~0.43–0.85 ms). At Llama H = 2048 × 16k tokens Liger is 2.59 vs F 3.00 vs compile 6.21 ms.

6. **Forward is at copy bandwidth; backward loses on narrow rows and wins past H ≈ 2048.**
   - Forward Liger: 146–227 GB/s on power-of-two H (1.0–1.72× F); H = 2049 → BLOCK 4096 drops to 92.6 GB/s (lane util 0.5) but still 1.53× F. Backward Liger: 0.47–0.86× F at H ≤ 1024 (SM reduction / launch overhead), 1.33–1.71× F at H ∈ {2048…16384}. `torch.compile` rarely beats F on this GPU for LayerNorm.

Takeaway: Liger LayerNorm is a safe, floor-accurate drop-in that fixes the stride hazards of the Triton tutorial via copies, is bit-deterministic on its SM-partitioned `dW`/`db` path, and caps H at 65536. On an RTX 4060 Laptop it does not beat cuDNN `F.layer_norm` end-to-end at BERT/GPT-2 width; the win shows up as ~1.3–1.7× backward (and matched forward) once H ≥ 2048. Prefer `F.layer_norm` / compile for narrow hidden sizes; use Liger when you already standardize on its kernels or need the contiguous-input guarantee without writing the copy yourself. Watch `bias=False` still exposing a trainable zero bias Parameter.

Full numbers: [results.md](results.md) / [results.json](results.json).
