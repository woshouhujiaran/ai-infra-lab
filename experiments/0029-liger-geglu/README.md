# 0029 — Liger-Kernel GeGLU (`LigerGELUMulFunction`)

- **Upstream**: [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) (submodule `upstream/Liger-Kernel`, commit `1ab779f`), imported unmodified from `src/`.
- **Component**: `src/liger_kernel/ops/geglu.py`. `LigerGELUMulFunction` (used by `LigerGEGLUMLP` / Gemma 1–4 MLP patches) computes `gelu_tanh(a) * b` in one Triton kernel. It runs one program per row with `BLOCK_SIZE = next_power_of_2(I)`, saves `a` and `b`, recomputes the tanh-approx GELU in the backward pass and **writes da / db into the storage of the saved `a` / `b`**. Upstream only implements the tanh approximation (a TODO notes exact GELU); HF Gemma uses `gelu_pytorch_tanh`.
- **Hardware**: RTX 4060 Laptop 8 GB, torch 2.14.1+cu126, triton-windows 3.8.0, memory capped at 60% by `lab/gpu_guard.py`. Peak allocated memory was 2.7 GB.
- **Run**: `.venv\Scripts\python.exe experiments\0029-liger-geglu\run.py` (about 3 min).

## Question

Liger's GeGLU mirrors SwiGLU's in-place backward. Does it hit the same silent aliasing failures on Gemma-style graphs? How far is the tanh approximation from exact-erf GELU, and does Liger match `F.gelu(approximate='tanh')`? Unlike SwiGLU, does the more complex da path stay on the rounding floor? What memory and bandwidth look like at Gemma-2B / Gemma-2-2B shapes?

## Setup

- **Aliasing** (bf16, T = 256, H = 512, I = 1536): GemmaMLP-style graphs from `F.linear` on bf16 leaves, compared with fp64 eager `F.gelu(..., approximate='tanh')` autograd. Variants: eager tanh, upstream Liger, and `liger_oop` (upstream forward, out-of-place backward in `run.py`, bit-identical grads to upstream). Cases: plain MLP; gate/up outputs kept alive; leaf inputs; aux `0.5 * gate.pow(2).sum()` before / after the activation; `retain_graph=True` twice; non-reentrant checkpoint.
- **Numerics**: T = 2048, I = 8192, a = std · N(0,1) for std ∈ {1, 4}. Relative L2 of c / da / db against fp64 tanh-approx and against fp64 exact-erf GELU. Floor = fp64 result rounded once to the dtype. Also Liger vs same-dtype eager tanh, and a forward without the `.cast(b.dtype)` before the multiply (`fp32-mul fwd`).
- **Full MLP**: bf16 gate / up / down at Gemma-2B (2048 → 16384) and Gemma-2-2B (2304 → 9216), T ∈ {1024, 4096, 8192}. Variants: eager tanh, eager exact, `torch.compile` of tanh MLP, Liger, `liger_oop`.
- **Bandwidth**: bf16, T = 4096, I from 2048 to 16384 (incl. 4104 just past 4096 and Gemma widths 9216 / 16384). Bytes: 3·T·I·2 forward, 5·T·I·2 backward.

## Findings

1. **Same in-place backward traps as SwiGLU; `liger_oop` fixes all of them.**
   - Stashed gate/up activations: 99.96% of elements become da/db after backward; gradients themselves stay correct.
   - Leaf inputs are overwritten (`grad_aliases_input=True`).
   - Aux `gate.pow(2)` built *before* the activation: max grad relative error 1.002 (238× eager bf16). Built *after*: correct (autograd order).
   - Second `retain_graph=True` call: first call correct, second silently wrong (rel err 1.10, 245× eager). No guard (SwiGLU's fused gate-up has one; GeGLU has no fused variant).
   - Plain GemmaMLP graph and non-reentrant checkpoint are safe.

2. **Liger matches HF tanh GELU bit-for-bit on `c` / `db`, but neither sits on the bf16 floor; `torch.compile` does.**
   - Same-dtype vs `F.gelu(approximate='tanh')`: `c` and `db` relative error 0; `da` at ~1e-8 (bf16) / ~1e-6 (fp16).
   - Against fp64 tanh: in bf16, Liger and eager are both ~1.44× the floor on `c`/`db` and ~1.41× on `da`. `torch.compile` hits the floor on all three. Dropping the forward cast of `gelu(a)` to the input dtype (`fp32-mul fwd`) puts `c` on the floor at unchanged bandwidth.
   - Contrast with SwiGLU (0021): there Liger's da/db were at the floor because the silu path stayed in fp32; GeGLU's da formula mixes half-precision `dc`/`b` with the fp32 tanh terms, so it inherits eager's extra rounding.
   - Tanh vs exact-erf in fp64 is only ~2e-4 on `c` and ~4e-4 on `da` — below the bf16 floor (~1.7e-3), so in bf16 the approx choice is invisible next to rounding. In fp32 it dominates if you wanted exact GELU (upstream's open TODO).

3. **Memory: in-place saves one `[T, I]` at peak vs compile / oop; held-after-forward already matches compile.**
   - Held after forward (Gemma-2B, T = 8192): eager 1056 MiB (4 activations), Liger / compile / oop 800 MiB (recompute gelu).
   - Extra peak at the same point: eager 1632, compile / oop 1376, Liger 1120 (−31% vs eager, −19% vs compile). The Liger–oop gap is exactly one `[T, I]` (256 MiB here; 144 MiB at Gemma-2-2B × 8192).
   - Whole-MLP time is GEMM-bound: Liger is 3.4–3.8% faster than eager tanh and within 0.6% of `torch.compile`. Exact-erf eager is within noise of tanh eager.

4. **Bandwidth: copy-speed for every I; power-of-two padding is free; I = 16384 backward spills 12 bytes.**
   - Forward 210–230 GB/s (1.40–1.66× eager), backward 223–229 GB/s (1.66–1.79× eager), tied with `torch.compile`. Exact-erf eager matches tanh eager bandwidth.
   - I = 4104 (50% of an 8192 block) is the only soft spot (~6% below copy on forward). Gemma widths 9216 / 16384 run at copy speed.
   - Out-of-place backward is within 3% of in-place (memory, not speed).
   - No spills until I = 16384, where the backward uses 128 regs/thread and spills 12 bytes (oop: 8). Forward stays spill-free at 94 regs.

Takeaway: Liger GeGLU is the Gemma twin of SwiGLU — copy-bandwidth, compile-tied whole-MLP speed, and the same silent in-place aliasing (stash / leaves / aux-before / second `retain_graph`). Numerically it tracks HF `gelu_pytorch_tanh` exactly, but unlike SwiGLU its da/db stay at eager's ~1.4× bf16 floor while `torch.compile` reaches the floor; the tanh–exact gap only matters in fp32. An out-of-place backward (same speed, one extra `[T, I]`) closes every aliasing hole, and dropping the forward dtype cast puts `c` on the floor for free.

Full numbers: [results.md](results.md) / [results.json](results.json).
