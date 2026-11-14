# 0021 — Liger-Kernel SwiGLU (`LigerSiLUMulFunction`, fused gate-up)

- **Upstream**: [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) (submodule `upstream/Liger-Kernel`, commit `1ab779f`), imported unmodified from `src/`.
- **Component**: `src/liger_kernel/ops/swiglu.py`. `LigerSiLUMulFunction` (used by `LigerSwiGLUMLP`, the Phi3 / Qwen3-MoE / Falcon-H1 MLP patches and `LigerTiledSwiGLUMLP`) computes `silu(a) * b` in one Triton kernel. It runs one program per row with `BLOCK_SIZE = next_power_of_2(I)`, saves `a` and `b`, recomputes silu in the backward pass and **writes da / db into the storage of the saved `a` / `b`**. `LigerFusedGateUpSiLUMulFunction` does the same on a single `[T, 2I]` gate_up tensor and only overwrites it when `in_place=True`.
- **Hardware**: RTX 4060 Laptop 8 GB, torch 2.14.1+cu126, triton-windows 3.8.0, memory capped at 60% by `lab/gpu_guard.py`. Peak allocated memory was 3.6 GB.
- **Run**: `.venv\Scripts\python.exe experiments\0021-liger-swiglu\run.py` (about 2 min).

## Question

Liger's SwiGLU saves memory by recomputing silu and reusing the saved activations as gradient buffers. Those writes happen inside a Triton kernel, so autograd's version counter never sees them. When does that silently corrupt tensors or gradients in real training code? How does the kernel's bf16 rounding compare with eager PyTorch and `torch.compile`? What do the memory savings and speedups look like for a whole MLP at small-model shapes? Does padding each row to the next power of two cost bandwidth?

## Setup

- **Aliasing** (bf16, T = 256, H = 512, I = 1536): MLP-style graphs built from `F.linear` on bf16 leaves. Each one is compared with the same graph in fp64 eager autograd, using a random-weighted sum as the loss. Every case runs with eager `F.silu(a) * b`, upstream Liger, and `liger_oop`, which keeps upstream's forward kernel but writes da / db to fresh buffers (`_swiglu_bwd_oop_kernel` in `run.py`, bit-identical to upstream). The cases are:
  - the plain HF LlamaMLP graph;
  - the same graph with the gate / up outputs kept alive, as a forward hook or activation logger would keep them;
  - `act(a, b)` called directly on leaf tensors;
  - an auxiliary loss `0.5 * gate.pow(2).sum()` built before or after the activation (`pow` saves `gate` for its own backward pass);
  - `autograd.grad` called twice with `retain_graph=True`;
  - non-reentrant activation checkpointing;
  - a fused `gate_up` Linear (Phi3 / Megatron layout), either through `chunk` + `LigerSiLUMulFunction` or through the fused kernel with `in_place` False or True.
  For stashed tensors, the metric is the fraction of elements that backward changed.
- **Numerics**: T = 2048, I = 8192, with a = std · N(0,1) for std ∈ {1, 4} and b, dc ~ N(0,1). The error metric is relative L2 error of c, da and db against fp64 autograd on the same rounded inputs. `floor` is the fp64 result rounded once to the dtype. `fp32-mul fwd` is upstream's forward without the `.cast(b.dtype)` before the multiply (`_swiglu_fwd_fp32_kernel`). The gate / down multipliers (0.5 / 2.0) go through `LigerSiLUMulFunction.apply(a, b, gate, down)`.
- **Full MLP**: bf16 gate / up / down Linear layers at Qwen3-0.6B (1024 → 3072) and Llama-3.2-1B (2048 → 8192), T ∈ {2048, 8192, 16384}. Each measurement is one `autograd.grad` with respect to x and all weights. "Held" is memory still allocated after forward, including the output. "Extra peak" is the maximum allocated during forward + backward above the inputs, including weight gradients. Variants:
  - `eager`: the HF LlamaMLP;
  - `eager_chunk`: a fused `gate_up` + `chunk`;
  - `torch_compile` of the HF MLP;
  - `liger`: `LigerSwiGLUMLP`;
  - `liger_chunk`: `LigerPhi3SwiGLUMLP`;
  - `liger_fused` / `liger_fused_inplace`: the fused gate-up kernel;
  - `liger_oop`.
- **Bandwidth**: bf16, T = 4096, I from 2048 to 18944. This includes I = 4104, just past 4096, and Qwen2.5-7B's 18944, which pads to 32768. Bytes are counted as 3·T·I·2 for forward and 5·T·I·2 for backward, divided by the `do_bench` median, taking the median of 3 rounds. Liger is timed at the op level (`swiglu_forward` / `swiglu_backward`). Eager and `torch.compile` backward are timed as `autograd.grad` with `retain_graph=True`.

## Findings

1. **The in-place backward silently corrupts anything else that holds the saved `a` / `b`.** Eager autograd would either be correct or raise a version-counter error. `liger_oop` was correct in every case below.
   - **Stashed activations**: if a forward hook or logger keeps the gate / up outputs, then after backward 99.96% of their elements hold da / db instead of activations. The gradients themselves are correct.
   - **Leaf inputs**: calling the op on leaf tensors overwrites the leaves with their own gradients (99.9% of elements changed), and `a.grad` shares storage with `a`.
   - **Another op saved the same tensor**: an auxiliary `0.5 * gate.pow(2).sum()` built *before* the activation gives gradients with a relative error of 1.0, 235× the bf16 eager error. Autograd runs the Liger node first, because it has the higher sequence number, so `PowBackward` then reads gradients instead of activations. Built *after* the activation, the same loss is correct only because autograd happens to run the aux branch first.
   - **`retain_graph=True`**: the first `autograd.grad` is correct and the second one returns gradients with a relative error of 1.15 (dX and dW of gate / up), with no warning. The fused gate-up kernel guards against this: with `in_place=True` it raises `RuntimeError` on the second call. `LigerSiLUMulFunction` has no such guard.
   - **The fused gate-up kernel with `in_place=True`** overwrites a stashed `gate_up` output in the same way (100% changed). With `in_place=False` it is safe.
   - **Safe cases**: the plain HF / `LigerSwiGLUMLP` graph (a Linear doesn't save its output), non-reentrant checkpointing (the saved tensors are recomputed copies) and the Phi3 `chunk` path (`ensure_contiguous` makes private copies) are all correct.
2. **The backward pass is exactly at the rounding floor, but the forward pass rounds twice, like eager.**
   - In bf16 and fp16, Liger's da / db equal the floor, because they are computed in fp32 and rounded once. Eager is at 1.29–1.45× the floor.
   - Liger's forward `c` is at 1.29–1.45× the floor, identical to eager, because silu(a) is cast to the input dtype before the multiply.
   - `torch.compile` is at the floor for c, da and db.
   - Dropping that one cast (`fp32-mul fwd`) puts `c` on the floor at unchanged bandwidth.
   - With gate / down multipliers, da / db stay on the floor. `c` is at 1.42–1.44× the floor, because `c * down_multiplier` is a separate bf16 op.
   - In fp32 every variant is within 1.9–2.4× the floor, except da at gate std 4 (5.2–5.4× for all variants, eager and `torch.compile` included).
3. **Memory: Liger saves two `[T, I]` buffers against eager at peak, one of them through the in-place trick.**
   - After forward, eager holds 4 `[T, I]` tensors (gate, silu(gate), up, product) and every other variant holds 3. That includes `torch.compile`, which recomputes silu.
   - Extra peak memory at Llama-3.2-1B with T = 16384: 1632 MiB eager, 1376 MiB for compile / fused / `liger_oop`, and 1120 MiB for Liger and fused in-place (−31% vs eager). The difference between in-place and out-of-place is exactly one `[T, I]` (256 MiB here, 96 MiB at Qwen3-0.6B with T = 16384).
   - `LigerPhi3SwiGLUMLP` (`chunk` + `ensure_contiguous` copies) gives back part of the saving (1280 MiB) and is **0.5–2.1% slower than plain eager** (4.4–8.9% slower than `LigerSwiGLUMLP`), because the two contiguous copies cost more than the fusion saves. The fused gate-up kernel is the right path for that layout.
   - On a whole MLP, the time is dominated by the three GEMMs. Liger is 3.7–7.0% faster than eager and within 1.4% of `torch.compile`.
4. **Bandwidth: the kernels run at copy speed for every I, and the power-of-two padding is free.**
   - Liger forward runs at 211–234 GB/s and backward at 223–231 GB/s, against a copy at 221–229. That is 1.4–1.7× eager forward and 1.65–1.8× eager backward, and the same as `torch.compile`.
   - Masked lanes cost no DRAM traffic. I = 3072 (75% of lanes used), 5632, 11008 and 18944 (58%) run as fast as the powers of two. The one exception is the separate-tensor forward at I = 4104 (50% of lanes), which is 6% below copy in both runs. The fused gate-up forward at the same I is not.
   - No kernel spills; the most registers used is 64 per thread at 32 warps.
   - The out-of-place backward is within 3.5% of the in-place one (mostly within 2%), so writing into `a` / `b` buys memory, not speed.

Takeaway: Liger's SwiGLU kernel is as fast as `torch.compile` (1.4–1.8× eager on the elementwise op), has floor-accurate gradients, and saves two `[T, I]` activations at peak against eager. It does so by overwriting the saved gate / up activations with their gradients behind autograd's back. That is fine for the stock `LigerSwiGLUMLP` graph. But a forward hook or logger that keeps those activations reads gradients after backward, leaf inputs are clobbered, any other op that saved the same tensor gets wrong gradients depending on autograd's execution order, and a second `retain_graph=True` backward is silently wrong (relative error 1.15). The fused gate-up kernel already raises for the last case; adding the same guard to `LigerSiLUMulFunction`, or falling back to the out-of-place backward (same speed, one extra `[T, I]`), would close the rest. Dropping the bf16 cast of silu(a) in the forward pass puts `c` on the rounding floor for free. On Phi3-style fused `gate_up` layers, `LigerPhi3SwiGLUMLP`'s chunk + copy path is slower than eager; the fused gate-up kernel avoids that.

Full numbers: [results.md](results.md) / [results.json](results.json).
