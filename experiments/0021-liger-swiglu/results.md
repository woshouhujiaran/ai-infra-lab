Liger-Kernel 1ab779ff09, triton-windows 3.8.0, torch 2.14.1+cu126, NVIDIA GeForce RTX 4060 Laptop GPU (memory cap 0.6), runtime 121.3 s, peak allocated 3636.2 MiB

## 1. Aliasing (bf16, T = 256, H = 512, I = 1536; max rel L2 grad error over inputs vs fp64 eager autograd; `x eager` = error / eager bf16 error; stash = fraction of elements changed by backward)

| case | variant | grad rel err | x eager | 1st call err | stash changed | grad aliases input | error |
|---|---|---|---|---|---|---|---|
| mlp | eager | 4.543e-03 | 1.0 |  |  |  |  |
| mlp | liger | 4.253e-03 | 0.94 |  |  |  |  |
| mlp | liger_oop | 4.253e-03 | 0.94 |  |  |  |  |
| stash_gate_up_outputs | eager | 4.543e-03 | 1.0 |  | gate_out 0.000, up_out 0.000 |  |  |
| stash_gate_up_outputs | liger | 4.253e-03 | 0.94 |  | gate_out 1.000, up_out 1.000 |  |  |
| stash_gate_up_outputs | liger_oop | 4.253e-03 | 0.94 |  | gate_out 0.000, up_out 0.000 |  |  |
| leaf_inputs | eager | 2.911e-03 | 1.0 |  | a 0.000, b 0.000 | False |  |
| leaf_inputs | liger | 2.351e-03 | 0.81 |  | a 0.999, b 0.999 | True |  |
| leaf_inputs | liger_oop | 2.351e-03 | 0.81 |  | a 0.000, b 0.000 | False |  |
| gate_l2_before | eager | 4.273e-03 | 1.0 |  |  |  |  |
| gate_l2_before | liger | 1.002e+00 | 234.58 |  |  |  |  |
| gate_l2_before | liger_oop | 4.253e-03 | 1.0 |  |  |  |  |
| gate_l2_after | eager | 4.273e-03 | 1.0 |  |  |  |  |
| gate_l2_after | liger | 4.253e-03 | 1.0 |  |  |  |  |
| gate_l2_after | liger_oop | 4.253e-03 | 1.0 |  |  |  |  |
| retain_graph_twice | eager | 4.543e-03 | 1.0 | 4.543e-03 |  |  |  |
| retain_graph_twice | liger | 1.150e+00 | 253.22 | 4.253e-03 |  |  |  |
| retain_graph_twice | liger_oop | 4.253e-03 | 0.94 | 4.253e-03 |  |  |  |
| checkpoint | eager | 4.543e-03 | 1.0 |  |  |  |  |
| checkpoint | liger | 4.253e-03 | 0.94 |  |  |  |  |
| checkpoint | liger_oop | 4.253e-03 | 0.94 |  |  |  |  |
| gate_up_stash | eager | 4.508e-03 | 1.0 |  | gate_up_out 0.000 |  |  |
| gate_up_stash | liger | 4.253e-03 | 0.94 |  | gate_up_out 0.000 |  |  |
| gate_up_stash | liger_oop | 4.253e-03 | 0.94 |  | gate_up_out 0.000 |  |  |
| gate_up_stash | liger_fused | 4.253e-03 | 0.94 |  | gate_up_out 0.000 |  |  |
| gate_up_stash | liger_fused_inplace | 4.253e-03 | 0.94 |  | gate_up_out 1.000 |  |  |
| gate_up_retain_twice | eager | 4.508e-03 | 1.0 | 4.508e-03 |  |  |  |
| gate_up_retain_twice | liger | 1.150e+00 | 255.14 | 4.253e-03 |  |  |  |
| gate_up_retain_twice | liger_oop | 4.253e-03 | 0.94 | 4.253e-03 |  |  |  |
| gate_up_retain_twice | liger_fused | 4.253e-03 | 0.94 | 4.253e-03 |  |  |  |
| gate_up_retain_twice | liger_fused_inplace |  |  |  |  |  | RuntimeError: LigerFusedGateUpSiLUMulFunction(in_place=True) can only run backward once -- the saved activation is overwritten with th |

Cases:

- `mlp`: HF LlamaMLP graph: down(act(gate(x), up(x)))
- `stash_gate_up_outputs`: same, gate/up outputs kept (forward hook / activation logging)
- `leaf_inputs`: act(a, b) on leaf tensors
- `gate_l2_before`: aux = 0.5 * gate.pow(2).sum() built before act
- `gate_l2_after`: aux = 0.5 * gate.pow(2).sum() built after act
- `retain_graph_twice`: MLP, autograd.grad twice with retain_graph=True
- `checkpoint`: MLP under torch.utils.checkpoint (non-reentrant)
- `gate_up_stash`: fused gate_up Linear (Phi3 / Megatron), output kept
- `gate_up_retain_twice`: fused gate_up Linear, autograd.grad twice

## 2. Error vs fp64 (T = 2048, I = 8192; a = std * N(0,1), b, dc ~ N(0,1); rel L2; `floor` = fp64 result rounded to the dtype)

| dtype | gate std | output | floor | eager | torch.compile | liger | liger_oop | liger_fused | fp32-mul fwd |
|---|---|---|---|---|---|---|---|---|---|
| fp32 | 1.0 | c | 2.530e-08 | 4.753e-08 | 5.347e-08 | 5.443e-08 | 5.443e-08 | 5.443e-08 | 5.443e-08 |
| fp32 | 1.0 | da | 2.533e-08 | 5.906e-08 | 6.024e-08 | 5.422e-08 | 5.422e-08 | 5.422e-08 |  |
| fp32 | 1.0 | db | 2.530e-08 | 4.752e-08 | 5.348e-08 | 5.448e-08 | 5.448e-08 | 5.448e-08 |  |
| fp32 | 4.0 | c | 2.530e-08 | 4.927e-08 | 5.358e-08 | 5.364e-08 | 5.364e-08 | 5.364e-08 | 5.364e-08 |
| fp32 | 4.0 | da | 2.533e-08 | 1.305e-07 | 1.370e-07 | 1.348e-07 | 1.348e-07 | 1.348e-07 |  |
| fp32 | 4.0 | db | 2.531e-08 | 4.928e-08 | 5.354e-08 | 5.360e-08 | 5.360e-08 | 5.360e-08 |  |
| fp16 | 1.0 | c | 2.073e-04 | 2.915e-04 | 2.073e-04 | 2.915e-04 | 2.915e-04 | 2.915e-04 | 2.073e-04 |
| fp16 | 1.0 | da | 2.072e-04 | 2.931e-04 | 2.072e-04 | 2.072e-04 | 2.072e-04 | 2.072e-04 |  |
| fp16 | 1.0 | db | 2.073e-04 | 2.917e-04 | 2.073e-04 | 2.073e-04 | 2.073e-04 | 2.073e-04 |  |
| fp16 | 4.0 | c | 2.072e-04 | 2.754e-04 | 2.072e-04 | 2.754e-04 | 2.754e-04 | 2.754e-04 | 2.072e-04 |
| fp16 | 4.0 | da | 2.075e-04 | 2.932e-04 | 2.075e-04 | 2.075e-04 | 2.075e-04 | 2.075e-04 |  |
| fp16 | 4.0 | db | 2.072e-04 | 2.754e-04 | 2.072e-04 | 2.072e-04 | 2.072e-04 | 2.072e-04 |  |
| bf16 | 1.0 | c | 1.661e-03 | 2.411e-03 | 1.661e-03 | 2.411e-03 | 2.411e-03 | 2.411e-03 | 1.661e-03 |
| bf16 | 1.0 | da | 1.659e-03 | 2.342e-03 | 1.659e-03 | 1.659e-03 | 1.659e-03 | 1.659e-03 |  |
| bf16 | 1.0 | db | 1.659e-03 | 2.411e-03 | 1.659e-03 | 1.659e-03 | 1.659e-03 | 1.659e-03 |  |
| bf16 | 4.0 | c | 1.651e-03 | 2.133e-03 | 1.651e-03 | 2.133e-03 | 2.133e-03 | 2.133e-03 | 1.651e-03 |
| bf16 | 4.0 | da | 1.657e-03 | 2.335e-03 | 1.657e-03 | 1.657e-03 | 1.657e-03 | 1.657e-03 |  |
| bf16 | 4.0 | db | 1.651e-03 | 2.132e-03 | 1.651e-03 | 1.651e-03 | 1.651e-03 | 1.651e-03 |  |

Out-of-place backward bit-identical to upstream: fp32/1.0 True, fp32/4.0 True, fp16/1.0 True, fp16/4.0 True, bf16/1.0 True, bf16/4.0 True

Multipliers (bf16, gate 0.5, down 2.0): gate std 1.0: c 2.379e-03 (floor 1.658e-03), da 1.661e-03 (floor 1.661e-03), db 1.658e-03; gate std 4.0: c 2.349e-03 (floor 1.658e-03), da 1.659e-03 (floor 1.659e-03), db 1.660e-03

## 3. Full MLP fwd + bwd (bf16; held = memory still allocated after forward incl. output; extra peak = max allocated during fwd + bwd above the inputs, incl. weight grads; do_bench median)

### held after forward (MiB)

| shape | T | [T, I] MiB | eager | eager_chunk | torch_compile | liger | liger_chunk | liger_fused | liger_fused_inplace | liger_oop |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 2048 | 12.0 | 52.0 | 52.0 | 40.0 | 40.0 | 40.0 | 40.0 | 40.0 | 40.0 |
| Qwen3-0.6B | 8192 | 48.0 | 208.0 | 208.0 | 160.0 | 160.0 | 160.0 | 160.0 | 160.0 | 160.0 |
| Qwen3-0.6B | 16384 | 96.0 | 416.0 | 416.0 | 320.0 | 320.0 | 320.0 | 320.0 | 320.0 | 320.0 |
| Llama-3.2-1B | 2048 | 32.0 | 136.0 | 136.0 | 104.0 | 104.0 | 104.0 | 104.0 | 104.0 | 104.0 |
| Llama-3.2-1B | 8192 | 128.0 | 544.0 | 544.0 | 416.0 | 416.0 | 416.0 | 416.0 | 416.0 | 416.0 |
| Llama-3.2-1B | 16384 | 256.0 | 1088.0 | 1088.0 | 832.0 | 832.0 | 832.0 | 832.0 | 832.0 | 832.0 |

### extra peak (MiB)

| shape | T | [T, I] MiB | eager | eager_chunk | torch_compile | liger | liger_chunk | liger_fused | liger_fused_inplace | liger_oop |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 2048 | 12.0 | 82.0 | 82.0 | 70.0 | 58.0 | 60.0 | 70.0 | 58.0 | 70.0 |
| Qwen3-0.6B | 8192 | 48.0 | 310.0 | 310.0 | 262.0 | 214.0 | 240.0 | 262.0 | 214.0 | 262.0 |
| Qwen3-0.6B | 16384 | 96.0 | 614.0 | 614.0 | 518.0 | 422.0 | 480.0 | 518.0 | 422.0 | 518.0 |
| Llama-3.2-1B | 2048 | 32.0 | 232.0 | 232.0 | 200.0 | 168.0 | 176.0 | 200.0 | 176.0 | 200.0 |
| Llama-3.2-1B | 8192 | 128.0 | 832.0 | 832.0 | 704.0 | 576.0 | 640.0 | 704.0 | 576.0 | 704.0 |
| Llama-3.2-1B | 16384 | 256.0 | 1632.0 | 1632.0 | 1376.0 | 1120.0 | 1280.0 | 1376.0 | 1120.0 | 1376.0 |

### fwd + bwd (ms)

| shape | T | [T, I] MiB | eager | eager_chunk | torch_compile | liger | liger_chunk | liger_fused | liger_fused_inplace | liger_oop |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 2048 | 12.0 | 4.485 | 4.67 | 4.217 | 4.252 | 4.579 | 4.224 | 4.209 | 4.264 |
| Qwen3-0.6B | 8192 | 48.0 | 18.436 | 19.048 | 16.983 | 17.149 | 18.674 | 16.894 | 16.898 | 17.206 |
| Qwen3-0.6B | 16384 | 96.0 | 36.741 | 37.868 | 33.74 | 34.199 | 37.186 | 33.657 | 33.78 | 34.199 |
| Llama-3.2-1B | 2048 | 32.0 | 22.456 | 22.852 | 21.491 | 21.609 | 22.624 | 21.543 | 21.456 | 21.536 |
| Llama-3.2-1B | 8192 | 128.0 | 89.567 | 91.365 | 85.676 | 86.237 | 90.434 | 85.791 | 85.801 | 86.288 |
| Llama-3.2-1B | 16384 | 256.0 | 178.584 | 181.227 | 170.784 | 171.926 | 179.447 | 170.348 | 170.367 | 171.96 |

## 4. Elementwise bandwidth (bf16, T = 4096; fwd 3 * T * I * 2 bytes, bwd 5 * T * I * 2 bytes; do_bench median of 3 rounds; `xE` = speedup vs eager)

| I | BLOCK | lanes used | copy | fwd liger | fwd fused | fwd fp32-mul | fwd eager | fwd compile | bwd liger | bwd oop | bwd fused | bwd fused in-place | bwd eager | bwd compile | fwd liger xE | bwd liger xE | bwd compile xE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 2048 | 1.0 | 221.4 | 218.5 | 218.5 | 218.5 | 156.0 | 218.5 | 222.6 | 219.0 | 221.4 | 225.1 | 134.8 | 219.6 | 1.4 | 1.65 | 1.63 |
| 3072 | 4096 | 0.75 | 223.4 | 221.1 | 223.4 | 220.7 | 136.8 | 219.4 | 226.7 | 223.0 | 227.3 | 228.0 | 128.0 | 224.2 | 1.62 | 1.77 | 1.75 |
| 4096 | 4096 | 1.0 | 225.2 | 225.5 | 225.5 | 225.2 | 136.1 | 225.5 | 227.2 | 225.7 | 229.5 | 229.1 | 127.4 | 223.5 | 1.66 | 1.78 | 1.75 |
| 4104 | 8192 | 0.501 | 225.6 | 211.3 | 222.3 | 211.4 | 135.1 | 224.4 | 230.7 | 222.7 | 226.4 | 226.4 | 128.0 | 224.9 | 1.56 | 1.8 | 1.76 |
| 5632 | 8192 | 0.688 | 225.3 | 225.3 | 227.0 | 226.0 | 136.9 | 226.8 | 226.8 | 227.8 | 227.8 | 227.1 | 129.2 | 228.9 | 1.65 | 1.76 | 1.77 |
| 8192 | 8192 | 1.0 | 228.0 | 228.9 | 227.3 | 228.9 | 137.4 | 229.7 | 223.7 | 225.7 | 226.3 | 226.1 | 129.3 | 227.6 | 1.67 | 1.73 | 1.76 |
| 11008 | 16384 | 0.672 | 228.4 | 230.7 | 230.9 | 232.6 | 138.0 | 228.8 | 229.3 | 230.7 | 231.0 | 228.9 | 129.8 | 228.9 | 1.67 | 1.77 | 1.76 |
| 14336 | 16384 | 0.875 | 228.2 | 231.4 | 230.2 | 230.8 | 138.5 | 231.3 | 230.1 | 231.6 | 235.8 | 230.5 | 129.9 | 229.5 | 1.67 | 1.77 | 1.77 |
| 16384 | 16384 | 1.0 | 228.3 | 233.8 | 233.8 | 231.8 | 138.7 | 230.9 | 229.4 | 232.4 | 232.0 | 230.8 | 129.3 | 226.7 | 1.69 | 1.77 | 1.75 |
| 18944 | 32768 | 0.578 | 229.1 | 230.1 | 229.5 | 230.4 | 138.5 | 230.3 | 226.7 | 227.5 | 227.5 | 226.3 | 132.0 | 228.2 | 1.66 | 1.72 | 1.73 |

### Kernel resources

| I | kernel | BLOCK | warps | regs/thread | spills (bytes) |
|---|---|---|---|---|---|
| 2048 | fwd | 2048 | 8 | 31 | 0 |
| 2048 | bwd | 2048 | 8 | 37 | 0 |
| 2048 | bwd_oop | 2048 | 8 | 35 | 0 |
| 3072 | fwd | 4096 | 8 | 39 | 0 |
| 3072 | bwd | 4096 | 8 | 40 | 0 |
| 3072 | bwd_oop | 4096 | 8 | 47 | 0 |
| 4096 | fwd | 4096 | 8 | 39 | 0 |
| 4096 | bwd | 4096 | 8 | 44 | 0 |
| 4096 | bwd_oop | 4096 | 8 | 43 | 0 |
| 4104 | fwd | 8192 | 16 | 40 | 0 |
| 4104 | bwd | 8192 | 16 | 52 | 0 |
| 4104 | bwd_oop | 8192 | 16 | 40 | 0 |
| 5632 | fwd | 8192 | 16 | 40 | 0 |
| 5632 | bwd | 8192 | 16 | 42 | 0 |
| 5632 | bwd_oop | 8192 | 16 | 44 | 0 |
| 8192 | fwd | 8192 | 16 | 40 | 0 |
| 8192 | bwd | 8192 | 16 | 45 | 0 |
| 8192 | bwd_oop | 8192 | 16 | 44 | 0 |
| 11008 | fwd | 16384 | 16 | 40 | 0 |
| 11008 | bwd | 16384 | 16 | 40 | 0 |
| 11008 | bwd_oop | 16384 | 16 | 55 | 0 |
| 14336 | fwd | 16384 | 16 | 44 | 0 |
| 14336 | bwd | 16384 | 16 | 64 | 0 |
| 14336 | bwd_oop | 16384 | 16 | 62 | 0 |
| 16384 | fwd | 16384 | 16 | 40 | 0 |
| 16384 | bwd | 16384 | 16 | 64 | 0 |
| 16384 | bwd_oop | 16384 | 16 | 53 | 0 |
| 18944 | fwd | 32768 | 32 | 64 | 0 |
| 18944 | bwd | 32768 | 32 | 64 | 0 |
| 18944 | bwd_oop | 32768 | 32 | 64 | 0 |
