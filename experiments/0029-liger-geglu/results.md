Liger-Kernel 1ab779ff09, triton-windows 3.8.0, torch 2.14.1+cu126, NVIDIA GeForce RTX 4060 Laptop GPU (memory cap 0.6), runtime 166.7 s, peak allocated 2704.2 MiB

## 1. Aliasing (bf16, T = 256, H = 512, I = 1536; max rel L2 grad error over inputs vs fp64 eager tanh-approx autograd; `x eager` = error / eager_tanh bf16 error; stash = fraction of elements changed by backward)

| case | variant | grad rel err | x eager | 1st call err | stash changed | grad aliases input | error |
|---|---|---|---|---|---|---|---|
| mlp | eager_tanh | 4.495e-03 | 1.0 |  |  |  |  |
| mlp | liger | 4.495e-03 | 1.0 |  |  |  |  |
| mlp | liger_oop | 4.495e-03 | 1.0 |  |  |  |  |
| stash_gate_up_outputs | eager_tanh | 4.495e-03 | 1.0 |  | gate_out 0.000, up_out 0.000 |  |  |
| stash_gate_up_outputs | liger | 4.495e-03 | 1.0 |  | gate_out 1.000, up_out 1.000 |  |  |
| stash_gate_up_outputs | liger_oop | 4.495e-03 | 1.0 |  | gate_out 0.000, up_out 0.000 |  |  |
| leaf_inputs | eager_tanh | 2.885e-03 | 1.0 |  | a 0.000, b 0.000 | False |  |
| leaf_inputs | liger | 2.885e-03 | 1.0 |  | a 1.000, b 1.000 | True |  |
| leaf_inputs | liger_oop | 2.885e-03 | 1.0 |  | a 0.000, b 0.000 | False |  |
| gate_l2_before | eager_tanh | 4.210e-03 | 1.0 |  |  |  |  |
| gate_l2_before | liger | 1.002e+00 | 238.05 |  |  |  |  |
| gate_l2_before | liger_oop | 4.210e-03 | 1.0 |  |  |  |  |
| gate_l2_after | eager_tanh | 4.210e-03 | 1.0 |  |  |  |  |
| gate_l2_after | liger | 4.210e-03 | 1.0 |  |  |  |  |
| gate_l2_after | liger_oop | 4.210e-03 | 1.0 |  |  |  |  |
| retain_graph_twice | eager_tanh | 4.495e-03 | 1.0 | 4.495e-03 |  |  |  |
| retain_graph_twice | liger | 1.103e+00 | 245.31 | 4.495e-03 |  |  |  |
| retain_graph_twice | liger_oop | 4.495e-03 | 1.0 | 4.495e-03 |  |  |  |
| checkpoint | eager_tanh | 4.495e-03 | 1.0 |  |  |  |  |
| checkpoint | liger | 4.495e-03 | 1.0 |  |  |  |  |
| checkpoint | liger_oop | 4.495e-03 | 1.0 |  |  |  |  |

Cases:

- `mlp`: HF GemmaMLP graph: down(gelu_tanh(gate(x)) * up(x))
- `stash_gate_up_outputs`: same, gate/up outputs kept (forward hook / activation logging)
- `leaf_inputs`: act(a, b) on leaf tensors
- `gate_l2_before`: aux = 0.5 * gate.pow(2).sum() built before act
- `gate_l2_after`: aux = 0.5 * gate.pow(2).sum() built after act
- `retain_graph_twice`: MLP, autograd.grad twice with retain_graph=True
- `checkpoint`: MLP under torch.utils.checkpoint (non-reentrant)

## 2. Error vs fp64 (T = 2048, I = 8192; a = std * N(0,1), b, dc ~ N(0,1); rel L2)

### vs fp64 tanh-approx (Liger / HF Gemma target)

| dtype | gate std | output | floor | eager_tanh | compile_tanh | liger | liger_oop | fp32-mul fwd |
|---|---|---|---|---|---|---|---|---|
| fp32 | 1.0 | c | 2.532e-08 | 4.603e-08 | 4.603e-08 | 4.603e-08 | 4.603e-08 | 4.603e-08 |
| fp32 | 1.0 | da | 2.530e-08 | 5.751e-08 | 5.764e-08 | 5.765e-08 | 5.765e-08 |  |
| fp32 | 1.0 | db | 2.530e-08 | 4.601e-08 | 4.601e-08 | 4.601e-08 | 4.601e-08 |  |
| fp32 | 4.0 | c | 2.530e-08 | 3.324e-08 | 3.324e-08 | 3.324e-08 | 3.324e-08 | 3.324e-08 |
| fp32 | 4.0 | da | 2.529e-08 | 1.552e-07 | 1.552e-07 | 1.552e-07 | 1.552e-07 |  |
| fp32 | 4.0 | db | 2.531e-08 | 3.323e-08 | 3.323e-08 | 3.323e-08 | 3.323e-08 |  |
| fp16 | 1.0 | c | 2.072e-04 | 2.924e-04 | 2.072e-04 | 2.924e-04 | 2.924e-04 | 2.072e-04 |
| fp16 | 1.0 | da | 2.072e-04 | 2.933e-04 | 2.072e-04 | 2.933e-04 | 2.933e-04 |  |
| fp16 | 1.0 | db | 2.074e-04 | 2.926e-04 | 2.074e-04 | 2.926e-04 | 2.926e-04 |  |
| fp16 | 4.0 | c | 2.072e-04 | 2.219e-04 | 2.072e-04 | 2.219e-04 | 2.219e-04 | 2.072e-04 |
| fp16 | 4.0 | da | 2.073e-04 | 2.723e-04 | 2.073e-04 | 2.723e-04 | 2.723e-04 |  |
| fp16 | 4.0 | db | 2.073e-04 | 2.219e-04 | 2.073e-04 | 2.219e-04 | 2.219e-04 |  |
| bf16 | 1.0 | c | 1.658e-03 | 2.382e-03 | 1.658e-03 | 2.382e-03 | 2.382e-03 | 1.658e-03 |
| bf16 | 1.0 | da | 1.658e-03 | 2.342e-03 | 1.658e-03 | 2.342e-03 | 2.342e-03 |  |
| bf16 | 1.0 | db | 1.659e-03 | 2.382e-03 | 1.659e-03 | 2.382e-03 | 2.382e-03 |  |
| bf16 | 4.0 | c | 1.653e-03 | 1.747e-03 | 1.653e-03 | 1.747e-03 | 1.747e-03 | 1.653e-03 |
| bf16 | 4.0 | da | 1.657e-03 | 2.146e-03 | 1.657e-03 | 2.146e-03 | 2.146e-03 |  |
| bf16 | 4.0 | db | 1.653e-03 | 1.748e-03 | 1.653e-03 | 1.748e-03 | 1.748e-03 |  |

### vs fp64 exact-erf GELU (and tanh-vs-exact gap)

| dtype | gate std | output | tanh_vs_exact fp64 | eager_exact | compile_exact | liger_vs_exact | eager_exact_vs_tanh |
|---|---|---|---|---|---|---|---|
| fp32 | 1.0 | c | 1.975e-04 | 4.557e-08 | 4.506e-08 | 1.975e-04 | 1.976e-04 |
| fp32 | 1.0 | da | 4.348e-04 | 5.421e-08 | 5.418e-08 | 4.348e-04 | 4.348e-04 |
| fp32 | 1.0 | db | 1.976e-04 | 4.555e-08 | 4.505e-08 | 1.976e-04 | 1.976e-04 |
| fp32 | 4.0 | c | 7.002e-05 | 3.503e-08 | 3.425e-08 | 7.002e-05 | 7.002e-05 |
| fp32 | 4.0 | da | 4.783e-04 | 5.248e-08 | 5.173e-08 | 4.783e-04 | 4.783e-04 |
| fp32 | 4.0 | db | 7.008e-05 | 3.505e-08 | 3.427e-08 | 7.008e-05 | 7.008e-05 |
| fp16 | 1.0 | c | 1.975e-04 | 2.930e-04 | 2.073e-04 | 3.532e-04 | 3.527e-04 |
| fp16 | 1.0 | da | 4.348e-04 | 2.930e-04 | 2.074e-04 | 5.243e-04 | 5.245e-04 |
| fp16 | 1.0 | db | 1.976e-04 | 2.930e-04 | 2.074e-04 | 3.537e-04 | 3.528e-04 |
| fp16 | 4.0 | c | 7.002e-05 | 2.225e-04 | 2.072e-04 | 2.334e-04 | 2.328e-04 |
| fp16 | 4.0 | da | 4.783e-04 | 2.737e-04 | 2.073e-04 | 5.510e-04 | 5.505e-04 |
| fp16 | 4.0 | db | 7.008e-05 | 2.225e-04 | 2.072e-04 | 2.334e-04 | 2.328e-04 |
| bf16 | 1.0 | c | 1.975e-04 | 2.383e-03 | 1.658e-03 | 2.387e-03 | 2.395e-03 |
| bf16 | 1.0 | da | 4.348e-04 | 2.343e-03 | 1.658e-03 | 2.383e-03 | 2.382e-03 |
| bf16 | 1.0 | db | 1.976e-04 | 2.383e-03 | 1.659e-03 | 2.387e-03 | 2.395e-03 |
| bf16 | 4.0 | c | 7.002e-05 | 1.750e-03 | 1.653e-03 | 1.751e-03 | 1.749e-03 |
| bf16 | 4.0 | da | 4.783e-04 | 2.152e-03 | 1.657e-03 | 2.204e-03 | 2.198e-03 |
| bf16 | 4.0 | db | 7.008e-05 | 1.751e-03 | 1.653e-03 | 1.752e-03 | 1.750e-03 |

Out-of-place backward bit-identical to upstream: fp32/1.0 True, fp32/4.0 True, fp16/1.0 True, fp16/4.0 True, bf16/1.0 True, bf16/4.0 True

Liger vs same-dtype eager `F.gelu(approximate='tanh')`: fp32/std1.0: c 0.000e+00, da 1.758e-08, db 0.000e+00; fp32/std4.0: c 0.000e+00, da 1.437e-08, db 0.000e+00; fp16/std1.0: c 0.000e+00, da 1.703e-06, db 0.000e+00; fp16/std4.0: c 0.000e+00, da 5.808e-07, db 0.000e+00; bf16/std1.0: c 0.000e+00, da 1.335e-08, db 0.000e+00; bf16/std4.0: c 0.000e+00, da 6.835e-09, db 0.000e+00

## 3. Full Gemma MLP fwd + bwd (bf16; held = memory still allocated after forward incl. output; extra peak = max allocated during fwd + bwd above the inputs, incl. weight grads; do_bench median)

### held after forward (MiB)

| shape | T | [T, I] MiB | eager_tanh | eager_exact | torch_compile | liger | liger_oop |
|---|---|---|---|---|---|---|---|
| Gemma-2B | 1024 | 32.0 | 132.9 | 132.9 | 100.9 | 100.9 | 100.9 |
| Gemma-2B | 4096 | 128.0 | 528.0 | 528.0 | 400.0 | 400.0 | 400.0 |
| Gemma-2B | 8192 | 256.0 | 1056.0 | 1056.0 | 800.0 | 800.0 | 800.0 |
| Gemma-2-2B | 1024 | 18.0 | 76.5 | 76.5 | 58.5 | 58.5 | 58.5 |
| Gemma-2-2B | 4096 | 72.0 | 306.0 | 306.0 | 234.0 | 234.0 | 234.0 |
| Gemma-2-2B | 8192 | 144.0 | 612.0 | 612.0 | 468.0 | 468.0 | 468.0 |

### extra peak (MiB)

| shape | T | [T, I] MiB | eager_tanh | eager_exact | torch_compile | liger | liger_oop |
|---|---|---|---|---|---|---|---|
| Gemma-2B | 1024 | 32.0 | 260.9 | 260.9 | 232.9 | 236.9 | 236.9 |
| Gemma-2B | 4096 | 128.0 | 848.0 | 848.0 | 720.0 | 592.0 | 720.0 |
| Gemma-2B | 8192 | 256.0 | 1632.0 | 1632.0 | 1376.0 | 1120.0 | 1376.0 |
| Gemma-2-2B | 1024 | 18.0 | 154.0 | 154.0 | 148.5 | 153.0 | 153.0 |
| Gemma-2-2B | 4096 | 72.0 | 490.5 | 490.5 | 418.5 | 346.5 | 418.5 |
| Gemma-2-2B | 8192 | 144.0 | 940.5 | 940.5 | 796.5 | 652.5 | 796.5 |

### fwd + bwd (ms)

| shape | T | [T, I] MiB | eager_tanh | eager_exact | torch_compile | liger | liger_oop |
|---|---|---|---|---|---|---|---|
| Gemma-2B | 1024 | 32.0 | 22.219 | 22.227 | 21.4 | 21.38 | 21.378 |
| Gemma-2B | 4096 | 128.0 | 88.941 | 88.956 | 85.129 | 85.575 | 85.599 |
| Gemma-2B | 8192 | 256.0 | 178.007 | 178.688 | 171.429 | 171.938 | 171.866 |
| Gemma-2-2B | 1024 | 18.0 | 13.89 | 13.884 | 13.363 | 13.456 | 13.413 |
| Gemma-2-2B | 4096 | 72.0 | 55.589 | 55.612 | 53.444 | 53.695 | 53.719 |
| Gemma-2-2B | 8192 | 144.0 | 111.901 | 111.912 | 107.578 | 108.179 | 108.116 |

## 4. Elementwise bandwidth (bf16, T = 4096; fwd 3 * T * I * 2 bytes, bwd 5 * T * I * 2 bytes; do_bench median of 3 rounds; `xE` = speedup vs eager_tanh)

| I | BLOCK | lanes used | copy | fwd liger | fwd fp32-mul | fwd eager_tanh | fwd eager_exact | fwd compile | bwd liger | bwd oop | bwd eager | bwd compile | fwd liger xE | bwd liger xE | bwd compile xE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 2048 | 1.0 | 218.5 | 218.5 | 218.5 | 156.5 | 157.0 | 218.9 | 222.6 | 219.0 | 134.1 | 219.6 | 1.4 | 1.66 | 1.64 |
| 3072 | 4096 | 0.75 | 222.4 | 222.7 | 223.4 | 137.2 | 137.3 | 221.4 | 224.2 | 224.0 | 127.2 | 223.8 | 1.62 | 1.76 | 1.76 |
| 4096 | 4096 | 1.0 | 224.4 | 224.4 | 225.0 | 136.5 | 136.3 | 223.9 | 226.9 | 225.1 | 127.5 | 222.0 | 1.64 | 1.78 | 1.74 |
| 4104 | 8192 | 0.501 | 224.9 | 210.0 | 210.5 | 136.2 | 136.2 | 224.4 | 229.3 | 222.1 | 128.3 | 226.4 | 1.54 | 1.79 | 1.76 |
| 5632 | 8192 | 0.688 | 225.3 | 226.4 | 226.8 | 136.7 | 136.7 | 228.3 | 227.1 | 226.9 | 129.3 | 228.2 | 1.66 | 1.76 | 1.76 |
| 8192 | 8192 | 1.0 | 227.2 | 227.3 | 227.8 | 137.6 | 137.8 | 227.3 | 227.1 | 226.6 | 129.3 | 226.9 | 1.65 | 1.76 | 1.75 |
| 9216 | 16384 | 0.562 | 226.9 | 225.2 | 225.5 | 137.8 | 137.9 | 227.8 | 225.6 | 226.2 | 129.5 | 227.0 | 1.63 | 1.74 | 1.75 |
| 11008 | 16384 | 0.672 | 228.1 | 226.4 | 226.6 | 137.8 | 138.0 | 229.3 | 226.4 | 227.8 | 129.2 | 227.9 | 1.64 | 1.75 | 1.76 |
| 16384 | 16384 | 1.0 | 228.7 | 230.4 | 230.1 | 138.8 | 138.6 | 231.2 | 227.9 | 228.7 | 132.2 | 238.9 | 1.66 | 1.72 | 1.81 |

### Kernel resources

| I | kernel | BLOCK | warps | regs/thread | spills (bytes) |
|---|---|---|---|---|---|
| 2048 | fwd | 2048 | 8 | 40 | 0 |
| 2048 | bwd | 2048 | 8 | 40 | 0 |
| 2048 | bwd_oop | 2048 | 8 | 39 | 0 |
| 3072 | fwd | 4096 | 8 | 48 | 0 |
| 3072 | bwd | 4096 | 8 | 68 | 0 |
| 3072 | bwd_oop | 4096 | 8 | 58 | 0 |
| 4096 | fwd | 4096 | 8 | 48 | 0 |
| 4096 | bwd | 4096 | 8 | 64 | 0 |
| 4096 | bwd_oop | 4096 | 8 | 60 | 0 |
| 4104 | fwd | 8192 | 16 | 38 | 0 |
| 4104 | bwd | 8192 | 16 | 64 | 0 |
| 4104 | bwd_oop | 8192 | 16 | 54 | 0 |
| 5632 | fwd | 8192 | 16 | 50 | 0 |
| 5632 | bwd | 8192 | 16 | 64 | 0 |
| 5632 | bwd_oop | 8192 | 16 | 76 | 0 |
| 8192 | fwd | 8192 | 16 | 49 | 0 |
| 8192 | bwd | 8192 | 16 | 64 | 0 |
| 8192 | bwd_oop | 8192 | 16 | 64 | 0 |
| 9216 | fwd | 16384 | 16 | 70 | 0 |
| 9216 | bwd | 16384 | 16 | 102 | 0 |
| 9216 | bwd_oop | 16384 | 16 | 103 | 0 |
| 11008 | fwd | 16384 | 16 | 70 | 0 |
| 11008 | bwd | 16384 | 16 | 102 | 0 |
| 11008 | bwd_oop | 16384 | 16 | 103 | 0 |
| 16384 | fwd | 16384 | 16 | 94 | 0 |
| 16384 | bwd | 16384 | 16 | 128 | 12 |
| 16384 | bwd_oop | 16384 | 16 | 128 | 8 |
