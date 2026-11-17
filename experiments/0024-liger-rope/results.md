Liger-Kernel 1ab779ff09, triton-windows 3.8.0, torch 2.14.1+cu126, NVIDIA GeForce RTX 4060 Laptop GPU (memory cap 0.6), runtime 30.3 s, peak allocated 746.0 MiB

## 0. API footguns

| case | detail |
|---|---|
| position_ids_ignored | has_param=True; liger(rev ids)==none q/k rel=0.0/0.0; HF rev-indexed vs liger q/k=0.9979/1.0018; same_as_ignoring=True |
| gpt_oss_half_width_without_rotary_dim | cos_width=32 head_dim=64; vs gpt_oss q/k=1.1064/1.0955; vs true partial=0.9449; tail_unchanged=True; gpt_oss_rotates_whole_head=True |

## 1. In-place aliasing (bf16, B=2, S=128, Hq/Hk=8/2, D=64)

| case | fields |
|---|---|
| contiguous_BHSD | input_q_mutated=False, input_k_mutated=False, out_q_same_ptr_as_in=False, out_k_same_ptr_as_in=False, out_q_is_input_q=False |
| transpose_view_of_BSHD | storage_q_mutated=True, storage_k_mutated=True, out_q_same_storage_as_bshd=True, view_and_out_share_ptr=True |
| rope_forward_contiguous_BHSD | input_mutated=False, out_contiguous_BSHD=True |
| grad_like_stride | out_q_contiguous=False, dQ_contiguous=False, dQ_stride=[65536, 64, 512, 1], out_q_stride=[65536, 64, 512, 1], same_stride_as_out=True |
| retain_graph_fresh_grads | second_vs_first_q=0.0, second_vs_first_k=0.0 |
| retain_graph_reuse_dQ_dK | reuse_first_vs_clean_q=0.0, reuse_first_vs_clean_k=0.0, reuse_second_vs_first_q=1.0029, reuse_second_vs_first_k=0.9927, caller_dQ_overwritten=True, caller_dK_overwritten=True, returned_grad_aliases_dQ=True |

## 2. Partial RoPE + GQA (bf16)

| label | Hq | Hk | D | rd | tail_q | tail_k | q vs hf | k vs hf |
|---|---|---|---|---|---|---|---|---|
| mha_full | 16 | 16 | 128 | 128 | True | True | 0.0018 | 0.0018 |
| gqa_full | 16 | 4 | 128 | 128 | True | True | 0.0018 | 0.0018 |
| gqa_partial_64 | 16 | 4 | 128 | 64 | True | True | 0.0013 | 0.0013 |
| gqa_partial_96 | 8 | 2 | 128 | 96 | True | True | 0.0016 | 0.0016 |

## 3. Error vs fp64 (B=1, S=512, Hq/Hk=16/4, D=128; floor is dtype-rounded fp64 for q/k only)

| dtype | mode | out | floor | liger | eager_hf | compile | liger vs hf |
|---|---|---|---|---|---|---|---|
| fp32 | full | q | 0.0 | 3.214e-08 | 4.352e-08 | 2.784e-08 | 3.460e-08 |
| fp32 | full | k | 0.0 | 3.227e-08 | 4.357e-08 | 2.783e-08 | 3.471e-08 |
| fp32 | full | dq | None | 2.783e-08 | 4.345e-08 | 2.783e-08 | 3.782e-08 |
| fp32 | full | dk | None | 2.769e-08 | 4.345e-08 | 2.769e-08 | 3.787e-08 |
| fp32 | partial | q | 0.0 | 2.268e-08 | 3.078e-08 | 1.974e-08 | 2.456e-08 |
| fp32 | partial | k | 0.0 | 2.274e-08 | 3.092e-08 | 1.990e-08 | 2.466e-08 |
| fp32 | partial | dq | None | 1.979e-08 | 3.079e-08 | 1.979e-08 | 2.682e-08 |
| fp32 | partial | dk | None | 1.980e-08 | 3.080e-08 | 1.980e-08 | 2.692e-08 |
| fp32 | gpt_oss | q | 0.0 | 3.214e-08 | 4.352e-08 | 2.784e-08 | 3.460e-08 |
| fp32 | gpt_oss | k | 0.0 | 3.227e-08 | 4.357e-08 | 2.783e-08 | 3.471e-08 |
| fp32 | gpt_oss | dq | None | 2.783e-08 | 4.345e-08 | 2.783e-08 | 3.782e-08 |
| fp32 | gpt_oss | dk | None | 2.769e-08 | 4.345e-08 | 2.769e-08 | 3.787e-08 |
| fp16 | full | q | 2.072e-04 | 2.523e-04 | 2.963e-04 | 2.072e-04 | 2.778e-04 |
| fp16 | full | k | 2.074e-04 | 2.512e-04 | 2.955e-04 | 2.074e-04 | 2.772e-04 |
| fp16 | full | dq | None | 2.412e-04 | 2.966e-04 | 2.072e-04 | 3.016e-04 |
| fp16 | full | dk | None | 2.414e-04 | 2.967e-04 | 2.073e-04 | 3.017e-04 |
| fp16 | partial | q | 1.466e-04 | 1.782e-04 | 2.094e-04 | 1.466e-04 | 1.973e-04 |
| fp16 | partial | k | 1.470e-04 | 1.789e-04 | 2.102e-04 | 1.470e-04 | 1.956e-04 |
| fp16 | partial | dq | None | 1.703e-04 | 2.095e-04 | 1.464e-04 | 2.129e-04 |
| fp16 | partial | dk | None | 1.707e-04 | 2.106e-04 | 1.473e-04 | 2.148e-04 |
| fp16 | gpt_oss | q | 2.072e-04 | 2.523e-04 | 2.963e-04 | 2.072e-04 | 2.778e-04 |
| fp16 | gpt_oss | k | 2.074e-04 | 2.512e-04 | 2.955e-04 | 2.074e-04 | 2.772e-04 |
| fp16 | gpt_oss | dq | None | 2.412e-04 | 2.966e-04 | 2.072e-04 | 3.016e-04 |
| fp16 | gpt_oss | dk | None | 2.414e-04 | 2.967e-04 | 2.073e-04 | 3.017e-04 |
| bf16 | full | q | 0.0017 | 0.002 | 0.0023 | 0.0017 | 0.0021 |
| bf16 | full | k | 0.0017 | 0.002 | 0.0023 | 0.0017 | 0.0021 |
| bf16 | full | dq | None | 0.0019 | 0.0023 | 0.0017 | 0.0023 |
| bf16 | full | dk | None | 0.0019 | 0.0023 | 0.0017 | 0.0022 |
| bf16 | partial | q | 0.0012 | 0.0014 | 0.0016 | 0.0012 | 0.0015 |
| bf16 | partial | k | 0.0012 | 0.0014 | 0.0016 | 0.0012 | 0.0015 |
| bf16 | partial | dq | None | 0.0014 | 0.0016 | 0.0012 | 0.0016 |
| bf16 | partial | dk | None | 0.0014 | 0.0016 | 0.0012 | 0.0016 |
| bf16 | gpt_oss | q | 0.0017 | 0.002 | 0.0023 | 0.0017 | 0.0021 |
| bf16 | gpt_oss | k | 0.0017 | 0.002 | 0.0023 | 0.0017 | 0.0021 |
| bf16 | gpt_oss | dq | None | 0.0019 | 0.0023 | 0.0017 | 0.0023 |
| bf16 | gpt_oss | dk | None | 0.0019 | 0.0023 | 0.0017 | 0.0022 |

## 4. Memory + latency (bf16; do_bench median)

### held after forward (MiB)

| shape | S | Hq | Hk | D | liger | eager_hf | torch_compile |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | 512 | 32 | 8 | 64 | 5.0 | 5.0 | 5.0 |
| Llama-3.2-1B | 2048 | 32 | 8 | 64 | 20.0 | 20.0 | 20.0 |
| Llama-3.2-1B | 8192 | 32 | 8 | 64 | 80.0 | 80.0 | 80.0 |
| Qwen3-0.6B | 512 | 16 | 8 | 128 | 6.0 | 6.0 | 6.0 |
| Qwen3-0.6B | 2048 | 16 | 8 | 128 | 24.0 | 24.0 | 24.0 |
| Qwen3-0.6B | 8192 | 16 | 8 | 128 | 96.0 | 96.0 | 96.0 |

### extra peak during bwd (MiB)

| shape | S | Hq | Hk | D | liger | eager_hf | torch_compile |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | 512 | 32 | 8 | 64 | 0.0 | 6.5 | 2.5 |
| Llama-3.2-1B | 2048 | 32 | 8 | 64 | 0.0 | 26.0 | 10.0 |
| Llama-3.2-1B | 8192 | 32 | 8 | 64 | 0.0 | 104.0 | 40.0 |
| Qwen3-0.6B | 512 | 16 | 8 | 128 | 0.0 | 7.0 | 3.0 |
| Qwen3-0.6B | 2048 | 16 | 8 | 128 | 0.0 | 28.0 | 12.0 |
| Qwen3-0.6B | 8192 | 16 | 8 | 128 | 0.0 | 112.0 | 48.0 |

### fwd + bwd (ms)

| shape | S | Hq | Hk | D | liger | eager_hf | torch_compile |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | 512 | 32 | 8 | 64 | 0.0778 | 0.4342 | 0.1229 |
| Llama-3.2-1B | 2048 | 32 | 8 | 64 | 0.2335 | 0.7895 | 0.3308 |
| Llama-3.2-1B | 8192 | 32 | 8 | 64 | 1.4346 | 5.5516 | 1.279 |
| Qwen3-0.6B | 512 | 16 | 8 | 128 | 0.085 | 0.3686 | 0.1044 |
| Qwen3-0.6B | 2048 | 16 | 8 | 128 | 0.2765 | 0.9498 | 0.3799 |
| Qwen3-0.6B | 8192 | 16 | 8 | 128 | 1.792 | 6.955 | 1.9722 |

## 5. Bandwidth (bf16 GQA 16/4 × D=128; payload = 2·(Q+K) bytes)

| S | copy ms | fwd liger ms/GBps | fwd eager | bwd liger | bwd eager | liger/eager fwd | liger/eager bwd |
|---|---|---|---|---|---|---|---|
| 256 | 0.0225 | 0.0399/65.6 | 0.1464/17.9 | 0.0256/102.4 | 0.1812/14.5 | 3.67 | 7.08 |
| 1024 | 0.0563 | 0.0952/110.1 | 0.1925/54.5 | 0.0686/152.8 | 0.321/32.7 | 2.02 | 4.68 |
| 4096 | 0.1915 | 0.4424/94.8 | 1.2595/33.3 | 0.2365/177.3 | 1.4418/29.1 | 2.85 | 6.1 |
| 8192 | 0.3799 | 1.2841/65.3 | 3.2123/26.1 | 0.7219/116.2 | 3.882/21.6 | 2.5 | 5.38 |

## 6. Kernel resources

| S | BLOCK | regs | spills | shared |
|---|---|---|---|---|
| 256 | 16 | 32 | 0 | 0 |
| 1024 | 16 | 32 | 0 | 0 |
| 4096 | 16 | 32 | 0 | 0 |
| 8192 | 16 | 32 | 0 | 0 |
