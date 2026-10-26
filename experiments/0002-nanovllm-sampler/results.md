## Correctness (vocab 32, T=0.7)

| method | samples | chi2/dof | total variation |
|---|---|---|---|
| nanovllm_gumbel_eager | 1998848 | 0.678 | 0.00081 |
| torch_multinomial | 1998848 | 0.841 | 0.00098 |
| nanovllm_gumbel_compiled | 1998848 | 0.67 | 0.00057 |

## Throughput (vocab 151936, bf16 logits)

| method | batch | ms/call | tokens/s | extra peak MiB |
|---|---|---|---|---|
| nanovllm_gumbel_eager | 1 | 0.2162 | 4625 | 1.7 |
| torch_multinomial | 1 | 0.4302 | 2325 | 1.2 |
| nanovllm_gumbel_compiled | 1 | 0.2952 | 3388 | 0.6 |
| nanovllm_gumbel_eager | 16 | 0.4903 | 32635 | 27.8 |
| torch_multinomial | 16 | 0.5626 | 28440 | 18.6 |
| nanovllm_gumbel_compiled | 16 | 0.523 | 30592 | 9.3 |
| nanovllm_gumbel_eager | 64 | 2.3544 | 27183 | 114.0 |
| torch_multinomial | 64 | 2.5335 | 25262 | 76.0 |
| nanovllm_gumbel_compiled | 64 | 1.6158 | 39610 | 38.0 |
| nanovllm_gumbel_eager | 256 | 9.3948 | 27249 | 445.1 |
| torch_multinomial | 256 | 8.958 | 28578 | 296.8 |
| nanovllm_gumbel_compiled | 256 | 4.5153 | 56696 | 148.4 |

compiled variant: first compile took 2.9 s
