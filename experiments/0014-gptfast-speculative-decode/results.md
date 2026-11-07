# 0014 results

## 1. Upstream generate(draft_model=...)

| attempt | result | where |
|---|---|---|
| generate(draft_model=None) | ok |  |
| as-is | RuntimeError: Tensors must have same number of dimensions: got 1 and 2 | `generate.py:120 torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),` |
| + flat decode_n_tokens | AttributeError: 'Tensor' object has no attribute 'mask_mod' | `model.py:151 mask.mask_mod = self.get_mask_mod(mask.mask_mod, input_pos[0])` |
| + flat decode_n_tokens + offset verify mask | RuntimeError: The expanded size of the tensor (48) must match the existing size (5) at non-singleton dimension 1.  Target sizes: [0, 48].  Tensor sizes: [5] | `generate.py:213 seq[input_pos + 1 : input_pos + num_added + 1] = next_tokens[: num_added]` |

## 2. Verify-forward mask vs sequential decode (stories110M fp32, 9 queries)

| mode | positions | prefill-style max logit err | wrong positions | max TV (T=1) | argmax changed | offset-aware max err |
|---|---|---|---|---|---|---|
| eager | 64–72 | 1.43e-05 | 0/9 | 0.000 | 0 | 1.43e-05 |
| eager | 124–132 | 1.42e-05 | 0/9 | 0.000 | 0 | 1.42e-05 |
| eager | 200–208 | 1.55e-05 | 0/9 | 0.000 | 0 | 1.55e-05 |
| eager | 380–388 | 1.68e-05 | 0/9 | 0.000 | 0 | 1.68e-05 |
| compiled | 64–72 | 1.53e-05 | 0/9 | 0.000 | 0 | 1.53e-05 |
| compiled | 124–132 | 2.68e+00 | 5/9 | 0.166 | 1 | 1.53e-05 |
| compiled | 200–208 | 2.64e+00 | 9/9 | 0.240 | 1 | 1.36e-05 |
| compiled | 380–388 | 2.34e+00 | 9/9 | 0.145 | 0 | 1.34e-05 |

## 3. Toy trigram exactness (V=8, 20000 trials, joint of 3 tokens)

| sampling | method | chi2 / df | p | TV | tokens/call | draft fills |
|---|---|---|---|---|---|---|
| t1.0 | target-only | 238.9 / 245 | 0.598 | 0.0391 | nan |  |
| t1.0 | draft-only | 74109.6 / 245 | 0.000 | 0.4937 | nan |  |
| t1.0 | speculative k=1 | 243.2 / 245 | 0.521 | 0.0381 | 1.747 | 30671 |
| t1.0 | speculative k=3 | 236.9 / 245 | 0.633 | 0.0401 | 2.691 | 11937 |
| t1.0 | speculative k=1 no fill | 228.2 / 245 | 0.772 | 0.0346 | 1.602 | 24729 |
| t0.7-top3 | target-only | 44.1 / 53 | 0.804 | 0.0119 | nan |  |
| t0.7-top3 | draft-only | 371102.6 / 53 | 0.000 | 0.6515 | nan |  |
| t0.7-top3 | speculative k=1 | 71.0 / 53 | 0.050 | 0.0199 | 1.799 | 32304 |
| t0.7-top3 | speculative k=3 | 57.5 / 53 | 0.312 | 0.0176 | 2.637 | 10569 |
| t0.7-top3 | speculative k=1 no fill | 73.7 / 53 | 0.032 | 0.0188 | 1.605 | 24457 |

Draft-cache fill ablation (t1.0, 24 new tokens, 1000 trials):

| k | fill | tokens/call | all-accepted calls |
|---|---|---|---|
| 1 | upstream | 1.722 | 0.722 |
| 1 | dropped | 1.531 | 0.531 |
| 3 | upstream | 2.630 | 0.374 |
| 3 | dropped | 2.387 | 0.316 |

- greedy k=1: 0 / 64 sequences differ from target argmax decoding
- greedy k=3: 0 / 64 sequences differ from target argmax decoding

## 4. Acceptance on TinyStories (target stories110M, bf16, compiled flex_attention)

| draft | sampling | k | tokens/call | pred. iid (mean a) | pred. position-wise | mean a | tok/s | speedup vs plain |
|---|---|---|---|---|---|---|---|---|
| stories15M | greedy | 1 | 1.91 | 1.88 | 1.88 | 0.883 | 49.0 | 0.82× |
| stories15M | greedy | 2 | 2.71 | 2.66 | 2.66 | 0.883 | 56.7 | 0.95× |
| stories15M | greedy | 4 | 3.90 | 3.96 | 3.98 | 0.883 | 55.8 | 0.93× |
| stories15M | greedy | 8 | 5.87 | 5.76 | 5.87 | 0.883 | 51.3 | 0.86× |
| stories15M | t0.8-top200 | 1 | 1.86 | 1.84 | 1.84 | 0.836 | 49.5 | 0.83× |
| stories15M | t0.8-top200 | 2 | 2.54 | 2.53 | 2.54 | 0.836 | 56.9 | 0.95× |
| stories15M | t0.8-top200 | 4 | 3.53 | 3.61 | 3.65 | 0.836 | 53.6 | 0.90× |
| stories15M | t0.8-top200 | 8 | 5.02 | 4.88 | 5.12 | 0.836 | 42.5 | 0.71× |
| stories15M | t1.0 | 1 | 1.83 | 1.84 | 1.84 | 0.842 | 46.9 | 0.84× |
| stories15M | t1.0 | 2 | 2.58 | 2.55 | 2.56 | 0.842 | 54.8 | 0.98× |
| stories15M | t1.0 | 4 | 3.58 | 3.65 | 3.71 | 0.842 | 54.7 | 0.98× |
| stories15M | t1.0 | 8 | 5.19 | 4.98 | 5.27 | 0.842 | 49.8 | 0.90× |
| stories42M | greedy | 1 | 1.91 | 1.92 | 1.92 | 0.921 | 49.1 | 0.82× |
| stories42M | greedy | 2 | 2.74 | 2.77 | 2.77 | 0.921 | 58.0 | 0.97× |
| stories42M | greedy | 4 | 4.34 | 4.27 | 4.26 | 0.921 | 58.1 | 0.97× |
| stories42M | greedy | 8 | 6.69 | 6.61 | 6.63 | 0.921 | 52.7 | 0.88× |
| stories42M | t0.8-top200 | 1 | 1.87 | 1.91 | 1.91 | 0.905 | 46.0 | 0.77× |
| stories42M | t0.8-top200 | 2 | 2.70 | 2.73 | 2.73 | 0.905 | 52.4 | 0.88× |
| stories42M | t0.8-top200 | 4 | 4.14 | 4.14 | 4.16 | 0.905 | 51.1 | 0.86× |
| stories42M | t0.8-top200 | 8 | 5.87 | 6.25 | 6.33 | 0.905 | 44.4 | 0.74× |
| stories42M | t1.0 | 1 | 1.88 | 1.89 | 1.89 | 0.887 | 48.1 | 0.86× |
| stories42M | t1.0 | 2 | 2.62 | 2.67 | 2.68 | 0.887 | 50.6 | 0.91× |
| stories42M | t1.0 | 4 | 3.96 | 3.99 | 4.02 | 0.887 | 46.6 | 0.84× |
| stories42M | t1.0 | 8 | 4.87 | 5.84 | 5.99 | 0.887 | 36.8 | 0.66× |

| plain decoding | tok/s |
|---|---|
| greedy | 59.9 |
| t0.8-top200 | 59.6 |
| t1.0 | 55.7 |

### Self-drafting (draft = target, k = 4)

| precision | sampling | tokens/call | calls with a rejection |
|---|---|---|---|
| bf16 | greedy | 4.988 | 1 / 161 |
| bf16 | t0.8-top200 | 4.975 | 3 / 161 |
| bf16 | t1.0 | 4.994 | 1 / 160 |
| fp32 | greedy | 5.000 | 0 / 160 |
| fp32 | t0.8-top200 | 5.000 | 0 / 160 |
| fp32 | t1.0 | 5.000 | 0 / 160 |

## 5. bf16 per-step costs, compiled flex_attention (position 100, median)

| model | params | layers | decode step ms | other |
|---|---|---|---|---|
| stories15M | 24.4M | 6 | 9.26 |  |
| stories42M | 58.1M | 8 | 11.19 |  |
| stories110M | 134.1M | 12 | 16.74 | verify_2_ms 18.13, verify_3_ms 15.41, verify_5_ms 16.43, verify_9_ms 16.63, offset_mask_build_ms 2.17 |

## Samples (plain decoding, prompt 0)

- greedy: 'Once upon a time, there was a dog named Max. Max was very loyal to his owner, a little girl named Lily. Max loved to play with Lily and follow her everywhere she went. One day, Lily and Max went for a walk in the park. They saw a big tree with a lot of leaves. Lily said, "Max, let\'s count how many leaves are on the tree!" Max barked happily and wagged his tail. They started counting the leaves, bu'
- t0.8-top200: 'Once upon a time, there was a dog named Max. Max was very obedient and always did what his owner said. One day, Max\'s owner asked him to perform a trick. Max was excited and barked happily.\nMax\'s owner said, "Max, can you stand on your paw?" Max tried and tried, but he couldn\'t do it. His owner said, "Don\'t worry Max, just keep trying and you will get it."\nMax tried again and this time he was able'
- t1.0: "Once upon a time, there was a dog named Max. Max was very obedient and always did what his owner said. One day, Max's owner asked him to perform a trick. Max stood up on his hind legs and wagged his tail. His owner was very impressed with Max's trick.\nLater that day, Max and his owner went for a walk in the park. Max saw a big puddle and wanted to jump in it. His owner told him not to because he w"
