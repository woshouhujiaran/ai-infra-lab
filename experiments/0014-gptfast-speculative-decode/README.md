# 0014 — gpt-fast speculative decoding on TinyStories Llamas

- **Upstream**: [meta-pytorch/gpt-fast](https://github.com/meta-pytorch/gpt-fast) (submodule `upstream/gpt-fast` at `6ecad9b`). `generate.py` and `model.py` are imported unmodified. `run.py` only swaps module attributes (`decode_n_tokens`, `model_forward`, `create_block_mask` for the toy models, `model.flex_attention` for timing) and replaces `generate()`'s speculative loop with a copy that fixes its indexing (see finding 1).
- **Component**: `speculative_decode(model, draft_model, cur_token, input_pos, speculate_k)`:
  - The draft decodes k tokens one at a time. The target then scores `[cur_token, d1..dk]` in a single forward pass.
  - Draft token i is accepted with probability `min(1, q_i / p_i)`. On the first rejection, a replacement is sampled from `normalize(max(q - p, 0))`.
  - If all k are accepted, a bonus token is sampled from the target, and `d_k` is fed to the draft so its KV cache stays complete.
- **Models**: [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas) llama2.c checkpoints, all with the Llama-2 32k vocab, loaded into gpt-fast's `Transformer` by `run.py`:
  - drafts: stories15M (6 layers, dim 288, 24.4M params incl. embeddings) and stories42M (8 layers, dim 512, 58.1M);
  - target: stories110M (12 layers, dim 768, 134.1M).

  Plain decoding produces coherent stories, e.g. "Once upon a time, there was a dog named Max. Max was very loyal to his owner, a little girl named Lily…" (full samples in [results.md](results.md)).
- **Hardware**: RTX 4060 Laptop 8 GB (through `lab/gpu_guard`, peak 1.1 GiB allocated), i9-13900H, torch 2.14.1+cu126, triton-windows.
- **Run**:

  ```powershell
  $d = "$env:USERPROFILE\.cache\ai-infra-lab\tinyllamas"; mkdir $d -Force
  foreach ($n in "stories15M","stories42M","stories110M") { curl.exe -L -o "$d\$n.bin" "https://huggingface.co/karpathy/tinyllamas/resolve/main/$n.bin" }
  curl.exe -L -o "$d\tokenizer.bin" https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin
  .venv\Scripts\python.exe experiments\0014-gptfast-speculative-decode\run.py
  ```

  The download is 666 MB. The run takes about 15 minutes.

## Question

Does gpt-fast's speculative decoding path work at HEAD, and does `speculative_decode` sample exactly from the target? How many tokens per target call does a 15M or 42M draft buy a 110M target, and does the textbook `(1 - a^(k+1)) / (1 - a)` predict it? Does that turn into wall-clock speedup on a laptop GPU?

## Setup

1. **Upstream as-is**: `generate(target, prompt, 32, draft_model=stories15M, speculate_k=4)`. Each failure is shimmed in turn to find the next one.
2. **Verify mask**: since the flex-attention rewrite, `Transformer.forward(mask, idx, input_pos)` needs a `BlockMask`, and it shifts `mask_mod` by `input_pos[0]` itself. Two ways to build the mask for the target's verify forward (9 queries at offset `off`):
   - **prefill-style**: `create_block_mask(causal, S, kv_len)`, as `prefill` does at offset 0;
   - **offset-aware**: block sparsity built from `q + off >= kv`, with an unshifted `mask_mod` handed to `forward`.

   Both are compared with 9 sequential `decode_one_token` calls (upstream's decode path) on stories110M in fp32, in eager mode and under `torch.compile`. Offsets are 64, 124 (straddles the 128-token block edge), 200 and 380.
3. **Exactness** (CPU, toy models). Upstream `speculative_decode` runs on trigram "models": logits depend on the previous two tokens, read from a token buffer written at `input_pos` that stands in for the KV cache.
   - Setup: vocabulary 8, target logits 2·N(0,1), draft = target + N(0,1) noise, prompt (3, 5).
   - Test: the joint distribution of generated tokens 2–4 (512 cells) over 20,000 runs, compared with the exact target distribution by chi-square and total variation (TV). Reference rows: target-only sampling (the noise floor) and draft-only sampling (what a broken acceptance rule would look like).
   - Variants: k = 1 and 3; temperature 1.0, and 0.7 with top-k 3; with and without the draft-cache fill. A fill ablation over 24 tokens; greedy decoding for all 64 prompts.
4. **Acceptance on real models** (bf16): 4 TinyStories prompts of 16 tokens (sampled from the target), 200 new tokens each.
   - Grid: k ∈ {1, 2, 4, 8} × greedy / temperature 0.8 + top-k 200 (gpt-fast's defaults) / temperature 1.0.
   - Prediction: the per-position acceptance `a_t = Σ min(p_t, q_t)` comes from full-sequence forward passes of both models over the k = 4 outputs. It is plugged into the iid formula with the mean `a`, and into a position-wise version `1 + Σ_j Π_{i<j} a_{t+i}`.
   - Self-drafting (draft = target) is run in bf16 and fp32.
5. **Speed**: tokens/s against plain `decode_n_tokens`, and median per-step costs at position 100. Eager `flex_attention` falls back to an unfused, vmap-based reference path (about 20 tok/s for stories110M). For parts 4–5 only the `flex_attention` op is therefore `torch.compile`d; the rest of the model and the decode loop stay eager, and gpt-fast's whole-step `reduce-overhead` compile is not used.

## Findings

1. **At HEAD, `generate(draft_model=...)` cannot run: it hits three bugs in a row**. Plain `generate()` works.
   - `generate.py:120` raises `Tensors must have same number of dimensions: got 1 and 2`. `decode_one_token` now returns `[1, 1]` tokens and `[1, V]` probabilities, while `speculative_decode` still assumes `[1]` and `[V]` (run.py flattens them in a `decode_n_tokens` wrapper).
   - `model.py:151` raises `'Tensor' object has no attribute 'mask_mod'`. `model_forward(model, x, input_pos)` still calls the pre-flex-attention signature, so the token tensor lands in the `mask` slot.
   - `generate.py:213` raises a shape error. `seq[input_pos + 1 : …] = next_tokens` slices the batch dimension of a `[B, T]` tensor. When a call returns exactly 1 token, the assignment broadcasts into an empty slice and drops the token silently; it only errors once 2 or more come back.
2. **The obvious mask fix is silently wrong under `torch.compile`.**
   - The prefill-style verify mask matches sequential decoding in eager mode (1.4e-5 logit error at every offset). Eager flex attention applies `mask_mod` densely and ignores block sparsity.
   - Compiled, every query at position 128 or later loses every key past the first 128-token block: max logit error 2.3–2.7, TV up to 0.24 at temperature 1, and a changed argmax at 2 of the 3 offsets that reach past 128. The block sparsity was computed for queries 0..k, and shifting `mask_mod` afterwards cannot add key blocks back.
   - So with gpt-fast's intended `compile=True`, this fix would bias the target distribution with no error raised.
   - The offset-aware mask matches in both modes (≤ 1.5e-5). The compiled sequential decode reference agrees with the eager one to ≤ 1.3e-5.
3. **Once those are fixed, the rejection sampler is exact.**
   - In all six toy configurations, speculative output is as close to the target as target-only sampling is: TV 0.035–0.040 (target-only 0.039) and 0.018–0.020 (target-only 0.012) for 20k runs. Draft-only sampling sits at TV 0.49–0.65 with p ≈ 0.
   - The lowest p-value was 0.032–0.050, for temperature 0.7, top-k 3, k = 1. A separate 100k-run check of that config with another seed (not part of run.py) gave p = 0.40 and TV 0.0076.
   - Greedy speculative decoding reproduced target argmax decoding for all 64 prompts at k = 1 and 3.
4. **The draft-cache fill matters only for speed.** Without it the output stays exact, but the draft's next step reads a stale cache slot. Over 24 tokens, tokens per call drop from 1.72 to 1.53 at k = 1 and from 2.63 to 2.39 at k = 3 (−11% and −9%).
5. **The textbook formula predicts acceptance well.**
   - Mean per-token acceptance against the 110M target is 0.84–0.88 for stories15M and 0.89–0.92 for stories42M. Greedy is the highest; temperature 0.8 / top-k 200 is the lowest for stories15M.
   - Measured tokens per target call: 1.83–1.91 at k = 1, 3.53–4.34 at k = 4, 4.87–6.69 at k = 8.
   - The iid formula with the mean acceptance is within 3% of the measurement in 21 of 24 cells. The position-wise version is no better (20 of 24), even though 3–11% of positions have `a < 0.5`.
   - All three misses are at k = 8, where a call's token count varies most: −4% for stories15M at temperature 1.0, and +7% / +20% for stories42M at temperature 0.8 / 1.0 (4.87 measured vs 5.84 predicted). These cells have only 120–164 calls from 4 prompts, and the acceptance statistics come from the k = 4 outputs, not the k = 8 ones.
6. **Self-drafting in bf16 is not lossless.** With draft = target at k = 4, fp32 accepts every token (5.000 tokens per call, 0 of 160 calls with a rejection). bf16 rejects in 1–3 of about 160 calls (4.975–4.994). The k-token batched verify and the one-token decode round differently, so `q / p` falls slightly below 1. Rejections still resample exactly from the target computed by the verify pass.
7. **On this laptop speculative decoding is always slower: 0.66–0.98× plain decoding (best 0.98×, with stories15M at k = 2–4).**
   - The models run at small batch and are launch-bound. A stories110M decode step costs 16.7 ms, and verifying 9 tokens costs the same (16.6 ms), so verification is essentially free.
   - But a draft step is not cheap. stories15M costs 9.3 ms (55% of the target with 18% of its parameters) and stories42M 11.2 ms (67%). Cost tracks layer count (6 and 8 vs 12), not parameters.
   - Count `k · t_draft + t_verify`, plus the fill step on all-accepted calls. With no overhead, the best possible speedup is 1.11× (stories15M, greedy, k = 4), and at most 1.05× for stories42M. At k = 1 it is 0.80–0.89×.
   - The measured per-call time is a further 0–25 ms above that model, growing with k. Per-step Python work in `decode_n_tokens` and `speculative_decode` (BlockMask slicing, probability clones, `.item()` / `nonzero` host syncs) eats the remaining margin.
   - Building the offset-aware mask eagerly costs 2.2 ms per call, so `run.py` caches it per offset.

Takeaway:
- gpt-fast's speculative path has been broken since the flex-attention rewrite. The natural fix (reuse the prefill mask) is correct in eager mode and silently wrong once compiled past position 128. The verify forward needs a mask whose block sparsity is built at the real offset.
- With that fixed, `speculative_decode`'s rejection sampling is exact, and `(1 - a^(k+1)) / (1 - a)` with the mean acceptance predicts its tokens per call to a few percent.
- Whether that becomes speed depends on draft cost in the launch-bound regime. There, cost scales with layer count, so a 6-layer draft for a 12-layer target cannot pay off without whole-step compilation / CUDA graphs (gpt-fast's `reduce-overhead` mode) and a much shallower draft.

Repeatability: parts 1–3 gave identical numbers in two runs (the first run was stopped during part 4 to switch to compiled `flex_attention`). Parts 4–5 come from a single run. Their tokens-per-call numbers are seeded, but tokens/s will move by a few percent between runs.

Full numbers: [results.md](results.md) / [results.json](results.json).
