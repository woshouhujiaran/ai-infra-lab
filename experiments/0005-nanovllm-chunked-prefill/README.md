# 0005 — nano-vllm Scheduler: chunked prefill vs `max_num_batched_tokens`

- **Upstream**: [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (pinned by the `upstream/nano-vllm` submodule, commit `bb823b3`, which includes the chunked-prefill refactor)
- **Component**: `nanovllm/engine/scheduler.py` — `Scheduler.schedule` (prefill-first, only the head of the waiting queue may be chunked), driven with the real `BlockManager`
- **Hardware**: CPU only — no model, no GPU; token ids are synthetic and step times come from an analytic cost model
- **Run**: `.venv\Scripts\python.exe experiments\0005-nanovllm-chunked-prefill\run.py` (about 20 s)

## Question

nano-vllm now splits a prompt that does not fit in `max_num_batched_tokens` across several steps. Does a smaller token budget improve time to first token (TTFT) or decode stalls (inter-token latency, ITL), and what does it cost? How would a decode-first mixed-batch policy (Sarathi / vLLM style) change that trade-off?

## Setup

- Real `Scheduler` + `BlockManager`, block size 256, 128 KV blocks (32,768 tokens: about 3.5 GiB of Qwen3-0.6B KV next to the weights on an 8 GB card), `max_num_seqs=512`. Budgets 256 to 16384 tokens.
- Policies:
  - `upstream`: `Scheduler` as is. A step is prefill-only whenever any waiting prompt can be admitted; decodes only run when nothing can be prefilled.
  - `decode_first`: a subclass in `run.py` (the submodule is untouched). Each step decodes every running sequence first, then fills the leftover budget with prefill, chunking only the head like upstream. Uses `postprocess(..., is_prefill=True)`, which is exact for decode rows.
- Simulated clock, step cost in ms (estimates for Qwen3-0.6B bf16 on an RTX 4060 Laptop, not measured): overhead 4.0 for eager prefill/mixed steps and 1.5 for CUDA-graph decode steps, plus `max(4.8 weight read, 0.06 × tokens)`, plus per sequence `max(1.15e-5 × attention pairs, 4.5e-4 × context tokens)`.
- Workloads (seed 0):
  - `hol`: 16 short requests (256 prompt / 256 output) start at 0 ms, one 8,192-token prompt (32 output) arrives at 200 ms, and 8 more short requests (256 / 64) arrive at 400 ms.
  - `poisson_3rps` / `poisson_4.5rps`: 300 requests, Poisson arrivals, 90% short prompts (128–1024 tokens), 10% long (4096–8192), outputs 64–256. 3 req/s is below capacity; 4.5 req/s is overloaded (last arrival 66.8 s, makespan about 81 s).
- TTFT = first token time − arrival; ITL = gap between consecutive tokens of one request (includes preemption/recompute stalls). After each run the script asserts every request produced all its tokens, every block is freed and every `ref_count` is 0.

## Findings

1. **Upstream chunked prefill is a memory knob, not a latency knob.** Prefill steps never carry decodes, so running requests freeze for the whole long prompt no matter how it is split. In `hol` the worst decode stall is 1,024 ms at budget ≥ 8192 and grows to 1,257 ms at 1024 and 1,472 ms at 256, because each extra chunk pays the eager overhead and the weight-read floor. The long prompt's TTFT grows from 937 to 1,121 ms. The only gain is a bounded step size (256 instead of 8,192 tokens per forward), i.e. lower activation memory.
2. **Small upstream budgets hurt TTFT under load.** At 3 req/s, median TTFT is 79–82 ms for budgets ≥ 1024 but 120 ms at 512 and 387 ms at 256 (4.9×). ITL p99 goes from 87 to 145 ms, and the worst stall stays at 2.3–2.7 s for every budget (queued long prompts prefilled back to back, plus preemption recompute). At this load, budgets ≥ 1024 are within 5% of unchunked on every latency metric.
3. **Decode-first mixing makes the budget bound decode stalls.** In `hol` the worst stall drops from 1,024 ms to 44 / 81 / 155 ms at budgets 256 / 512 / 1024. At 3 req/s, ITL p99 is 43 ms at budget 256 (upstream's best is 87 ms) and the worst stall falls from ≥ 2.29 s to 316 ms.
4. **The cost is TTFT, and mid-size budgets are the worst of both.** At budget 256 the long prompt in `hol` waits 1,405 ms (vs 937) and median TTFT at 3 req/s is 254 ms (vs 79). At budget 1024 the trade is mild: median TTFT 100 ms (+25%), p99 TTFT 2,466 ms (4% better than upstream), worst stall 564 ms. Budgets of 2048–4096 give ITL p99 of 206–333 ms, worse than upstream, because decodes ride along in a few 300–550 ms mixed steps per long prompt.
5. **Under overload, decode-first preempts less.** At 4.5 req/s upstream preempts 27–33 times at every budget, while decode-first preempts 7 / 18 / 19 times at budgets 256 / 512 / 1024. Upstream admits every waiting prompt that fits before any decode step, whereas decode-first admits only what the leftover budget allows, so KV blocks fill more slowly. p99 TTFT drops from 14.1 s to 12.9–13.0 s at budgets 512–1024, with makespan within 1%.
6. **Throughput is unaffected.** Makespan varies by less than 1% across policies and budgets at 3 req/s; the choice only moves latency between TTFT and ITL.

Takeaway: with upstream's prefill-first policy, leave `max_num_batched_tokens` at 1024 or more and treat chunking only as a way to cap activation memory. Chunking pays off for latency only when decodes share the step with prefill chunks; a decode-first policy with a budget of about 512–1024 tokens cut worst-case decode stalls 2–13× here for a modest TTFT increase.

Caveats: absolute times come from the cost model, not a GPU run; the conclusions depend mainly on prefill steps being exclusive and on the per-step overhead / weight-read floor. Mixed steps are charged as eager steps. The fixed 128-block KV pool ignores that upstream's warmup sizes activation memory from `max_num_batched_tokens`, so smaller budgets would also leave somewhat more room for KV.

Full numbers: [results.md](results.md) / [results.json](results.json).
