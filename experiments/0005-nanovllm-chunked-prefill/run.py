"""Chunked prefill in nano-vllm's Scheduler: TTFT and decode stalls vs max_num_batched_tokens.

Drives the real `Scheduler` + `BlockManager` (CPU, synthetic token ids, no model) on a
simulated clock. Every engine step is charged by an analytic cost model of Qwen3-0.6B
(bf16) on an RTX 4060 Laptop, so all latencies are proxies, not measurements.

Two policies are compared:
- `upstream`: nano-vllm as is. A step is either prefill-only or decode-only; any waiting
  prompt takes priority, and only the head of the waiting queue may be chunked.
- `decode_first`: a local subclass (no submodule change). Every step first decodes all
  running sequences, then fills the leftover token budget with prefill (head may be chunked).
"""
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from lab import nanovllm_shim  # noqa: E402

nanovllm_shim.install()
from nanovllm.engine.scheduler import Scheduler  # noqa: E402
from nanovllm.engine.sequence import Sequence, SequenceStatus  # noqa: E402
from nanovllm.sampling_params import SamplingParams  # noqa: E402

VOCAB = 151936
SEED = 0
BLOCK_SIZE = 256
NUM_KV_BLOCKS = 128  # 32768 tokens: ~3.5 GiB of Qwen3-0.6B KV (112 KiB/token), what fits next to weights on 8 GB
BUDGETS = [256, 512, 1024, 2048, 4096, 8192, 16384]


@dataclass(frozen=True)
class CostModel:
    """Step time in ms. Constants are estimates for Qwen3-0.6B bf16 on an RTX 4060 Laptop."""
    eager_overhead_ms: float = 4.0         # prefill / mixed steps run eagerly (launch-bound floor)
    graph_overhead_ms: float = 1.5         # decode-only steps replay a CUDA graph
    weight_read_ms: float = 4.8            # 1.2 GB of weights at ~250 GB/s, paid once per step
    linear_ms_per_token: float = 0.06      # ~1.2 GFLOP/token at ~20 TFLOPS effective
    attn_ms_per_pair: float = 1.15e-5      # ~229 kFLOP per (query, key) pair at ~20 TFLOPS
    kv_read_ms_per_token: float = 4.5e-4   # 112 KiB of KV per context token at ~250 GB/s

    def step(self, seqs: list[Sequence], decode_only: bool) -> float:
        tokens = sum(s.num_scheduled_tokens for s in seqs)
        attn = 0.0
        for s in seqs:
            q, ctx = s.num_scheduled_tokens, s.num_cached_tokens
            pairs = q * ctx + q * (q + 1) / 2
            attn += max(self.attn_ms_per_pair * pairs, self.kv_read_ms_per_token * (ctx + q))
        overhead = self.graph_overhead_ms if decode_only else self.eager_overhead_ms
        return overhead + max(self.weight_read_ms, self.linear_ms_per_token * tokens) + attn


class DecodeFirstScheduler(Scheduler):
    """Sarathi-style mixed batches on top of the upstream data structures."""

    def schedule(self) -> tuple[list[Sequence], bool]:
        budget = self.max_num_batched_tokens
        decodes = []
        while self.running and len(decodes) < min(self.max_num_seqs, budget):
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                decodes.append(seq)
        budget -= len(decodes)

        prefills = []
        while self.waiting and len(decodes) + len(prefills) < self.max_num_seqs and budget > 0:
            seq = self.waiting[0]
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if budget < num_tokens and prefills:
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, budget)
            budget -= seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            prefills.append(seq)

        self.running.extendleft(reversed(decodes))
        scheduled = decodes + prefills
        assert scheduled
        # postprocess(is_prefill=True) is exact for decode rows too: they always complete their token.
        return scheduled, bool(prefills)


@dataclass
class Request:
    arrival_ms: float
    prompt_len: int
    output_len: int
    kind: str


def hol_workload() -> list[Request]:
    """16 short chats decoding, one 8K-token prompt lands at 200 ms, 8 more short chats at 400 ms."""
    reqs = [Request(0.0, 256, 256, "short_early") for _ in range(16)]
    reqs.append(Request(200.0, 8192, 32, "long"))
    reqs += [Request(400.0, 256, 64, "short_late") for _ in range(8)]
    return reqs


def poisson_workload(rng: random.Random, n: int = 300, rate_per_s: float = 3.0) -> list[Request]:
    """90% short prompts (128-1024), 10% long (4096-8192); outputs 64-256; Poisson arrivals."""
    t, reqs = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate_per_s) * 1000
        if rng.random() < 0.1:
            reqs.append(Request(t, rng.randint(4096, 8192), rng.randint(64, 256), "long"))
        else:
            reqs.append(Request(t, rng.randint(128, 1024), rng.randint(64, 256), "short"))
    return reqs


def pct(xs, q):
    return round(float(np.percentile(xs, q)), 1) if len(xs) else None


def simulate(requests: list[Request], policy: str, budget: int, cost: CostModel) -> dict:
    Sequence.block_size = BLOCK_SIZE
    cfg = nanovllm_shim.LiteConfig(max_num_batched_tokens=budget, kvcache_block_size=BLOCK_SIZE,
                                   num_kvcache_blocks=NUM_KV_BLOCKS)
    sched = (Scheduler if policy == "upstream" else DecodeFirstScheduler)(cfg)
    bm = sched.block_manager
    preemptions = 0
    orig_preempt = sched.preempt

    def counting_preempt(seq):
        nonlocal preemptions
        preemptions += 1
        orig_preempt(seq)

    sched.preempt = counting_preempt
    tok_rng = random.Random(SEED)
    order = sorted(range(len(requests)), key=lambda i: requests[i].arrival_ms)
    seqs = {}
    for i in order:
        r = requests[i]
        params = SamplingParams(temperature=1.0, max_tokens=r.output_len, ignore_eos=True)
        seqs[i] = Sequence([tok_rng.randrange(VOCAB) for _ in range(r.prompt_len)], params)
    kind_of = {seqs[i].seq_id: requests[i].kind for i in order}
    arrival_of = {seqs[i].seq_id: requests[i].arrival_ms for i in order}

    now, nxt = 0.0, 0
    first_token, last_emit = {}, {}
    gaps = {k: [] for k in set(kind_of.values())}
    steps = {"prefill_only": 0, "decode_only": 0, "mixed": 0}
    chunked_steps = 0
    max_step_ms = max_step_tokens = 0
    wall = time.perf_counter()
    while nxt < len(order) or not sched.is_finished():
        while nxt < len(order) and requests[order[nxt]].arrival_ms <= now:
            sched.add(seqs[order[nxt]])
            nxt += 1
        if sched.is_finished():
            now = requests[order[nxt]].arrival_ms
            continue
        batch, has_prefill = sched.schedule()
        decode_only = all(not s.is_prefill for s in batch)
        mixed = has_prefill and not all(s.is_prefill for s in batch)
        steps["decode_only" if decode_only else "mixed" if mixed else "prefill_only"] += 1
        dt = cost.step(batch, decode_only)
        step_tokens = sum(s.num_scheduled_tokens for s in batch)
        max_step_ms, max_step_tokens = max(max_step_ms, dt), max(max_step_tokens, step_tokens)
        if any(s.num_cached_tokens + s.num_scheduled_tokens < s.num_tokens for s in batch):
            chunked_steps += 1
        before = {s.seq_id: s.num_completion_tokens for s in batch}
        now += dt
        sched.postprocess(batch, [tok_rng.randrange(VOCAB) for _ in batch], has_prefill)
        for s in batch:
            if s.num_completion_tokens == before[s.seq_id]:
                continue
            if s.seq_id not in first_token:
                first_token[s.seq_id] = now
            else:
                gaps[kind_of[s.seq_id]].append(now - last_emit[s.seq_id])
            last_emit[s.seq_id] = now
    sim_seconds = time.perf_counter() - wall

    assert len(first_token) == len(order), "a request never produced a token"
    assert all(seqs[i].num_completion_tokens == requests[i].output_len for i in order)
    assert not bm.used_block_ids and len(bm.free_block_ids) == NUM_KV_BLOCKS, "block leak"
    assert all(b.ref_count == 0 for b in bm.blocks), "dangling ref_count"

    ttft = {}
    for sid, t in first_token.items():
        ttft.setdefault(kind_of[sid], []).append(t - arrival_of[sid])
    all_ttft = [v for vs in ttft.values() for v in vs]
    all_gaps = [g for gs in gaps.values() for g in gs]
    out = {
        "ttft_p50_ms": pct(all_ttft, 50), "ttft_p99_ms": pct(all_ttft, 99),
        "itl_p50_ms": pct(all_gaps, 50), "itl_p99_ms": pct(all_gaps, 99),
        "itl_max_ms": round(max(all_gaps), 1) if all_gaps else None,
    }
    for kind in sorted(ttft):
        out[f"ttft_{kind}_p50_ms"] = pct(ttft[kind], 50)
        out[f"ttft_{kind}_p99_ms"] = pct(ttft[kind], 99)
        out[f"itl_{kind}_max_ms"] = round(max(gaps[kind]), 1) if gaps[kind] else None
    out.update({
        "makespan_s": round(now / 1000, 2),
        **{f"steps_{k}": v for k, v in steps.items()},
        "chunked_steps": chunked_steps,
        "max_step_ms": round(max_step_ms, 1),
        "max_step_tokens": max_step_tokens,
        "preemptions": preemptions,
        "sim_seconds": round(sim_seconds, 2),
    })
    return out


def main():
    cost = CostModel()
    workloads = {
        "hol": hol_workload(),
        "poisson_3rps": poisson_workload(random.Random(SEED), rate_per_s=3.0),
        "poisson_4.5rps": poisson_workload(random.Random(SEED), rate_per_s=4.5),
    }
    rows = []
    for wname, reqs in workloads.items():
        for policy in ["upstream", "decode_first"]:
            for budget in BUDGETS:
                r = simulate(reqs, policy, budget, cost)
                rows.append({"workload": wname, "policy": policy, "budget": budget, **r})
                print(wname, policy, budget, {k: r[k] for k in
                      ["ttft_p50_ms", "ttft_p99_ms", "itl_p99_ms", "itl_max_ms", "makespan_s", "preemptions"]},
                      flush=True)

    result = {
        "experiment": HERE.name,
        "upstream": "GeeeekExplorer/nano-vllm",
        "upstream_commit": nanovllm_shim.upstream_commit(),
        "component": "nanovllm.engine.scheduler.Scheduler (chunked prefill)",
        "seed": SEED,
        "block_size": BLOCK_SIZE,
        "num_kv_blocks": NUM_KV_BLOCKS,
        "cost_model_ms": asdict(cost),
        "workloads": {k: {"n_requests": len(v),
                          "prompt_tokens": sum(r.prompt_len for r in v),
                          "output_tokens": sum(r.output_len for r in v),
                          "last_arrival_s": round(max(r.arrival_ms for r in v) / 1000, 2)}
                      for k, v in workloads.items()},
        "rows": rows,
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = []
    for wname in workloads:
        wrows = [r for r in rows if r["workload"] == wname]
        cols = ["policy", "budget"] + [c for c in wrows[0] if c not in ("workload", "policy", "budget", "sim_seconds")]
        lines += [f"### {wname}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in wrows]
        lines.append("")
    (HERE / "results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
