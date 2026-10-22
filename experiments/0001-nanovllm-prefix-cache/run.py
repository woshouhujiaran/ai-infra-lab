"""Prefix-cache hit rate of nano-vllm's paged KV-cache BlockManager.

Drives the real `Scheduler` + `BlockManager` with synthetic token ids (no model,
no GPU) and counts how many prompt tokens actually need prefill compute.
"""
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from lab import nanovllm_shim  # noqa: E402

nanovllm_shim.install()
from nanovllm.engine.scheduler import Scheduler  # noqa: E402
from nanovllm.engine.sequence import Sequence  # noqa: E402
from nanovllm.sampling_params import SamplingParams  # noqa: E402

VOCAB = 151936
SEED = 0


def make_prompts(n: int, prefix_len: int, suffix_len: int, rng: random.Random) -> list[list[int]]:
    prefix = [rng.randrange(VOCAB) for _ in range(prefix_len)]
    return [prefix + [rng.randrange(VOCAB) for _ in range(suffix_len)] for _ in range(n)]


def simulate(prompts, block_size, num_blocks, arrival, output_len, max_batched_tokens=16384):
    Sequence.block_size = block_size
    cfg = nanovllm_shim.LiteConfig(kvcache_block_size=block_size, num_kvcache_blocks=num_blocks,
                                   max_num_batched_tokens=max_batched_tokens)
    sched = Scheduler(cfg)
    bm = sched.block_manager
    preemptions = 0
    orig_preempt = sched.preempt

    def counting_preempt(seq):
        nonlocal preemptions
        preemptions += 1
        orig_preempt(seq)

    sched.preempt = counting_preempt
    params = SamplingParams(temperature=1.0, max_tokens=output_len, ignore_eos=True)
    pending = [Sequence(p, params) for p in prompts]
    rng = random.Random(SEED)
    prefill_tokens = steps = peak_used = 0
    t0 = time.perf_counter()
    while pending or not sched.is_finished():
        if arrival == "burst":
            while pending:
                sched.add(pending.pop(0))
        elif pending:
            sched.add(pending.pop(0))
        if sched.is_finished():
            continue
        seqs, is_prefill = sched.schedule()
        if is_prefill:
            prefill_tokens += sum(s.num_scheduled_tokens for s in seqs)
        sched.postprocess(seqs, [rng.randrange(VOCAB) for _ in seqs], is_prefill)
        peak_used = max(peak_used, len(bm.used_block_ids))
        steps += 1
    elapsed = time.perf_counter() - t0

    assert not bm.used_block_ids and len(bm.free_block_ids) == num_blocks, "block leak"
    assert all(b.ref_count == 0 for b in bm.blocks), "dangling ref_count"
    prompt_tokens = sum(len(p) for p in prompts)
    return {
        "prompt_tokens": prompt_tokens,
        "prefill_tokens_computed": prefill_tokens,
        "prefix_hit_rate": round(1 - prefill_tokens / prompt_tokens, 4),
        "steps": steps,
        "peak_used_blocks": peak_used,
        "preemptions": preemptions,
        "sim_seconds": round(elapsed, 3),
    }


def main():
    n_requests, suffix_len, output_len = 64, 128, 32
    rows = []
    for arrival in ["burst", "staggered"]:
        for block_size in [16, 64, 256]:
            for prefix_len in [0, 256, 1024, 2048]:
                prompts = make_prompts(n_requests, prefix_len, suffix_len, random.Random(SEED))
                r = simulate(prompts, block_size, num_blocks=100_000 // block_size,
                             arrival=arrival, output_len=output_len)
                rows.append({"scenario": "no_pressure", "arrival": arrival, "block_size": block_size,
                             "prefix_len": prefix_len, **r})

    for num_blocks in [24, 48, 96]:
        prompts = make_prompts(n_requests, 1024, suffix_len, random.Random(SEED))
        r = simulate(prompts, 256, num_blocks=num_blocks, arrival="burst", output_len=256)
        rows.append({"scenario": "memory_pressure", "arrival": "burst", "block_size": 256,
                     "prefix_len": 1024, "num_blocks": num_blocks, **r})

    result = {
        "experiment": HERE.name,
        "upstream": "GeeeekExplorer/nano-vllm",
        "upstream_commit": nanovllm_shim.upstream_commit(),
        "component": "nanovllm.engine.block_manager.BlockManager (+ Scheduler)",
        "workload": {"n_requests": n_requests, "suffix_len": suffix_len, "output_len": output_len,
                     "seed": SEED},
        "rows": rows,
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    cols = ["scenario", "arrival", "block_size", "prefix_len", "num_blocks", "prompt_tokens",
            "prefill_tokens_computed", "prefix_hit_rate", "steps", "peak_used_blocks", "preemptions"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
