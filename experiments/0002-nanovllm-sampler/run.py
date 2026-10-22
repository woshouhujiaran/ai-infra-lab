"""nano-vllm Sampler (exponential-race / Gumbel-max) vs torch.multinomial.

1. Correctness: empirical token frequencies vs softmax(logits / T).
2. Throughput + peak memory at Qwen3 vocab size on the local GPU.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from lab import gpu_guard, nanovllm_shim  # noqa: E402

nanovllm_shim.install()
import torch  # noqa: E402
from nanovllm.layers.sampler import Sampler  # noqa: E402

SEED = 0
QWEN3_VOCAB = 151936


def nanovllm_eager(logits, temperatures):
    fn = getattr(Sampler.forward, "_torchdynamo_orig_callable", Sampler.forward)
    return fn(None, logits, temperatures)


def multinomial(logits, temperatures):
    probs = torch.softmax(logits.float() / temperatures.unsqueeze(1), dim=-1)
    return torch.multinomial(probs, 1).squeeze(1)


def try_compiled():
    try:
        import triton  # noqa: F401
    except ImportError:
        return None, "triton not installed (torch.compile needs it on Windows)"
    sampler = Sampler()
    try:
        x = torch.randn(2, 8, device="cuda")
        sampler(x, torch.ones(2, device="cuda"))
        return (lambda l, t: sampler(l, t)), None
    except Exception as e:  # noqa: BLE001
        return None, f"torch.compile failed: {type(e).__name__}: {str(e)[:200]}"


def check_distribution(method, vocab=32, temperature=0.7, n_samples=2_000_000, rows=8192):
    g = torch.Generator(device="cuda").manual_seed(SEED)
    logits = torch.randn(vocab, device="cuda", generator=g) * 2
    expected = torch.softmax(logits / temperature, dim=-1).double()
    batch = logits.unsqueeze(0).expand(rows, vocab).contiguous()
    temps = torch.full((rows,), temperature, device="cuda")
    counts = torch.zeros(vocab, device="cuda", dtype=torch.float64)
    for _ in range(n_samples // rows):
        counts += torch.bincount(method(batch.clone(), temps), minlength=vocab).double()
    n = counts.sum()
    exp_counts = expected * n
    chi2 = (((counts - exp_counts) ** 2) / exp_counts).sum().item()
    tvd = 0.5 * (counts / n - expected).abs().sum().item()
    return {"vocab": vocab, "temperature": temperature, "samples": int(n.item()),
            "chi2_per_dof": round(chi2 / (vocab - 1), 3), "total_variation": round(tvd, 5)}


def benchmark(method, batch, vocab=QWEN3_VOCAB, iters=50, warmup=10):
    g = torch.Generator(device="cuda").manual_seed(SEED)
    logits = torch.randn(batch, vocab, device="cuda", dtype=torch.bfloat16, generator=g)
    temps = torch.full((batch,), 0.8, device="cuda")
    for _ in range(warmup):
        method(logits, temps)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        method(logits, temps)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    return {"batch": batch, "ms_per_call": round(ms, 4),
            "tokens_per_s": round(batch / ms * 1e3),
            "extra_peak_mib": round((torch.cuda.max_memory_allocated() - base) / 2**20, 1)}


def main():
    gpu = gpu_guard.acquire()
    torch.manual_seed(SEED)
    methods = {"nanovllm_gumbel_eager": nanovllm_eager, "torch_multinomial": multinomial}
    compiled, compile_note = try_compiled()
    if compiled is not None:
        methods["nanovllm_gumbel_compiled"] = compiled

    correctness = {name: check_distribution(fn) for name, fn in methods.items()}
    perf = [{"method": name, **benchmark(fn, b)} for b in [1, 16, 64, 256] for name, fn in methods.items()]

    result = {
        "experiment": HERE.name,
        "upstream": "GeeeekExplorer/nano-vllm",
        "upstream_commit": nanovllm_shim.upstream_commit(),
        "component": "nanovllm.layers.sampler.Sampler",
        "gpu": gpu,
        "compile_note": compile_note,
        "correctness": correctness,
        "throughput": perf,
        "peak_memory_mib": gpu_guard.peak_memory_mib(),
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = ["## Correctness (vocab 32, T=0.7)", "", "| method | samples | chi2/dof | total variation |",
             "|---|---|---|---|"]
    lines += [f"| {k} | {v['samples']} | {v['chi2_per_dof']} | {v['total_variation']} |"
              for k, v in correctness.items()]
    lines += ["", f"## Throughput (vocab {QWEN3_VOCAB}, bf16 logits)", "",
              "| method | batch | ms/call | tokens/s | extra peak MiB |", "|---|---|---|---|---|"]
    lines += [f"| {r['method']} | {r['batch']} | {r['ms_per_call']} | {r['tokens_per_s']} | {r['extra_peak_mib']} |"
              for r in perf]
    if compile_note:
        lines += ["", f"compiled variant skipped: {compile_note}"]
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
