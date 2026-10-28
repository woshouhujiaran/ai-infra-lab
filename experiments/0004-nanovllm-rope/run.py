"""nano-vllm RotaryEmbedding: cached cos/sin table vs on-the-fly cos/sin.

1. Numerics: cos/sin table and bf16 q/k output error vs an fp64 reference,
   bucketed by position (long positions are where fp32 angles drift).
2. Throughput + peak temp memory on the local GPU (eager and compiled).
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from lab import gpu_guard, nanovllm_shim  # noqa: E402

nanovllm_shim.install()
import torch  # noqa: E402
from nanovllm.layers.rotary_embedding import RotaryEmbedding, apply_rotary_emb  # noqa: E402

SEED = 0
# Qwen3-0.6B attention: 16 q heads, 8 kv heads, head_dim 128, rope_theta 1e6.
HEAD_DIM = 128
NUM_HEADS = 16
NUM_KV_HEADS = 8
BASE = 1_000_000.0
MAX_POS = 40960
POS_BUCKETS = [(0, 1024), (1024, 8192), (8192, 32768), (32768, MAX_POS)]
TOKEN_COUNTS = [1, 16, 256, 2048, 8192]


def _orig(method):
    return getattr(method, "_torchdynamo_orig_callable", method)


def cos_sin_fp64(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (BASE ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64,
                                            device=positions.device) / HEAD_DIM))
    freqs = positions.double()[:, None] * inv_freq[None, :]
    return freqs.cos(), freqs.sin()


def fp64_cache_rope() -> RotaryEmbedding:
    """Same module, but the table is built in fp64 and stored as fp32."""
    rope = RotaryEmbedding(HEAD_DIM, HEAD_DIM, MAX_POS, BASE)
    cos, sin = cos_sin_fp64(torch.arange(MAX_POS))
    rope.cos_sin_cache = torch.cat((cos, sin), dim=-1).float().unsqueeze_(1)
    return rope


class OnTheFly(torch.nn.Module):
    """Recompute cos/sin from positions every call; keeps only inv_freq."""

    def __init__(self):
        super().__init__()
        inv_freq = 1.0 / (BASE ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float) / HEAD_DIM))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions, query, key):
        freqs = positions.float()[:, None] * self.inv_freq[None, :]
        cos, sin = freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)
        return apply_rotary_emb(query, cos, sin), apply_rotary_emb(key, cos, sin)


def ref_rope(positions, x):
    cos, sin = cos_sin_fp64(positions)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    x1, x2 = torch.chunk(x.double(), 2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


def make_inputs(tokens: int, g: torch.Generator, positions: torch.Tensor | None = None):
    if positions is None:
        positions = torch.randint(0, MAX_POS, (tokens,), device="cuda", generator=g)
    q = torch.randn(tokens, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(tokens, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=g)
    return positions, q, k


def table_numerics(cached32: RotaryEmbedding, cached64: RotaryEmbedding, otf: OnTheFly):
    """Max abs error of the cos/sin values each method feeds into the rotation."""
    pos = torch.arange(MAX_POS, device="cuda")
    ref_cos, ref_sin = cos_sin_fp64(pos)
    tables = {}
    for name, rope in (("cached_fp32", cached32), ("cached_fp64_built", cached64)):
        cos, sin = rope.cos_sin_cache[pos].squeeze(1).chunk(2, dim=-1)
        tables[name] = (cos, sin)
    freqs = pos.float()[:, None] * otf.inv_freq[None, :]
    tables["on_the_fly_fp32"] = (freqs.cos(), freqs.sin())

    rows = []
    for name, (cos, sin) in tables.items():
        err = torch.maximum((cos.double() - ref_cos).abs(), (sin.double() - ref_sin).abs())
        for lo, hi in POS_BUCKETS:
            rows.append({
                "method": name,
                "positions": f"[{lo}, {hi})",
                "max_abs": float(f"{err[lo:hi].max().item():.3g}"),
                "mean_abs": float(f"{err[lo:hi].mean().item():.3g}"),
            })
    return rows


def output_numerics(methods: dict, tokens_per_bucket: int = 512):
    """bf16 q/k after rotation vs fp64 reference; mismatch = differs from ref rounded to bf16."""
    g = torch.Generator(device="cuda").manual_seed(SEED)
    rows = []
    for lo, hi in POS_BUCKETS:
        pos = torch.randint(lo, hi, (tokens_per_bucket,), device="cuda", generator=g)
        _, q, k = make_inputs(tokens_per_bucket, g, pos)
        ref_q, ref_k = ref_rope(pos, q), ref_rope(pos, k)
        ref_q16, ref_k16 = ref_q.to(torch.bfloat16), ref_k.to(torch.bfloat16)
        for name, fn in methods.items():
            oq, ok = fn(pos, q.clone(), k.clone())
            err = torch.cat([(oq.double() - ref_q).flatten(), (ok.double() - ref_k).flatten()]).abs()
            mism = torch.cat([(oq != ref_q16).flatten(), (ok != ref_k16).flatten()]).float().mean()
            rows.append({
                "method": name,
                "positions": f"[{lo}, {hi})",
                "max_abs": round(err.max().item(), 5),
                "mean_abs": float(f"{err.mean().item():.3g}"),
                "bf16_mismatch_pct": round(mism.item() * 100, 3),
            })
    return rows


def try_compiled(cached32: RotaryEmbedding, otf: OnTheFly):
    try:
        import triton  # noqa: F401
    except ImportError:
        return None, "triton not installed (torch.compile needs it on Windows)"
    otf_c = torch.compile(otf)
    try:
        t0 = time.perf_counter()
        g = torch.Generator(device="cuda").manual_seed(SEED)
        for tokens in (2, 3):
            args = make_inputs(tokens, g)
            cached32(*args)
            otf_c(*args)
        torch.cuda.synchronize()
        compiled = {"cached_fp32_compiled": cached32, "on_the_fly_compiled": otf_c}
        return compiled, f"first compile took {time.perf_counter() - t0:.1f} s"
    except Exception as e:  # noqa: BLE001
        return None, f"torch.compile failed: {type(e).__name__}: {str(e)[:200]}"


def benchmark(fn, tokens: int, iters: int = 100, warmup: int = 10):
    g = torch.Generator(device="cuda").manual_seed(SEED)
    args = make_inputs(tokens, g)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    return {
        "tokens": tokens,
        "ms_per_call": round(ms, 4),
        "tokens_per_s": round(tokens / ms * 1e3),
        "extra_peak_mib": round((torch.cuda.max_memory_allocated() - base) / 2**20, 2),
    }


def main():
    gpu = gpu_guard.acquire()
    torch.manual_seed(SEED)

    cached32 = RotaryEmbedding(HEAD_DIM, HEAD_DIM, MAX_POS, BASE).cuda()
    cached64 = fp64_cache_rope().cuda()
    otf = OnTheFly().cuda()
    cache_mib = cached32.cos_sin_cache.numel() * cached32.cos_sin_cache.element_size() / 2**20

    eager = {
        "cached_fp32_eager": lambda p, q, k: _orig(cached32.forward)(cached32, p, q, k),
        "cached_fp64_built_eager": lambda p, q, k: _orig(cached64.forward)(cached64, p, q, k),
        "on_the_fly_eager": otf,
    }
    compiled, compile_note = try_compiled(cached32, otf)
    methods = {**eager, **(compiled or {})}

    tables = table_numerics(cached32, cached64, otf)
    outputs = output_numerics(methods)
    perf = [{"method": name, **benchmark(fn, t)} for t in TOKEN_COUNTS for name, fn in methods.items()]

    result = {
        "experiment": HERE.name,
        "upstream": "GeeeekExplorer/nano-vllm",
        "upstream_commit": nanovllm_shim.upstream_commit(),
        "component": "nanovllm.layers.rotary_embedding.RotaryEmbedding",
        "head_dim": HEAD_DIM,
        "num_heads": NUM_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "rope_theta": BASE,
        "max_position": MAX_POS,
        "cos_sin_cache_mib": round(cache_mib, 2),
        "gpu": gpu,
        "compile_note": compile_note,
        "table_numerics": tables,
        "output_numerics": outputs,
        "throughput": perf,
        "peak_memory_mib": gpu_guard.peak_memory_mib(),
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        f"cos/sin cache: {MAX_POS} positions x {HEAD_DIM} fp32 = {cache_mib:.2f} MiB "
        f"(shared by all layers via get_rope's lru_cache)",
        "",
        "## cos/sin values vs fp64 reference",
        "",
        "| method | positions | max abs | mean abs |",
        "|---|---|---|---|",
    ]
    lines += [f"| {r['method']} | {r['positions']} | {r['max_abs']} | {r['mean_abs']} |" for r in tables]
    lines += [
        "",
        "## Rotated q/k (bf16) vs fp64 reference",
        "",
        "| method | positions | max abs | mean abs | % elements != ref rounded to bf16 |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['method']} | {r['positions']} | {r['max_abs']} | {r['mean_abs']} | {r['bf16_mismatch_pct']} |"
        for r in outputs
    ]
    lines += [
        "",
        f"## Throughput (bf16 q {NUM_HEADS}x{HEAD_DIM}, k {NUM_KV_HEADS}x{HEAD_DIM}, random positions)",
        "",
        "| method | tokens | ms/call | tokens/s | extra peak MiB |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['method']} | {r['tokens']} | {r['ms_per_call']} | {r['tokens_per_s']} | {r['extra_peak_mib']} |"
        for r in perf
    ]
    if compile_note:
        prefix = "compiled variants" if compiled else "compiled variants skipped"
        lines += ["", f"{prefix}: {compile_note}"]
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
