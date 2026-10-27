"""nano-vllm RMSNorm: fused residual-add vs separate add + norm.

1. Numerics (bf16): max abs / relative error vs an fp32 reference.
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
from nanovllm.layers.layernorm import RMSNorm  # noqa: E402

SEED = 0
EPS = 1e-6
# Qwen3-8B-class hidden size; tokens = batch * seq for decode / short prefill.
HIDDEN = 4096
TOKEN_COUNTS = [1, 16, 256, 2048]


def _orig(method):
    return getattr(method, "_torchdynamo_orig_callable", method)


def ref_fp32(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor):
    """Textbook fused residual + RMSNorm in fp32 (no inplace tricks)."""
    y = x.float() + residual.float()
    out_residual = y.to(x.dtype)
    var = y.pow(2).mean(dim=-1, keepdim=True)
    out = (y * torch.rsqrt(var + EPS)).to(x.dtype) * weight
    return out, out_residual


def separate_eager(norm: RMSNorm, x: torch.Tensor, residual: torch.Tensor):
    """Naive path: cast-add in bf16, then the upstream rms_forward body."""
    y = x + residual
    out = _orig(norm.rms_forward)(norm, y)
    return out, y


def fused_eager(norm: RMSNorm, x: torch.Tensor, residual: torch.Tensor):
    return _orig(norm.add_rms_forward)(norm, x, residual)


def make_inputs(tokens: int, device: str, dtype: torch.dtype, g: torch.Generator):
    x = torch.randn(tokens, HIDDEN, device=device, dtype=dtype, generator=g)
    residual = torch.randn(tokens, HIDDEN, device=device, dtype=dtype, generator=g)
    return x, residual


def numerics(device: str = "cuda"):
    g = torch.Generator(device=device).manual_seed(SEED)
    norm = RMSNorm(HIDDEN, eps=EPS).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.normal_(mean=1.0, std=0.02, generator=g)

    rows = []
    for tokens in TOKEN_COUNTS:
        x, residual = make_inputs(tokens, device, torch.bfloat16, g)
        ref_out, ref_res = ref_fp32(x, residual, norm.weight)
        for name, fn in (("fused_eager", fused_eager), ("separate_eager", separate_eager)):
            out, res = fn(norm, x.clone(), residual.clone())
            out_err = (out.float() - ref_out.float()).abs()
            res_err = (res.float() - ref_res.float()).abs()
            scale = ref_out.float().abs().clamp_min(1e-3)
            rows.append({
                "method": name,
                "tokens": tokens,
                "out_max_abs": round(out_err.max().item(), 6),
                "out_mean_abs": round(out_err.mean().item(), 6),
                "out_max_rel": round((out_err / scale).max().item(), 6),
                "residual_max_abs": round(res_err.max().item(), 6),
                "allclose_rtol_1e2_atol_1e2": bool(
                    torch.allclose(out.float(), ref_out.float(), rtol=1e-2, atol=1e-2)
                ),
            })
    return rows


def try_compiled(norm: RMSNorm):
    try:
        import triton  # noqa: F401
    except ImportError:
        return None, None, "triton not installed (torch.compile needs it on Windows)"

    def fused(x, residual):
        return norm(x, residual)

    def separate(x, residual):
        y = x + residual
        return norm(y), y

    try:
        t0 = time.perf_counter()
        x = torch.randn(2, HIDDEN, device="cuda", dtype=torch.bfloat16)
        r = torch.randn(2, HIDDEN, device="cuda", dtype=torch.bfloat16)
        fused(x, r)
        separate(x, r)
        torch.cuda.synchronize()
        note = f"first compile took {time.perf_counter() - t0:.1f} s"
        return fused, separate, note
    except Exception as e:  # noqa: BLE001
        return None, None, f"torch.compile failed: {type(e).__name__}: {str(e)[:200]}"


def benchmark(fn, tokens: int, iters: int = 50, warmup: int = 10):
    g = torch.Generator(device="cuda").manual_seed(SEED)
    x, residual = make_inputs(tokens, "cuda", torch.bfloat16, g)
    for _ in range(warmup):
        fn(x, residual)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(x, residual)
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

    norm = RMSNorm(HIDDEN, eps=EPS).cuda().to(torch.bfloat16)
    with torch.no_grad():
        g = torch.Generator(device="cuda").manual_seed(SEED)
        norm.weight.normal_(mean=1.0, std=0.02, generator=g)

    methods = {
        "fused_eager": lambda x, r: fused_eager(norm, x, r),
        "separate_eager": lambda x, r: separate_eager(norm, x, r),
    }
    fused_c, separate_c, compile_note = try_compiled(norm)
    if fused_c is not None:
        methods["fused_compiled"] = fused_c
        methods["separate_compiled"] = separate_c

    num = numerics()
    perf = [{"method": name, **benchmark(fn, t)} for t in TOKEN_COUNTS for name, fn in methods.items()]

    result = {
        "experiment": HERE.name,
        "upstream": "GeeeekExplorer/nano-vllm",
        "upstream_commit": nanovllm_shim.upstream_commit(),
        "component": "nanovllm.layers.layernorm.RMSNorm",
        "hidden_size": HIDDEN,
        "eps": EPS,
        "dtype": "bfloat16",
        "gpu": gpu,
        "compile_note": compile_note,
        "numerics": num,
        "throughput": perf,
        "peak_memory_mib": gpu_guard.peak_memory_mib(),
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        f"## Numerics (bf16 vs fp32 reference, hidden={HIDDEN})",
        "",
        "| method | tokens | out max abs | out mean abs | out max rel | residual max abs | allclose 1e-2 |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['method']} | {r['tokens']} | {r['out_max_abs']} | {r['out_mean_abs']} | "
        f"{r['out_max_rel']} | {r['residual_max_abs']} | {r['allclose_rtol_1e2_atol_1e2']} |"
        for r in num
    ]
    lines += [
        "",
        f"## Throughput (bf16, hidden={HIDDEN})",
        "",
        "| method | tokens | ms/call | tokens/s | extra peak MiB |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['method']} | {r['tokens']} | {r['ms_per_call']} | {r['tokens_per_s']} | {r['extra_peak_mib']} |"
        for r in perf
    ]
    if compile_note:
        prefix = "compiled variants" if fused_c is not None else "compiled variants skipped"
        lines += ["", f"{prefix}: {compile_note}"]
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
