"""bitsandbytes Linear8bitLt / Linear4bit vs fp16 nn.Linear.

1. Resident weight memory on the GPU per layer shape.
2. Weight and output error vs an fp64 reference, with plain Gaussian activations
   and with a few large-magnitude "outlier" channels (the LLM.int8() case),
   plus Gaussian vs heavy-tailed weights.
3. Latency and temp memory vs token count (decode = 1 token ... prefill = 2048).
"""
import gc
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from lab import gpu_guard  # noqa: E402

import torch  # noqa: E402
import bitsandbytes as bnb  # noqa: E402
import bitsandbytes.functional as bnbF  # noqa: E402

SEED = 0
# (name, in_features, out_features)
SHAPES = [
    ("qwen3-0.6b up_proj", 1024, 3072),
    ("llama-7b q_proj", 4096, 4096),
    ("llama-7b up_proj", 4096, 11008),
]
TOKEN_COUNTS = [1, 16, 256, 2048]
NUMERICS_TOKENS = 256
OUTLIER_SCALE = 25.0
INT8_THRESHOLD = 6.0
VARIANTS = ["fp16", "int8", "int8_t6", "fp4", "nf4", "nf4_dq"]


def make_weight(in_f: int, out_f: int, dist: str, g: torch.Generator) -> torch.Tensor:
    std = in_f ** -0.5
    if dist == "gaussian":
        w = torch.randn(out_f, in_f, generator=g) * std
    elif dist == "student_t3":
        # Heavy tails, rescaled to the same std as the Gaussian case.
        normal = torch.randn(out_f, in_f, generator=g)
        chi2 = sum(torch.randn(out_f, in_f, generator=g) ** 2 for _ in range(3))
        t = normal / (chi2 / 3).sqrt()
        w = t / t.std() * std
    else:
        raise ValueError(dist)
    return w.to(torch.float16)


def make_input(tokens: int, in_f: int, kind: str, g: torch.Generator) -> torch.Tensor:
    x = torch.randn(tokens, in_f, generator=g)
    if kind == "outlier":
        n = max(2, in_f // 512)
        idx = torch.randperm(in_f, generator=g)[:n]
        x[:, idx] *= OUTLIER_SCALE
    return x.to(torch.float16).cuda()


def build(variant: str, w: torch.Tensor) -> torch.nn.Module:
    out_f, in_f = w.shape
    if variant == "fp16":
        layer = torch.nn.Linear(in_f, out_f, bias=False, dtype=torch.float16)
    elif variant in ("int8", "int8_t6"):
        layer = bnb.nn.Linear8bitLt(in_f, out_f, bias=False, has_fp16_weights=False,
                                    threshold=INT8_THRESHOLD if variant == "int8_t6" else 0.0)
    else:
        layer = bnb.nn.Linear4bit(in_f, out_f, bias=False, compute_dtype=torch.float16,
                                  quant_type="fp4" if variant == "fp4" else "nf4",
                                  compress_statistics=variant == "nf4_dq")
    layer.weight.data.copy_(w)
    return layer.cuda().eval()


def dequantized_weight(variant: str, layer: torch.nn.Module) -> torch.Tensor:
    p = layer.weight
    if variant == "fp16":
        return p.data.double()
    if variant.startswith("int8"):
        return p.data.double() * p.SCB.double()[:, None] / 127.0
    return bnbF.dequantize_4bit(p.data, p.quant_state).double()


def free_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def resident_mib(variant: str, w: torch.Tensor) -> tuple[torch.nn.Module, float]:
    free_cuda()
    base = torch.cuda.memory_allocated()
    layer = build(variant, w)
    free_cuda()
    return layer, (torch.cuda.memory_allocated() - base) / 2**20


def rel_err(a: torch.Tensor, ref: torch.Tensor) -> float:
    return float(f"{((a.double() - ref).norm() / ref.norm()).item():.4g}")


@torch.inference_mode()
def numerics(variant, layer, w_ref, inputs):
    row = {"variant": variant, "weight_rel_err": rel_err(dequantized_weight(variant, layer), w_ref)}
    for kind, x in inputs.items():
        ref = x.double() @ w_ref.t()
        row[f"out_rel_err_{kind}"] = rel_err(layer(x), ref)
    return row


@torch.inference_mode()
def benchmark(layer, x, iters: int = 100, warmup: int = 10):
    for _ in range(warmup):
        layer(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        layer(x)
    end.record()
    torch.cuda.synchronize()
    return {
        "ms_per_call": round(start.elapsed_time(end) / iters, 4),
        "extra_peak_mib": round((torch.cuda.max_memory_allocated() - base) / 2**20, 2),
    }


def upstream_commit() -> str:
    return subprocess.run(["git", "-C", str(ROOT / "upstream" / "bitsandbytes"), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def main():
    gpu = gpu_guard.acquire()
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED)

    memory, numerics_rows, perf = [], [], []
    for shape_name, in_f, out_f in SHAPES:
        weights = {"gaussian": make_weight(in_f, out_f, "gaussian", g)}
        if (in_f, out_f) == (4096, 4096):
            weights["student_t3"] = make_weight(in_f, out_f, "student_t3", g)
        inputs = {kind: make_input(NUMERICS_TOKENS, in_f, kind, g) for kind in ("normal", "outlier")}
        bench_inputs = {t: make_input(t, in_f, "normal", g) for t in TOKEN_COUNTS}
        bench_outlier = {t: make_input(t, in_f, "outlier", g) for t in TOKEN_COUNTS}

        for dist, w in weights.items():
            w_ref = w.double().cuda()
            for variant in VARIANTS:
                layer, mib = resident_mib(variant, w)
                if dist == "gaussian":
                    fp16_bytes = in_f * out_f * 2
                    memory.append({"shape": shape_name, "variant": variant, "resident_mib": round(mib, 3),
                                   "vs_fp16": round(mib * 2**20 / fp16_bytes, 4),
                                   "bits_per_weight": round(mib * 2**20 * 8 / (in_f * out_f), 3)})
                numerics_rows.append({"shape": shape_name, "weights": dist, **numerics(variant, layer, w_ref, inputs)})
                if dist == "gaussian":
                    for t in TOKEN_COUNTS:
                        perf.append({"shape": shape_name, "variant": variant, "input": "normal", "tokens": t,
                                     **benchmark(layer, bench_inputs[t])})
                        if variant.startswith("int8"):
                            perf.append({"shape": shape_name, "variant": variant, "input": "outlier", "tokens": t,
                                         **benchmark(layer, bench_outlier[t])})
                del layer
            del w_ref
        print(f"done: {shape_name}", flush=True)

    fp16_ms = {(r["shape"], r["tokens"]): r["ms_per_call"] for r in perf if r["variant"] == "fp16"}
    for r in perf:
        r["speedup_vs_fp16"] = round(fp16_ms[(r["shape"], r["tokens"])] / r["ms_per_call"], 3)

    result = {
        "experiment": HERE.name,
        "upstream": "bitsandbytes-foundation/bitsandbytes",
        "upstream_commit": upstream_commit(),
        "bitsandbytes": bnb.__version__,
        "components": ["bitsandbytes.nn.Linear8bitLt", "bitsandbytes.nn.Linear4bit"],
        "variants": {
            "fp16": "torch.nn.Linear, fp16 weights",
            "int8": "Linear8bitLt(has_fp16_weights=False, threshold=0)",
            "int8_t6": f"Linear8bitLt(has_fp16_weights=False, threshold={INT8_THRESHOLD})",
            "fp4": "Linear4bit(quant_type='fp4', compress_statistics=False, compute_dtype=fp16)",
            "nf4": "Linear4bit(quant_type='nf4', compress_statistics=False, compute_dtype=fp16)",
            "nf4_dq": "Linear4bit(quant_type='nf4', compress_statistics=True, compute_dtype=fp16)",
        },
        "numerics_tokens": NUMERICS_TOKENS,
        "outlier_scale": OUTLIER_SCALE,
        "gpu": gpu,
        "memory": memory,
        "numerics": numerics_rows,
        "latency": perf,
        "peak_memory_mib": gpu_guard.peak_memory_mib(),
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        f"bitsandbytes {bnb.__version__}, torch {torch.__version__}, {gpu.get('name', 'cuda')}",
        "",
        "## Resident weight memory (no bias)",
        "",
        "| shape | variant | MiB | vs fp16 | bits/weight |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {r['shape']} | {r['variant']} | {r['resident_mib']} | {r['vs_fp16']} | {r['bits_per_weight']} |"
              for r in memory]
    lines += [
        "",
        f"## Relative error vs fp64 reference ({NUMERICS_TOKENS} tokens; "
        f"outlier = {OUTLIER_SCALE:g}x on in_features/512 channels)",
        "",
        "| shape | weights | variant | weight rel err | output rel err (normal x) | output rel err (outlier x) |",
        "|---|---|---|---|---|---|",
    ]
    lines += [f"| {r['shape']} | {r['weights']} | {r['variant']} | {r['weight_rel_err']} | "
              f"{r['out_rel_err_normal']} | {r['out_rel_err_outlier']} |" for r in numerics_rows]
    lines += [
        "",
        "## Latency (fp16 activations, 100 calls after 10 warmup, CUDA events)",
        "",
        "| shape | variant | input | tokens | ms/call | speedup vs fp16 | extra peak MiB |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [f"| {r['shape']} | {r['variant']} | {r['input']} | {r['tokens']} | {r['ms_per_call']} | "
              f"{r['speedup_vs_fp16']} | {r['extra_peak_mib']} |" for r in perf]
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
