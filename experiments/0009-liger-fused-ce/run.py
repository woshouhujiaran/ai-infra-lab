"""Liger-Kernel cross entropy / fused linear cross entropy vs PyTorch CE.

LM head + loss, forward and backward, bf16 hidden states and weights:

1. Numerics: loss and dX / dW error vs an fp32 reference built from the same
   bf16 inputs (and the error of simply rounding that reference to bf16).
2. Extra peak memory and latency of one training step vs token count on real
   (hidden, vocab) shapes, including where PyTorch runs out of memory under the
   60% GPU cap.
"""
import gc
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LIGER_SRC = ROOT / "upstream" / "Liger-Kernel" / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LIGER_SRC))

from lab import gpu_guard  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import triton  # noqa: E402
from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction  # noqa: E402
from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction  # noqa: E402

SEED = 0
# (name, hidden, vocab)
SHAPES = [
    ("Qwen3-0.6B head", 1024, 151936),
    ("Llama-3.2-1B head", 2048, 128256),
    ("h1024 / 32k vocab", 1024, 32000),
]
TOKEN_COUNTS = [1024, 4096, 16384]
NUMERICS_TOKENS = 1024
LOGIT_STD = 3.0
WARMUP = 2
ITERS = 10
DTYPE = torch.bfloat16


def torch_bf16(x, w, y):
    return F.cross_entropy(x @ w.t(), y)


def torch_fp32(x, w, y):
    return F.cross_entropy((x @ w.t()).float(), y)


torch_compiled = torch.compile(torch_fp32, dynamic=False)


def liger_ce(x, w, y):
    return LigerCrossEntropyFunction.apply(x @ w.t(), y, None)[0]


def flce(chunk_mem_const=1, accum_dtype=None):
    def fn(x, w, y):
        return LigerFusedLinearCrossEntropyFunction.apply(
            x, w, y, None, None, -100, 0.0, 0.0, "mean", None, False, accum_dtype,
            False, False, False, None, None, chunk_mem_const)[0]
    return fn


VARIANTS = {
    "torch_bf16": torch_bf16,
    "torch_fp32": torch_fp32,
    "torch_compile": torch_compiled,
    "liger_ce": liger_ce,
    "liger_flce": flce(),
    "liger_flce_c8": flce(chunk_mem_const=8),
    "liger_flce_fp32acc": flce(accum_dtype=torch.float32),
}
DESCRIPTIONS = {
    "torch_bf16": "F.cross_entropy(x @ W.T) on bf16 logits",
    "torch_fp32": "F.cross_entropy((x @ W.T).float()) - the HF upcast path",
    "torch_compile": "torch.compile(torch_fp32, dynamic=False) via triton-windows",
    "liger_ce": "x @ W.T in bf16, then LigerCrossEntropyFunction (grad written in place into the logits)",
    "liger_flce": "LigerFusedLinearCrossEntropyFunction, default chunk_mem_const=1, bf16 dW accumulation",
    "liger_flce_c8": "LigerFusedLinearCrossEntropyFunction, chunk_mem_const=8 (8x larger chunks)",
    "liger_flce_fp32acc": "LigerFusedLinearCrossEntropyFunction, chunk_mem_const=1, accum_dtype=float32",
}


def flce_chunks(tokens: int, hidden: int, vocab: int, chunk_mem_const: int) -> dict:
    inc_factor = triton.cdiv(vocab, chunk_mem_const * hidden)
    chunk = min(triton.next_power_of_2(triton.cdiv(tokens, inc_factor)), tokens)
    return {"chunk_size": chunk, "num_chunks": triton.cdiv(tokens, chunk)}


def free_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def make_inputs(tokens, hidden, vocab, g):
    x = torch.randn(tokens, hidden, generator=g).to(DTYPE).cuda().requires_grad_()
    w = (torch.randn(vocab, hidden, generator=g) * (LOGIT_STD / math.sqrt(hidden))).to(DTYPE).cuda().requires_grad_()
    return x, w


def rel_err(a: torch.Tensor, ref: torch.Tensor) -> float:
    return float(f"{((a.float() - ref).norm() / ref.norm()).item():.4g}")


def fp32_reference(x, w, y):
    """Loss and grads of mean CE in fp32 from the bf16 inputs."""
    x32, w32 = x.detach().float(), w.detach().float()
    logits = x32 @ w32.t()
    lse = torch.logsumexp(logits, dim=-1)
    loss = (lse - logits.gather(1, y[:, None]).squeeze(1)).mean()
    grad = logits.sub_(lse[:, None]).exp_()
    grad[torch.arange(len(y), device=y.device), y] -= 1.0
    grad /= len(y)
    dx = grad @ w32
    del w32
    dw = grad.t() @ x32
    return loss.item(), dx, dw


def step(fn, x, w, y):
    x.grad = None
    w.grad = None
    loss = fn(x, w, y)
    loss.backward()
    return loss


def numerics(shape_name, hidden, vocab, g):
    x, w = make_inputs(NUMERICS_TOKENS, hidden, vocab, g)
    with torch.no_grad():
        probs = torch.softmax((x.float() @ w.float().t()), dim=-1)
        y = torch.multinomial(probs, 1, generator=torch.Generator("cuda").manual_seed(SEED)).squeeze(1)
        del probs
    ref_loss, ref_dx, ref_dw = fp32_reference(x, w, y)
    rows = [{"shape": shape_name, "variant": "fp32 ref rounded to bf16", "loss": round(ref_loss, 6),
             "loss_abs_err": 0.0, "dx_rel_err": rel_err(ref_dx.to(DTYPE), ref_dx),
             "dw_rel_err": rel_err(ref_dw.to(DTYPE), ref_dw)}]
    for name, fn in VARIANTS.items():
        loss = step(fn, x, w, y)
        rows.append({"shape": shape_name, "variant": name, "loss": round(loss.item(), 6),
                     "loss_abs_err": float(f"{abs(loss.item() - ref_loss):.3g}"),
                     "dx_rel_err": rel_err(x.grad, ref_dx), "dw_rel_err": rel_err(w.grad, ref_dw)})
    del x, w, y, ref_dx, ref_dw
    free_cuda()
    return rows


def benchmark(fn, x, w, y):
    for _ in range(WARMUP):
        step(fn, x, w, y)
    x.grad = None
    w.grad = None
    free_cuda()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    step(fn, x, w, y)
    torch.cuda.synchronize()
    extra_peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        step(fn, x, w, y)
    end.record()
    torch.cuda.synchronize()
    return {"status": "ok", "extra_peak_mib": round(extra_peak, 1),
            "ms_per_step": round(start.elapsed_time(end) / ITERS, 3)}


def perf(shape_name, hidden, vocab, g):
    rows = []
    for tokens in TOKEN_COUNTS:
        x, w = make_inputs(tokens, hidden, vocab, g)
        y = torch.randint(0, vocab, (tokens,), generator=g).cuda()
        mib = lambda n: round(n * 2 / 2**20, 1)  # noqa: E731
        sizes = {"logits_bf16_mib": mib(tokens * vocab), "dx_plus_dw_bf16_mib": mib(tokens * hidden + vocab * hidden)}
        for name, fn in VARIANTS.items():
            row = {"shape": shape_name, "hidden": hidden, "vocab": vocab, "tokens": tokens, "variant": name, **sizes}
            if name.startswith("liger_flce"):
                row.update(flce_chunks(tokens, hidden, vocab, 8 if name == "liger_flce_c8" else 1))
            try:
                row.update(benchmark(fn, x, w, y))
            except torch.OutOfMemoryError:
                row.update({"status": "OOM", "extra_peak_mib": None, "ms_per_step": None})
            x.grad = None
            w.grad = None
            free_cuda()
            rows.append(row)
            print(f"{shape_name} tokens={tokens} {name}: {row['status']} "
                  f"{row['extra_peak_mib']} MiB {row['ms_per_step']} ms", flush=True)
        del x, w, y
        free_cuda()
    return rows


def upstream_commit() -> str:
    return subprocess.run(["git", "-C", str(ROOT / "upstream" / "Liger-Kernel"), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def main():
    gpu = gpu_guard.acquire()
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED)

    numerics_rows, perf_rows = [], []
    for shape_name, hidden, vocab in SHAPES:
        numerics_rows += numerics(shape_name, hidden, vocab, g)
        perf_rows += perf(shape_name, hidden, vocab, g)

    base_ms = {(r["shape"], r["tokens"]): r["ms_per_step"] for r in perf_rows
               if r["variant"] == "torch_fp32" and r["status"] == "ok"}
    for r in perf_rows:
        b = base_ms.get((r["shape"], r["tokens"]))
        r["speedup_vs_torch_fp32"] = round(b / r["ms_per_step"], 3) if b and r["ms_per_step"] else None

    result = {
        "experiment": HERE.name,
        "upstream": "linkedin/Liger-Kernel",
        "upstream_commit": upstream_commit(),
        "components": ["liger_kernel/ops/cross_entropy.py (LigerCrossEntropyFunction)",
                       "liger_kernel/ops/fused_linear_cross_entropy.py (LigerFusedLinearCrossEntropyFunction)"],
        "variants": DESCRIPTIONS,
        "dtype": str(DTYPE),
        "logit_std": LOGIT_STD,
        "numerics_tokens": NUMERICS_TOKENS,
        "warmup": WARMUP,
        "iters": ITERS,
        "triton": triton.__version__,
        "gpu": gpu,
        "numerics": numerics_rows,
        "perf": perf_rows,
        "peak_memory_mib": gpu_guard.peak_memory_mib(),
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fmt = lambda v: "OOM" if v is None else v  # noqa: E731
    lines = [
        f"Liger-Kernel {upstream_commit()[:10]}, torch {torch.__version__}, triton {triton.__version__}, "
        f"{gpu.get('name', 'cuda')} (memory cap {gpu.get('memory_fraction_cap')})",
        "",
        f"## Numerics ({NUMERICS_TOKENS} tokens, bf16 inputs, targets sampled from the model's softmax, "
        f"logit std {LOGIT_STD:g}; reference = fp32 math on the same bf16 inputs)",
        "",
        "| shape | variant | loss | loss abs err | dX rel err | dW rel err |",
        "|---|---|---|---|---|---|",
    ]
    lines += [f"| {r['shape']} | {r['variant']} | {r['loss']} | {r['loss_abs_err']} | {r['dx_rel_err']} | "
              f"{r['dw_rel_err']} |" for r in numerics_rows]
    lines += [
        "",
        f"## One training step of LM head + CE (forward + backward; {WARMUP} warmup, {ITERS} timed, CUDA events)",
        "",
        "Extra peak = max allocated during the step minus allocated before it (x, W, targets); "
        "it includes the dX and dW that the step returns.",
        "",
        "| shape | tokens | bf16 logits MiB | dX+dW MiB | variant | chunks | extra peak MiB | ms/step | "
        "speedup vs torch_fp32 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [f"| {r['shape']} | {r['tokens']} | {r['logits_bf16_mib']} | {r['dx_plus_dw_bf16_mib']} | "
              f"{r['variant']} | {r.get('num_chunks', '')} | {fmt(r['extra_peak_mib'])} | {fmt(r['ms_per_step'])} | "
              f"{r['speedup_vs_torch_fp32'] or ''} |" for r in perf_rows]
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
