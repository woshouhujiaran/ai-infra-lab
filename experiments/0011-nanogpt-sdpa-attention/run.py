"""nanoGPT CausalSelfAttention: SDPA ("flash") path vs the manual path on an RTX 4060 Laptop.

nanoGPT sets `self.flash = hasattr(F, 'scaled_dot_product_attention')` and then
calls SDPA with is_causal=True, trusting PyTorch to pick FlashAttention. The
manual path materialises the (B, nh, T, T) scores, masks them with a
block_size x block_size tril buffer and runs softmax + dropout + matmul.

1. Which SDPA backend actually runs (CUDA kernel names from the profiler) when
   the module is used as-is, and when cuDNN / math backends are forced.
2. Forward (no_grad) and forward+backward latency and extra peak memory vs
   sequence length at a fixed 16k tokens per batch, GPT-2 small attention
   (768 dims, 12 heads), under nanoGPT's bf16 autocast and fp32 (TF32) modes;
   torch.compile of the manual path; a linears-only baseline.
3. Training with attention dropout 0.1.
4. Output / input-gradient / weight-gradient error vs an fp64 manual reference.

model.py is loaded unmodified from the submodule. The manual path is obtained
by hiding F.scaled_dot_product_attention while the module is constructed, so
nanoGPT's own __init__ registers its causal-mask buffer.
"""
import contextlib
import gc
import importlib.util
import io
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NANOGPT_DIR = ROOT / "upstream" / "nanoGPT"
sys.path.insert(0, str(ROOT))

from lab import gpu_guard  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.autograd import DeviceType  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402
from triton.testing import do_bench  # noqa: E402

SEED = 0
N_EMBD = 768
N_HEAD = 12
TOKENS = 16384
TS = [256, 512, 1024, 2048, 4096, 8192, 16384]
DETECT_T = 1024
NUMERICS_T = 1024
NUMERICS_B = 2
DROPOUT = 0.1
DROPOUT_TS = [1024, 4096]
MODES = ["bf16_autocast", "fp32_tf32"]

# nanoGPT's train.py enables TF32 for every run.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch._dynamo.config.recompile_limit = 64

GPU = gpu_guard.acquire()


def load_nanogpt():
    spec = importlib.util.spec_from_file_location("nanogpt_model", NANOGPT_DIR / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ng = load_nanogpt()

VARIANTS = {
    "linears_only": "c_attn + c_proj only (attention replaced by passing v through): lower bound",
    "sdpa_default": "nanoGPT as-is (self.flash=True): F.scaled_dot_product_attention(is_causal=True), default dispatch",
    "sdpa_cudnn": "same call inside sdpa_kernel(CUDNN_ATTENTION)",
    "sdpa_math": "same call inside sdpa_kernel(MATH): PyTorch's unfused reference",
    "manual": "nanoGPT manual path (self.flash=False): q@k^T, masked_fill(tril buffer), softmax, dropout, @v",
    "manual_compile": "torch.compile(manual module, dynamic=False) (bf16 autocast only)",
}
BACKENDS = {"sdpa_cudnn": [SDPBackend.CUDNN_ATTENTION], "sdpa_math": [SDPBackend.MATH]}


def make_attn(T, flash, dropout=0.0):
    cfg = ng.GPTConfig(block_size=T, n_head=N_HEAD, n_embd=N_EMBD, dropout=dropout, bias=True)
    torch.manual_seed(SEED)
    if flash:
        m = ng.CausalSelfAttention(cfg)
    else:
        sdpa = F.scaled_dot_product_attention
        del F.scaled_dot_product_attention
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                m = ng.CausalSelfAttention(cfg)
        finally:
            F.scaled_dot_product_attention = sdpa
    assert m.flash == flash
    return m.cuda()


class Runner:
    """One variant at one (mode, T): builds the module and runs fwd / fwd+bwd under the right contexts."""

    def __init__(self, variant, mode, T, B, dropout=0.0):
        self.variant, self.mode, self.T, self.B = variant, mode, T, B
        self.module = make_attn(T, flash=not variant.startswith("manual"), dropout=dropout)
        self.fn = self.module
        if variant == "linears_only":
            m = self.module
            self.fn = lambda x: m.c_proj(m.c_attn(x).split(N_EMBD, dim=2)[2])
        elif variant == "manual_compile":
            self.fn = torch.compile(self.module, dynamic=False)
        self.params = list(self.module.parameters())

    def ctx(self):
        stack = contextlib.ExitStack()
        if self.mode == "bf16_autocast":
            stack.enter_context(torch.autocast("cuda", dtype=torch.bfloat16))
        if self.variant in BACKENDS:
            stack.enter_context(sdpa_kernel(BACKENDS[self.variant]))
        return stack

    def call(self, phase, x, dy):
        if phase == "fwd":
            with torch.no_grad(), self.ctx():
                return self.fn(x)
        with self.ctx():
            y = self.fn(x)
        y.backward(dy)
        return y

    def zero_grad(self, x):
        x.grad = None
        for p in self.params:
            p.grad = None


def mask_buffer_mib(module):
    buf = getattr(module, "bias", None)
    return round(buf.numel() * buf.element_size() / 2**20, 1) if isinstance(buf, torch.Tensor) else 0.0


def cleanup():
    gc.collect()  # a failed call's temporaries are kept alive by the traceback cycle
    torch.cuda.empty_cache()


def classify(names):
    s = " ".join(names).lower()
    if "cudnn" in s and "sdpa" in s or "fort_native" in s:
        return "cudnn"
    if "fmha_cutlass" in s or "memeffattention" in s:
        return "efficient (CUTLASS mem-efficient)"
    if "flash_fwd" in s or "flash_bwd" in s:
        return "flash"
    if "triton" in s:
        return "inductor triton (unfused attention)"
    if "softmax" in s:
        return "unfused (bmm + softmax)"
    return "no attention kernels"


def detect(runner, x, dy):
    """Backend label + top-3 CUDA kernels by device time for one fwd+bwd."""
    try:
        runner.call("fwd_bwd", x, dy)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            runner.call("fwd_bwd", x, dy)
            torch.cuda.synchronize()
    except torch.OutOfMemoryError:
        cleanup()
        return {"backend": "OOM", "error": "OOM", "top_kernels": []}
    except RuntimeError as e:
        if "No available kernel" not in str(e):
            raise
        cleanup()
        return {"backend": "unsupported", "error": str(e).splitlines()[0][:200], "top_kernels": []}
    evts = [e for e in prof.key_averages() if e.device_type == DeviceType.CUDA]
    evts.sort(key=lambda e: e.self_device_time_total, reverse=True)
    names = [e.key for e in evts]
    return {"backend": classify(names), "error": None,
            "top_kernels": [n[:90] for n in names[:3]]}


def measure(runner, x, dy, phases=("fwd", "fwd_bwd")):
    row = {}
    for phase in phases:
        try:
            runner.zero_grad(x)
            runner.call(phase, x, dy)  # warmup / compile
            runner.zero_grad(x)
            cleanup()
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            runner.call(phase, x, dy)
            torch.cuda.synchronize()
            extra = (torch.cuda.max_memory_allocated() - base) / 2**20
            ms = do_bench(lambda: runner.call(phase, x, dy), grad_to_none=[x] + runner.params,
                          return_mode="median")
            row.update({f"{phase}_status": "ok", f"{phase}_ms": round(ms, 3), f"{phase}_extra_mib": round(extra, 1)})
        except torch.OutOfMemoryError:
            row.update({f"{phase}_status": "OOM", f"{phase}_ms": None, f"{phase}_extra_mib": None})
            cleanup()
        except RuntimeError as e:
            if "No available kernel" not in str(e):
                raise
            row.update({f"{phase}_status": "unsupported", f"{phase}_ms": None, f"{phase}_extra_mib": None})
            cleanup()
    runner.zero_grad(x)
    return row


def inputs(B, T, mode, g):
    x = torch.randn(B, T, N_EMBD, generator=g).cuda().requires_grad_()
    dy = torch.randn(B, T, N_EMBD, generator=g).cuda()
    if mode == "bf16_autocast":
        dy = dy.bfloat16()
    return x, dy


def variants_for(mode):
    return [v for v in VARIANTS if not (v == "manual_compile" and mode != "bf16_autocast")]


def dispatch(g):
    rows = []
    B = TOKENS // DETECT_T
    for mode in MODES:
        x, dy = inputs(B, DETECT_T, mode, g)
        for v in variants_for(mode):
            r = Runner(v, mode, DETECT_T, B)
            rows.append({"mode": mode, "variant": v, "dropout": 0.0, "T": DETECT_T, **detect(r, x, dy)})
            print(f"dispatch {mode} {v}: {rows[-1]['backend']} {rows[-1]['top_kernels'][:1]}", flush=True)
            del r
            cleanup()
        del x, dy
        cleanup()
    return rows


def sweep(g):
    rows = []
    for mode in MODES:
        for T in TS:
            B = TOKENS // T
            x, dy = inputs(B, T, mode, g)
            for v in variants_for(mode):
                try:
                    r = Runner(v, mode, T, B)
                except torch.OutOfMemoryError:
                    cleanup()
                    rows.append({"mode": mode, "T": T, "B": B, "variant": v, "mask_buffer_mib": None,
                                 "fwd_status": "OOM", "fwd_ms": None, "fwd_extra_mib": None,
                                 "fwd_bwd_status": "OOM", "fwd_bwd_ms": None, "fwd_bwd_extra_mib": None})
                    continue
                row = {"mode": mode, "T": T, "B": B, "variant": v, "mask_buffer_mib": mask_buffer_mib(r.module),
                       **measure(r, x, dy)}
                rows.append(row)
                print(f"{mode} T={T} B={B} {v}: fwd {row['fwd_ms'] or row['fwd_status']} ms "
                      f"({row['fwd_extra_mib']} MiB), fwd+bwd {row['fwd_bwd_ms'] or row['fwd_bwd_status']} ms "
                      f"({row['fwd_bwd_extra_mib']} MiB)", flush=True)
                del r
                cleanup()
            del x, dy
            cleanup()
    by_key = {(r["mode"], r["T"], r["variant"]): r for r in rows}
    for r in rows:
        ref = by_key.get((r["mode"], r["T"], "manual"))
        for phase in ("fwd", "fwd_bwd"):
            ok = ref and ref[f"{phase}_ms"] and r[f"{phase}_ms"]
            r[f"{phase}_speedup_vs_manual"] = round(ref[f"{phase}_ms"] / r[f"{phase}_ms"], 2) if ok else None
    return rows


def dropout_runs(g):
    rows = []
    mode = "bf16_autocast"
    for T in DROPOUT_TS:
        B = TOKENS // T
        x, dy = inputs(B, T, mode, g)
        for v in ("sdpa_default", "sdpa_cudnn", "manual", "manual_compile"):
            r = Runner(v, mode, T, B, dropout=DROPOUT)
            info = detect(r, x, dy)
            row = {"mode": mode, "T": T, "B": B, "variant": v, "dropout": DROPOUT, "backend": info["backend"],
                   "top_kernels": info["top_kernels"], **measure(r, x, dy, phases=("fwd_bwd",))}
            rows.append(row)
            print(f"dropout T={T} {v}: {row['backend']} {row['fwd_bwd_ms'] or row['fwd_bwd_status']} ms "
                  f"({row['fwd_bwd_extra_mib']} MiB)", flush=True)
            del r
            cleanup()
        del x, dy
        cleanup()
    return rows


def rel(a, ref):
    return float(f"{((a.double() - ref).norm() / ref.norm()).item():.3g}")


def numerics(g):
    T, B = NUMERICS_T, NUMERICS_B
    x64 = torch.randn(B, T, N_EMBD, generator=g, dtype=torch.float64).cuda().requires_grad_()
    dy64 = torch.randn(B, T, N_EMBD, generator=g, dtype=torch.float64).cuda()
    ref_m = make_attn(T, flash=False).double()
    y_ref = ref_m(x64)
    y_ref.backward(dy64)
    ref = {"y": y_ref.detach(), "dx": x64.grad, "dw": ref_m.c_attn.weight.grad}
    rows = []
    for mode in MODES:
        for v in variants_for(mode):
            if v == "linears_only":
                continue
            r = Runner(v, mode, T, B)
            x = x64.detach().float().requires_grad_()
            dy = dy64.bfloat16() if mode == "bf16_autocast" else dy64.float()
            try:
                y = r.call("fwd_bwd", x, dy)
            except RuntimeError as e:
                if "No available kernel" not in str(e):
                    raise
                rows.append({"mode": mode, "variant": v, "status": "unsupported"})
                cleanup()
                continue
            rows.append({"mode": mode, "variant": v, "status": "ok",
                         "y_rel_err": rel(y.detach(), ref["y"]),
                         "y_max_abs_err": float(f"{(y.detach().double() - ref['y']).abs().max().item():.3g}"),
                         "dx_rel_err": rel(x.grad, ref["dx"]),
                         "dw_qkv_rel_err": rel(r.module.c_attn.weight.grad, ref["dw"])})
            print(f"numerics {mode} {v}: {rows[-1]}", flush=True)
            del r
            cleanup()
    return rows


def upstream_commit() -> str:
    return subprocess.run(["git", "-C", str(NANOGPT_DIR), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def main():
    t_start = time.time()
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED)

    dispatch_rows = dispatch(g)
    perf = sweep(g)
    drop = dropout_runs(g)
    numeric_rows = numerics(g)

    result = {
        "experiment": HERE.name,
        "upstream": "karpathy/nanoGPT",
        "upstream_commit": upstream_commit(),
        "component": "model.py (CausalSelfAttention)",
        "variants": VARIANTS,
        "n_embd": N_EMBD,
        "n_head": N_HEAD,
        "tokens_per_batch": TOKENS,
        "flash_attention_compiled": torch.backends.cuda.is_flash_attention_available(),
        "cudnn": torch.backends.cudnn.version(),
        "gpu": GPU,
        "dispatch": dispatch_rows,
        "perf": perf,
        "dropout": drop,
        "numerics": numeric_rows,
        "peak_memory_mib": gpu_guard.peak_memory_mib(),
        "runtime_s": round(time.time() - t_start, 1),
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    def cell(r, phase, key):
        return r[f"{phase}_{key}"] if r[f"{phase}_status"] == "ok" else r[f"{phase}_status"]

    lines = [
        f"nanoGPT {upstream_commit()[:10]}, torch {torch.__version__}, cuDNN {torch.backends.cudnn.version()}, "
        f"FlashAttention compiled: {torch.backends.cuda.is_flash_attention_available()}, "
        f"{GPU.get('name', 'cuda')} (memory cap {GPU.get('memory_fraction_cap')}), runtime {result['runtime_s']} s",
        "",
        f"## SDPA dispatch (fwd+bwd at T={DETECT_T}, B={TOKENS // DETECT_T}; top CUDA kernels by device time)",
        "",
        "| mode | variant | backend | top kernels |",
        "|---|---|---|---|",
    ]
    lines += [f"| {r['mode']} | {r['variant']} | {r['backend']} | {'<br>'.join(r['top_kernels']) or r['error']} |"
              for r in dispatch_rows]
    for mode in MODES:
        lines += [
            "",
            f"## {mode}: latency (ms, `do_bench` median) and extra peak memory (MiB) at {TOKENS} tokens per batch",
            "",
            "| T | B | variant | fwd ms | fwd MiB | fwd+bwd ms | fwd+bwd MiB | fwd x manual | fwd+bwd x manual | "
            "mask buffer MiB |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        lines += [f"| {r['T']} | {r['B']} | {r['variant']} | {cell(r, 'fwd', 'ms')} | {cell(r, 'fwd', 'extra_mib')} | "
                  f"{cell(r, 'fwd_bwd', 'ms')} | {cell(r, 'fwd_bwd', 'extra_mib')} | "
                  f"{r['fwd_speedup_vs_manual'] or ''} | {r['fwd_bwd_speedup_vs_manual'] or ''} | "
                  f"{r['mask_buffer_mib'] or ''} |" for r in perf if r["mode"] == mode]
    lines += [
        "",
        f"## Training with attention dropout {DROPOUT} (bf16 autocast, fwd+bwd)",
        "",
        "| T | B | variant | backend | fwd+bwd ms | fwd+bwd MiB |",
        "|---|---|---|---|---|---|",
    ]
    lines += [f"| {r['T']} | {r['B']} | {r['variant']} | {r['backend']} | {cell(r, 'fwd_bwd', 'ms')} | "
              f"{cell(r, 'fwd_bwd', 'extra_mib')} |" for r in drop]
    lines += [
        "",
        f"## Error vs fp64 manual attention (T={NUMERICS_T}, B={NUMERICS_B}; relative L2 error)",
        "",
        "| mode | variant | y rel | y max abs | dx rel | dW_qkv rel |",
        "|---|---|---|---|---|---|",
    ]
    lines += [f"| {r['mode']} | {r['variant']} | {r.get('y_rel_err', r['status'])} | {r.get('y_max_abs_err', '')} | "
              f"{r.get('dx_rel_err', '')} | {r.get('dw_qkv_rel_err', '')} |" for r in numeric_rows]
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
