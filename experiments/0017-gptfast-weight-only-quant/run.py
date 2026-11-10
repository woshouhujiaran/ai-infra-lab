"""gpt-fast weight-only quantization (quantize.py) on a laptop GPU.

Component:
- WeightOnlyInt8QuantHandler / WeightOnlyInt8Linear: symmetric per-channel int8,
  forward = F.linear(x, w_int8.to(x.dtype)) * scales (bf16 scales).
- WeightOnlyInt4QuantHandler / WeightOnlyInt4Linear: asymmetric groupwise int4
  (groupsize 32-256) packed for torch's tinygemm kernel aten._weight_int4pack_mm.

1. Upstream handlers at HEAD with torch 2.14: what runs and what breaks; a repacking fix
   checked against the dequantized weight (and the wrong nibble order / no packing);
   int4 padding of stories15M's 288-wide layers; torch's _weight_int8pack_mm on CUDA.
2. Per-layer output error vs fp64 on stories110M weights with real fp32 activations:
   bf16 F.linear vs upstream int8 vs ideal int8 (fp32 scales, fp64 math) vs int4 g32-256
   (tinygemm, ideal fp64 math, and nearest codes computed in fp32 instead of bf16).
3. Whole model: KL to the fp32 model, top-1 agreement and perplexity on 8 x 256 tokens
   sampled from the fp32 model, for bf16 / int8 / int4 g32-256.
4. Linear latency vs token count M for stories110M and Llama-2-7B shapes: bf16 vs int8
   eager / torch.compile vs int4 tinygemm (g32 / g128, inner_k_tiles 2-8), weights rotated
   past L2, eager launches and CUDA-graph replay; effective weight bandwidth.

quantize.py / model.py are imported unmodified. run.py swaps quantize's
prepare_int4_weight_and_scales_and_zeros for a repacking copy (finding 1) and replaces
model.flex_attention with causal SDPA for teacher-forced forwards from position 0.
"""
import contextlib
import hashlib
import io
import json
import math
import platform
import struct
import sys
import time
import traceback
import types
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GPTFAST = ROOT / "upstream" / "gpt-fast"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GPTFAST))
# tokenizer.py (imported by quantize.py) imports sentencepiece at module level; it is not used here
sys.modules.setdefault("sentencepiece", types.ModuleType("sentencepiece"))

from lab import gpu_guard  # noqa: E402

GPU_INFO = gpu_guard.acquire()

import model as M  # noqa: E402
import quantize as Q  # noqa: E402
from model import ModelArgs, Transformer  # noqa: E402

SEED = 0
DEVICE = "cuda"
CKPT_DIR = Path.home() / ".cache" / "ai-infra-lab" / "tinyllamas"
MODEL = "stories110M"
EVAL_B = 8
EVAL_T = 256
ACT_SEQS = 4
GROUPS = [32, 64, 128, 256]
INNER_K_TILES = [2, 4, 8]
M_LIST = [1, 4, 16, 64, 256, 1024, 2048]
ITK_M = [1, 16]
SHAPES = [  # (model, layer, out_features N, in_features K)
    ("110M", "wqkv", 2304, 768),
    ("110M", "wo", 768, 768),
    ("110M", "w1/w3", 2048, 768),
    ("110M", "w2", 768, 2048),
    ("110M", "output", 32000, 768),
    ("7B", "wqkv", 12288, 4096),
    ("7B", "wo", 4096, 4096),
    ("7B", "w1/w3", 11008, 4096),
    ("7B", "w2", 4096, 11008),
]
ROT_BYTES = 96 * 2**20  # 3x the 4060's 32 MiB L2, so every call streams its weight from DRAM
TARGET_S = 0.03
GRAPH_MAX_CALLS = 400

for _k in ("cache_size_limit", "recompile_limit"):
    if hasattr(torch._dynamo.config, _k):
        setattr(torch._dynamo.config, _k, 512)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

UPSTREAM_PREPARE_INT4 = Q.prepare_int4_weight_and_scales_and_zeros


def sync():
    torch.cuda.synchronize()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def where(e: BaseException) -> str:
    frames = [f for f in traceback.extract_tb(e.__traceback__) if str(GPTFAST) in f.filename]
    f = frames[-1] if frames else traceback.extract_tb(e.__traceback__)[-1]
    return f"{Path(f.filename).name}:{f.lineno}"


def err_text(e: BaseException) -> str:
    return f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"


# ---------------------------------------------------------------- model

_raw: dict = {}


def load_llama2c(name: str, dtype: torch.dtype, batch: int = 1) -> Transformer:
    """llama2.c export format v0: 7 int32 header, then fp32 weights in a fixed order."""
    if name not in _raw:
        _raw[name] = (CKPT_DIR / f"{name}.bin").read_bytes()
    raw = _raw[name]
    dim, hidden, n_layer, n_head, n_kv, vocab, seq_len = struct.unpack("7i", raw[:28])
    shared = vocab > 0
    vocab = abs(vocab)
    w = torch.frombuffer(bytearray(raw[28:]), dtype=torch.float32)
    head = dim // n_head
    off = 0

    def take(*shape):
        nonlocal off
        n = math.prod(shape)
        t = w[off:off + n].view(*shape)
        off += n
        return t

    emb = take(vocab, dim)
    rms_att = take(n_layer, dim)
    wq = take(n_layer, dim, dim)
    wk = take(n_layer, n_kv * head, dim)
    wv = take(n_layer, n_kv * head, dim)
    wo = take(n_layer, dim, dim)
    rms_ffn = take(n_layer, dim)
    w1 = take(n_layer, hidden, dim)
    w2 = take(n_layer, dim, hidden)
    w3 = take(n_layer, hidden, dim)
    rms_final = take(dim)
    take(seq_len, head // 2)
    take(seq_len, head // 2)
    wcls = emb if shared else take(vocab, dim)
    assert off == w.numel(), (name, off, w.numel())

    args = ModelArgs(block_size=seq_len, vocab_size=vocab, n_layer=n_layer, n_head=n_head, dim=dim,
                     intermediate_size=hidden, n_local_heads=n_kv)
    sd = {"tok_embeddings.weight": emb, "norm.weight": rms_final, "output.weight": wcls}
    for i in range(n_layer):
        p = f"layers.{i}."
        sd[p + "attention_norm.weight"] = rms_att[i]
        sd[p + "attention.wqkv.weight"] = torch.cat([wq[i], wk[i], wv[i]])
        sd[p + "attention.wo.weight"] = wo[i]
        sd[p + "ffn_norm.weight"] = rms_ffn[i]
        sd[p + "feed_forward.w1.weight"] = w1[i]
        sd[p + "feed_forward.w2.weight"] = w2[i]
        sd[p + "feed_forward.w3.weight"] = w3[i]
    with torch.device("meta"):
        model = Transformer(args)
    model.load_state_dict(sd, assign=True)
    model = model.to(device=DEVICE, dtype=dtype).eval()
    with torch.device(DEVICE):
        model.setup_caches(max_batch_size=batch, max_seq_length=EVAL_T)
    return model


def sdpa_from_zero(q, k, v, block_mask=None, enable_gqa=False):
    """Causal attention for a forward over positions 0..S-1 (all this experiment needs)."""
    S = q.shape[2]
    return F.scaled_dot_product_attention(q, k[:, :, :S], v[:, :, :S], is_causal=True, enable_gqa=enable_gqa)


M.flex_attention = sdpa_from_zero


def forward_full(model, seqs):
    return model(types.SimpleNamespace(mask_mod=None), seqs, torch.arange(seqs.shape[1], device=DEVICE))


def prepare_int4_fixed(weight_bf16, groupsize, inner_k_tiles):
    """Upstream helper with the torch >= 2.5 input format: [N, K/2] uint8, even column in the high nibble."""
    weight_int32, scales_and_zeros = Q.group_quantize_tensor(weight_bf16, n_bit=4, groupsize=groupsize)
    packed = (weight_int32[:, ::2] << 4 | weight_int32[:, 1::2]).to(torch.uint8)
    return torch.ops.aten._convert_weight_to_int4pack(packed, inner_k_tiles), scales_and_zeros


def quantize_model(model, cfg: str):
    if cfg == "bf16":
        return model
    if cfg == "int8":
        h = Q.WeightOnlyInt8QuantHandler(model)
    else:
        h = Q.WeightOnlyInt4QuantHandler(model, groupsize=int(cfg.split("-g")[1]), inner_k_tiles=8)
    with contextlib.redirect_stdout(io.StringIO()):
        sd = h.create_quantized_state_dict()
    h.convert_for_runtime()
    model.load_state_dict(sd)
    return model.to(DEVICE)


def linear_bytes(model) -> int:
    n = 0
    for mod in model.modules():
        if isinstance(mod, (torch.nn.Linear, Q.WeightOnlyInt8Linear, Q.WeightOnlyInt4Linear)):
            n += sum(t.numel() * t.element_size() for t in list(mod.parameters()) + list(mod.buffers()))
    return n


def linear_weights(model) -> int:
    return sum(m.in_features * m.out_features for m in model.modules()
               if isinstance(m, (torch.nn.Linear, Q.WeightOnlyInt8Linear, Q.WeightOnlyInt4Linear)))


def rel(y, ref) -> float:
    return float((y.double() - ref).norm() / ref.norm())


def int4_dequant64(w_bf16, g, codes=None):
    scales, zeros = Q.get_group_qparams(w_bf16, 4, g)
    if codes is None:
        codes = Q.group_quantize_tensor_from_qparams(w_bf16, scales, zeros, 4, g)
    wd = Q.group_dequantize_tensor_from_qparams(codes.double(), scales.double(), zeros.double(), 4, g)
    return wd, codes, scales, zeros


def int4_nearest_codes(w_bf16, scales, zeros, g):
    """Same qparams, but (w - min) / scale computed in fp32 instead of bf16."""
    w = w_bf16.float().reshape(-1, g)
    s = scales.float().reshape(-1, 1)
    mn = zeros.float().reshape(-1, 1) - s * 8
    return w.sub(mn).div(s).round().clamp(0, 15).to(torch.int32).reshape_as(w_bf16)


# ---------------------------------------------------------------- 1. upstream as-is

def part_upstream():
    res = {"checks": []}
    add = res["checks"].append
    torch.manual_seed(SEED)

    m = load_llama2c(MODEL, torch.bfloat16)
    try:
        quantize_model(m, "int8")
        x = torch.tensor([[1, 450, 2217]], device=DEVICE)
        with torch.no_grad():
            out = forward_full(m, x)
        add({"attempt": "WeightOnlyInt8QuantHandler + convert_for_runtime + forward", "ok": True,
             "detail": f"logits {tuple(out.shape)} {out.dtype}"})
    except Exception as e:
        add({"attempt": "WeightOnlyInt8QuantHandler + convert_for_runtime + forward", "ok": False,
             "error": err_text(e), "where": where(e)})
    del m

    m = load_llama2c(MODEL, torch.bfloat16)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            Q.WeightOnlyInt4QuantHandler(m, groupsize=128, inner_k_tiles=8).create_quantized_state_dict()
        add({"attempt": "WeightOnlyInt4QuantHandler.create_quantized_state_dict (g128)", "ok": True})
    except Exception as e:
        add({"attempt": "WeightOnlyInt4QuantHandler.create_quantized_state_dict (g128)", "ok": False,
             "error": err_text(e), "where": where(e)})

    w = m.layers[0].feed_forward.w1.weight.detach()  # [2048, 768] bf16
    del m
    x = torch.randn(16, w.shape[1], device=DEVICE, dtype=torch.bfloat16)
    g = 128
    wd, wi, _, _ = int4_dequant64(w, g)
    _, sz = Q.group_quantize_tensor(w, 4, g)
    ref = x.double() @ wd.T
    variants = {
        "uint8, even column in high nibble (fix)": lambda: (wi[:, ::2] << 4 | wi[:, 1::2]).to(torch.uint8),
        "uint8, odd column in high nibble": lambda: (wi[:, 1::2] << 4 | wi[:, ::2]).to(torch.uint8),
        "int32 codes .to(uint8), no packing": lambda: wi.to(torch.uint8),
    }
    res["packing"] = []
    for name, fn in variants.items():
        r = {"variant": name}
        try:
            p = torch.ops.aten._convert_weight_to_int4pack(fn(), 8)
            r["packed_shape"] = list(p.shape)
            y = torch.ops.aten._weight_int4pack_mm(x, p, g, sz)
            r["out_shape"] = list(y.shape)
            r["rel_err_vs_dequant"] = rel(y, ref)
        except Exception as e:
            r["error"] = err_text(e)
        res["packing"].append(r)
        print("  packing", r)

    try:
        Q.group_dequantize_tensor(*Q.group_quantize_tensor(w, 4, g), 4, g)
        add({"attempt": "group_dequantize_tensor(*group_quantize_tensor(w))", "ok": True})
    except Exception as e:
        add({"attempt": "group_dequantize_tensor(*group_quantize_tensor(w))", "ok": False,
             "error": err_text(e) or "AssertionError", "where": where(e)})

    q8, s8, _ = Q.dynamically_quantize_per_channel(w.float(), -128, 127, torch.int8)
    try:
        y = torch.ops.aten._weight_int8pack_mm(x, q8, s8.to(torch.bfloat16))
        ref8 = x.double() @ (q8.double() * s8.double()[:, None]).T
        add({"attempt": "torch.ops.aten._weight_int8pack_mm on CUDA", "ok": True, "detail": f"rel err {rel(y, ref8):.2e}"})
    except Exception as e:
        add({"attempt": "torch.ops.aten._weight_int8pack_mm on CUDA", "ok": False, "error": err_text(e)})

    Q.prepare_int4_weight_and_scales_and_zeros = prepare_int4_fixed
    m = load_llama2c(MODEL, torch.bfloat16)
    try:
        quantize_model(m, "int4-g128")
        with torch.no_grad():
            out = forward_full(m, torch.tensor([[1, 450, 2217]], device=DEVICE))
        add({"attempt": "WeightOnlyInt4QuantHandler with repacking fix + forward", "ok": True,
             "detail": f"logits {tuple(out.shape)} {out.dtype}"})
    except Exception as e:
        add({"attempt": "WeightOnlyInt4QuantHandler with repacking fix + forward", "ok": False,
             "error": err_text(e), "where": where(e)})
    del m
    for c in res["checks"]:
        print("  ", c)

    # stories15M: dim 288 is a multiple of 32 but not of inner_k_tiles * 16 = 128
    m15 = load_llama2c("stories15M", torch.bfloat16)
    bf16_bytes = linear_bytes(m15)
    lins = {n: (mod.out_features, mod.in_features) for n, mod in m15.named_modules() if isinstance(mod, torch.nn.Linear)}
    pad = []
    for g in (32, 128):
        with contextlib.redirect_stdout(io.StringIO()):
            sd = Q.WeightOnlyInt4QuantHandler(m15, groupsize=g, inner_k_tiles=8).create_quantized_state_dict()
        up = sum(sd[f"{n}.weight"].numel() * 4 + sd[f"{n}.scales_and_zeros"].numel() * 2 for n in lins)
        step = math.lcm(g, 8 * 16)
        minimal = sum(N * M.find_multiple(K, step) // 2 + M.find_multiple(K, step) // g * N * 4 for N, K in lins.values())
        unpadded = sum(N * K // 2 + K // g * N * 4 for N, K in lins.values())
        n_padded = sum(1 for N, K in lins.values() if not Q._check_linear_int4_k(K, g, 8))
        pad.append({"groupsize": g, "padded_layers": n_padded, "layers": len(lins), "bf16_bytes": bf16_bytes,
                    "upstream_bytes": up, "pad_to_lcm_bytes": minimal, "no_pad_bytes": unpadded})
        print("  stories15M int4 padding", pad[-1])
    n, (N, K) = "layers.0.attention.wqkv", lins["layers.0.attention.wqkv"]
    w = m15.get_submodule(n).weight.detach()
    xs = torch.randn(16, K, device=DEVICE, dtype=torch.bfloat16)
    Kp = M.find_multiple(K, 1024)
    wd, _, _, _ = int4_dequant64(F.pad(w, (0, Kp - K)), 128)
    lin = Q.WeightOnlyInt4Linear(K, N, bias=False, groupsize=128, inner_k_tiles=8, padding=True)
    lin.load_state_dict({"weight": sd[f"{n}.weight"], "scales_and_zeros": sd[f"{n}.scales_and_zeros"]})
    lin = lin.to(DEVICE)
    res["stories15M_padding"] = {"rows": pad, "layer": n, "in_features": K, "padded_in_features": Kp,
                                 "padded_forward_rel_err": rel(lin(xs), xs.double() @ wd[:, :K].T)}
    print("  padded forward", res["stories15M_padding"]["padded_forward_rel_err"])
    del m15
    return res


# ---------------------------------------------------------------- 2-3. accuracy

def sample_eval_tokens(m32):
    gen = torch.Generator(device=DEVICE).manual_seed(SEED)
    seq = torch.ones(EVAL_B, 1, dtype=torch.long, device=DEVICE)  # BOS
    with torch.no_grad():
        for _ in range(EVAL_T - 1):
            p = torch.softmax(forward_full(m32, seq)[:, -1].float(), -1)
            seq = torch.cat([seq, torch.multinomial(p, 1, generator=gen)], 1)
    return seq


def layer_type(name: str) -> str:
    return name.split(".")[-1]


def part_layers(m32, acts):
    rows = []
    for name, mod in m32.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        W = mod.weight.detach()
        x32 = acts[name]
        x64, xb = x32.double(), x32.bfloat16()
        ref = x64 @ W.double().T
        Wb = W.bfloat16()
        N, K = W.shape
        r = {"name": name, "type": layer_type(name), "N": N, "K": K}
        rms = Wb.float().pow(2).mean(1).sqrt()
        r["row_max_over_rms_median"] = float((Wb.float().abs().amax(1) / rms).median())
        r["row_max_over_rms_max"] = float((Wb.float().abs().amax(1) / rms).max())
        r["bf16"] = rel(F.linear(xb, Wb), ref)

        q8, s8, _ = Q.dynamically_quantize_per_channel(Wb.float(), -128, 127, torch.int8)
        lin8 = Q.WeightOnlyInt8Linear(K, N).to(DEVICE)
        lin8.weight.copy_(q8)
        lin8.scales.copy_(s8.to(torch.bfloat16))
        r["int8"] = rel(lin8(xb), ref)
        r["int8_ideal"] = rel(x64 @ (q8.double() * s8.double()[:, None]).T, ref)
        r["int8_bf16_scales_fp64_math"] = rel(x64 @ (q8.double() * s8.bfloat16().double()[:, None]).T, ref)

        for g in GROUPS:
            wd, codes, scales, zeros = int4_dequant64(Wb, g)
            p, sz = prepare_int4_fixed(Wb, g, 8)
            r[f"int4_g{g}"] = rel(Q.linear_forward_int4(xb, p, sz, N, g), ref)
            r[f"int4_g{g}_ideal"] = rel(x64 @ wd.T, ref)
            if g == 128:
                near = int4_nearest_codes(Wb, scales, zeros, g)
                r["int4_g128_nonnearest_frac"] = float((near != codes).double().mean())
                wdn, _, _, _ = int4_dequant64(Wb, g, codes=near)
                r["int4_g128_nearest_ideal"] = rel(x64 @ wdn.T, ref)
                r["int4_g128_weight_rel_err"] = rel(wd, W.double())
                r["int4_g128_nearest_weight_rel_err"] = rel(wdn, W.double())
        r["int8_weight_rel_err"] = rel(q8.double() * s8.double()[:, None], W.double())
        rows.append(r)
    keys = [k for k in rows[0] if k not in ("name", "type", "N", "K")]
    summary = {}
    for t in ["wqkv", "wo", "w1", "w3", "w2", "output"]:
        rs = [r for r in rows if r["type"] == t]
        summary[t] = {k: sorted(r[k] for r in rs)[len(rs) // 2] for k in keys}
        summary[t]["layers"] = len(rs)
        summary[t]["max_int4_g128"] = max(r["int4_g128"] for r in rs)
    return {"rows": rows, "median_by_type": summary}


def compare_logits(lq, l32, seqs):
    kl, top1, nll = [], [], []
    for b in range(seqs.shape[0]):
        lp32 = torch.log_softmax(l32[b].float(), -1)
        lpq = torch.log_softmax(lq[b].float(), -1)
        kl.append((lp32.exp() * (lp32 - lpq)).sum(-1))
        top1.append((l32[b].argmax(-1) == lq[b].argmax(-1)).double())
        nll.append(-lpq[:-1].gather(-1, seqs[b, 1:, None]).squeeze(-1))
    kl, top1, nll = torch.cat(kl), torch.cat(top1), torch.cat(nll)
    return {"kl_mean": float(kl.mean()), "kl_p99": float(kl.quantile(0.99)), "top1_agree": float(top1.mean()),
            "ppl": float(nll.double().mean().exp())}


def part_accuracy():
    m32 = load_llama2c(MODEL, torch.float32, EVAL_B)
    t0 = time.time()
    seqs = sample_eval_tokens(m32)
    print(f"  sampled {tuple(seqs.shape)} in {time.time() - t0:.0f}s")
    acts, hooks = {}, []
    for name, mod in m32.named_modules():
        if isinstance(mod, torch.nn.Linear):
            hooks.append(mod.register_forward_hook(
                lambda mod, inp, out, name=name: acts.__setitem__(name, inp[0][:ACT_SEQS].reshape(-1, inp[0].shape[-1]).detach().clone())))
    with torch.no_grad():
        l32 = forward_full(m32, seqs).float()
    for h in hooks:
        h.remove()
    with torch.no_grad():
        layers = part_layers(m32, acts)
    for t, s in layers["median_by_type"].items():
        print(f"  {t:6s} bf16 {s['bf16']:.2e} int8 {s['int8']:.2e} ideal {s['int8_ideal']:.2e} "
              f"int4 g32 {s['int4_g32']:.2e} g128 {s['int4_g128']:.2e} nonnearest {s['int4_g128_nonnearest_frac']:.3f}")
    del acts
    nll32 = []
    for b in range(EVAL_B):
        lp = torch.log_softmax(l32[b], -1)
        nll32.append(-lp[:-1].gather(-1, seqs[b, 1:, None]).squeeze(-1))
    model_rows = [{"config": "fp32", "linear_bytes": linear_bytes(m32), "linear_weights": linear_weights(m32),
                   "ppl": float(torch.cat(nll32).double().mean().exp())}]
    del m32
    for cfg in ["bf16", "int8"] + [f"int4-g{g}" for g in GROUPS]:
        m = quantize_model(load_llama2c(MODEL, torch.bfloat16, EVAL_B), cfg)
        with torch.no_grad():
            lq = forward_full(m, seqs)
        r = {"config": cfg, "linear_bytes": linear_bytes(m), "linear_weights": linear_weights(m), **compare_logits(lq, l32, seqs)}
        model_rows.append(r)
        print(f"  model {cfg:10s} KL {r['kl_mean']:.2e} top1 {r['top1_agree']:.4f} ppl {r['ppl']:.3f}")
        del m, lq
    torch.cuda.empty_cache()
    return {"layers": layers, "model": model_rows, "eval_tokens": list(seqs.shape)}


# ---------------------------------------------------------------- 4. speed

def int8_linear(x, weight, scales):
    """WeightOnlyInt8Linear.forward's body."""
    return F.linear(x, weight.to(dtype=x.dtype)) * scales


int8_linear_compiled = torch.compile(int8_linear, dynamic=False)


def bench(call, n_rot):
    for i in range(max(3, n_rot)):
        call(i % n_rot)
    sync()
    t0 = time.perf_counter()
    for i in range(n_rot):
        call(i)
    sync()
    est = (time.perf_counter() - t0) / n_rot
    reps = math.ceil(max(n_rot, min(4000, int(TARGET_S / est))) / n_rot) * n_rot
    eager = []
    for _ in range(3):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for i in range(reps):
            call(i % n_rot)
        en.record()
        sync()
        eager.append(st.elapsed_time(en) * 1e3 / reps)
    greps = math.ceil(min(reps, GRAPH_MAX_CALLS) / n_rot) * n_rot
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call(0)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(greps):
            call(i % n_rot)
    g.replay()
    sync()
    graph = []
    for _ in range(3):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        g.replay()
        en.record()
        sync()
        graph.append(st.elapsed_time(en) * 1e3 / greps)
    del g
    return sorted(eager)[1], sorted(graph)[1]


def uses_extern_mm(x, w, s) -> bool | None:
    try:
        from torch._inductor.utils import run_and_get_code
        _, codes = run_and_get_code(int8_linear_compiled, x, w, s)
        return any("extern_kernels." in c for c in codes) if codes else None
    except Exception:
        return None


def part_speed():
    rows, itk_rows = [], []
    for model_name, layer, N, K in SHAPES:
        torch.manual_seed(SEED)
        W = (torch.randn(N, K, device=DEVICE) * 0.02).bfloat16()
        q8, s8, _ = Q.dynamically_quantize_per_channel(W.float(), -128, 127, torch.int8)
        s8 = s8.bfloat16()
        packs = {(g, itk): prepare_int4_fixed(W, g, itk) for g in (32, 128) for itk in INNER_K_TILES
                 if itk == 8 or g == 128}

        def copies(tensors):
            nbytes = sum(t.numel() * t.element_size() for t in tensors)
            n = max(1, math.ceil(ROT_BYTES / nbytes))
            return [tuple(t.clone() for t in tensors) for _ in range(n)], nbytes

        sets = {"bf16": copies([W]), "int8": copies([q8, s8])}
        for (g, itk), (p, sz) in packs.items():
            sets[f"int4-g{g}-itk{itk}"] = copies([p, sz])
        torch.cuda.empty_cache()
        for Mtok in M_LIST:
            x = torch.randn(Mtok, K, device=DEVICE, dtype=torch.bfloat16)
            row = {"model": model_name, "layer": layer, "N": N, "K": K, "M": Mtok}
            ref = F.linear(x, W)
            assert torch.equal(int8_linear(x, q8, s8), Q.WeightOnlyInt8Linear.forward(
                types.SimpleNamespace(weight=q8, scales=s8), x))
            row["int8_compile_extern_mm"] = uses_extern_mm(x, q8, s8)
            meths = {
                "bf16": ("bf16", lambda S, i: F.linear(x, S[i][0])),
                "int8-eager": ("int8", lambda S, i: int8_linear(x, *S[i])),
                "int8-compile": ("int8", lambda S, i: int8_linear_compiled(x, *S[i])),
                "int4-g32": ("int4-g32-itk8", lambda S, i: Q.linear_forward_int4(x, *S[i], N, 32)),
                "int4-g128": ("int4-g128-itk8", lambda S, i: Q.linear_forward_int4(x, *S[i], N, 128)),
            }
            for mname, (sname, fn) in meths.items():
                S, nbytes = sets[sname]
                e_us, g_us = bench(lambda i: fn(S, i), len(S))
                row[mname] = {"eager_us": e_us, "graph_us": g_us, "weight_bytes": nbytes,
                              "graph_GBps": nbytes / g_us / 1e3}
            row["int4_g128_rel_err_vs_bf16"] = rel(Q.linear_forward_int4(x, *packs[(128, 8)], N, 128), ref.double())
            rows.append(row)
            print(f"  {model_name} {layer:6s} M={Mtok:5d} " + " ".join(
                f"{k} {row[k]['graph_us']:.1f}/{row[k]['eager_us']:.1f}" for k in meths) +
                f" extern_mm={row['int8_compile_extern_mm']}")
            if Mtok in ITK_M:
                r2 = {"model": model_name, "layer": layer, "M": Mtok}
                for itk in INNER_K_TILES:
                    S, _ = sets[f"int4-g128-itk{itk}"]
                    r2[f"itk{itk}_graph_us"] = bench(lambda i: Q.linear_forward_int4(x, *S[i], N, 128), len(S))[1]
                itk_rows.append(r2)
        del sets, packs
        torch.cuda.empty_cache()
    return {"rows": rows, "inner_k_tiles": itk_rows}


def per_layer_sums(speed):
    """Sum of one transformer layer's linears (w1/w3 counted twice); 110M also x12 + output."""
    out = []
    meths = ["bf16", "int8-eager", "int8-compile", "int4-g32", "int4-g128"]
    for model_name in ("110M", "7B"):
        for Mtok in M_LIST:
            rs = {r["layer"]: r for r in speed["rows"] if r["model"] == model_name and r["M"] == Mtok}
            row = {"model": model_name, "M": Mtok}
            for timing in ("graph_us", "eager_us"):
                for m in meths:
                    t = rs["wqkv"][m][timing] + rs["wo"][m][timing] + 2 * rs["w1/w3"][m][timing] + rs["w2"][m][timing]
                    if model_name == "110M":
                        t = 12 * t + rs["output"][m][timing]
                    row[f"{m}_{timing}"] = t
            out.append(row)
    return out


# ---------------------------------------------------------------- report

def write_md(R):
    L = ["# 0017 results", ""]
    u = R["upstream"]
    L += ["## 1. Upstream quantize.py at HEAD (torch " + torch.__version__ + ")", "", "| attempt | result | where |", "|---|---|---|"]
    for c in u["checks"]:
        res = ("ok" + (f" ({c['detail']})" if c.get("detail") else "")) if c["ok"] else c["error"]
        L.append(f"| {c['attempt']} | {res} | {('`' + c['where'] + '`') if c.get('where') else ''} |")
    L += ["", "int4 packing variants for `_convert_weight_to_int4pack` (stories110M layers.0 w1, g128, inner_k_tiles 8, 16 tokens):", "",
          "| packing | packed shape | output shape | rel err vs dequantized weight (fp64) |", "|---|---|---|---|"]
    for r in u["packing"]:
        L.append(f"| {r['variant']} | {r.get('packed_shape', '')} | {r.get('out_shape', '')} | "
                 f"{('%.2e' % r['rel_err_vs_dequant']) if 'rel_err_vs_dequant' in r else r.get('error', '')} |")
    p = u["stories15M_padding"]
    L += ["", f"stories15M int4 storage of all Linear layers (in_features 288 padded to {p['padded_in_features']} by upstream):", "",
          "| groupsize | padded layers | bf16 MiB | upstream int4 MiB | pad to lcm(g, 128) MiB | unpadded MiB |", "|---|---|---|---|---|---|"]
    for r in p["rows"]:
        L.append(f"| {r['groupsize']} | {r['padded_layers']} / {r['layers']} | {r['bf16_bytes'] / 2**20:.1f} | {r['upstream_bytes'] / 2**20:.1f} | "
                 f"{r['pad_to_lcm_bytes'] / 2**20:.1f} | {r['no_pad_bytes'] / 2**20:.1f} |")
    L += ["", f"Padded forward of `{p['layer']}` vs fp64 dequantized weight: rel err {p['padded_forward_rel_err']:.2e}"]

    a = R["accuracy"]
    L += ["", f"## 2. Per-layer output error vs fp64 (stories110M, {ACT_SEQS} x {EVAL_T} real fp32 activations, median over layers)", "",
          "Relative Frobenius error of the layer output. `ideal` = same codes, fp64 math with fp32 activations.", "",
          "| layer | row max/RMS | bf16 | int8 (upstream) | int8 ideal | int8 bf16 scales, fp64 | int4 g32 | g64 | g128 | g256 | g128 ideal | g128 nearest codes | g128 non-nearest codes |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t, s in a["layers"]["median_by_type"].items():
        L.append(f"| {t} ({s['layers']}) | {s['row_max_over_rms_median']:.1f} | {s['bf16']:.2e} | {s['int8']:.2e} | {s['int8_ideal']:.2e} | "
                 f"{s['int8_bf16_scales_fp64_math']:.2e} | {s['int4_g32']:.2e} | {s['int4_g64']:.2e} | {s['int4_g128']:.2e} | {s['int4_g256']:.2e} | "
                 f"{s['int4_g128_ideal']:.2e} | {s['int4_g128_nearest_ideal']:.2e} | {100 * s['int4_g128_nonnearest_frac']:.2f}% |")
    L += ["", f"## 3. Whole model vs fp32 (stories110M, {a['eval_tokens'][0]} x {a['eval_tokens'][1]} tokens sampled from the fp32 model at T = 1)", "",
          "| config | Linear MiB | bits / weight | mean KL | p99 KL | top-1 agree | ppl |", "|---|---|---|---|---|---|---|"]
    for r in a["model"]:
        bits = 8 * r["linear_bytes"] / r["linear_weights"]
        L.append(f"| {r['config']} | {r['linear_bytes'] / 2**20:.1f} | {bits:.2f} | {r.get('kl_mean', 0):.2e} | {r.get('kl_p99', 0):.2e} | "
                 f"{r.get('top1_agree', 1):.4f} | {r['ppl']:.3f} |")

    sp = R["speed"]
    meths = ["bf16", "int8-eager", "int8-compile", "int4-g32", "int4-g128"]
    for timing, title in (("graph_us", "CUDA-graph replay (GPU time)"), ("eager_us", "eager launches (wall per call)")):
        L += ["", f"## 4. Linear latency, {title}, µs; weights rotated past L2", "",
              "| model | layer | N x K | M | " + " | ".join(meths) + " | int8-compile uses extern mm |",
              "|---|---|---|---|" + "---|" * len(meths) + "---|"]
        for r in sp["rows"]:
            L.append(f"| {r['model']} | {r['layer']} | {r['N']}x{r['K']} | {r['M']} | " +
                     " | ".join(f"{r[m][timing]:.1f}" for m in meths) + f" | {r['int8_compile_extern_mm']} |")
    L += ["", "### Effective weight bandwidth at M = 1 (CUDA graph), GB/s", "", "| model | layer | " + " | ".join(meths) + " |",
          "|---|---|" + "---|" * len(meths)]
    for r in sp["rows"]:
        if r["M"] == 1:
            L.append(f"| {r['model']} | {r['layer']} | " + " | ".join(f"{r[m]['graph_GBps']:.0f}" for m in meths) + " |")
    L += ["", "### One layer's linears (110M: 12 layers + output), µs, and speedup vs bf16", "",
          "| model | M | timing | " + " | ".join(meths) + " |", "|---|---|---|" + "---|" * len(meths)]
    for r in R["layer_sums"]:
        for timing in ("graph_us", "eager_us"):
            b = r[f"bf16_{timing}"]
            L.append(f"| {r['model']} | {r['M']} | {timing[:-3]} | " +
                     " | ".join(f"{r[f'{m}_{timing}']:.0f} ({b / r[f'{m}_{timing}']:.2f}×)" for m in meths) + " |")
    L += ["", "### int4 g128 inner_k_tiles, CUDA graph µs", "", "| model | layer | M | 2 | 4 | 8 |", "|---|---|---|---|---|---|"]
    for r in sp["inner_k_tiles"]:
        L.append(f"| {r['model']} | {r['layer']} | {r['M']} | {r['itk2_graph_us']:.1f} | {r['itk4_graph_us']:.1f} | {r['itk8_graph_us']:.1f} |")
    (HERE / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    t_start = time.time()
    torch.manual_seed(SEED)
    for n in (MODEL, "stories15M"):
        if not (CKPT_DIR / f"{n}.bin").exists():
            raise SystemExit(f"missing {CKPT_DIR / (n + '.bin')}; see README for the download commands")
    R = {"meta": {"gpu": GPU_INFO, "python": platform.python_version(), "torch": torch.__version__,
                  "platform": platform.platform(), "seed": SEED,
                  "checkpoints": {n: sha256(CKPT_DIR / f"{n}.bin")[:16] for n in (MODEL, "stories15M")}}}
    print("1. upstream as-is")
    R["upstream"] = part_upstream()
    print("2-3. accuracy")
    t0 = time.time()
    R["accuracy"] = part_accuracy()
    print(f"   {time.time() - t0:.0f}s")
    print("4. speed")
    t0 = time.time()
    R["speed"] = part_speed()
    R["layer_sums"] = per_layer_sums(R["speed"])
    print(f"   {time.time() - t0:.0f}s")
    R["meta"]["wall_seconds"] = round(time.time() - t_start, 1)
    R["meta"]["peak_memory_mib"] = gpu_guard.peak_memory_mib()
    (HERE / "results.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    write_md(R)
    print(f"done in {R['meta']['wall_seconds']}s")


if __name__ == "__main__":
    main()
