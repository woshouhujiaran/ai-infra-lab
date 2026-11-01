"""StreamingLLM attention-sink KV eviction (upstream StartRecentKVCache) on Pythia-160M.

Token-by-token perplexity on PG19 test books, past the model's 2048-token
training context, for:
  dense        full KV cache, no eviction
  window       keep only the most recent C tokens (StartRecentKVCache start_size=0)
  sinkS        keep the first S tokens + the most recent C-S tokens
               (StartRecentKVCache start_size=S), with and without StreamingLLM's
               position shift (RoPE positions assigned by cache slot, not by text position)
  recompute    re-encode the last C+1 tokens from scratch for every prediction
               (sliding window with re-computation, the quality upper bound for a C-token budget)

Also: per-layer attention mass on the first tokens (why sinks exist), ms/token, and
the per-token cost of upstream's torch.cat-based eviction.

The model forward is a small re-implementation of GPT-NeoX (checked against
transformers' GPTNeoXForCausalLM) because upstream's pos-shift patch
(streaming_llm/pos_shift/modify_gpt_neox.py) targets the transformers 4.33 attention API.
"""
import argparse
import gc
import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "streaming-llm"))

from lab import gpu_guard  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from streaming_llm.kv_cache import StartRecentKVCache  # noqa: E402

SEED = 0
MODEL_ID, MODEL_REV = "EleutherAI/pythia-160m", "50f5173d932e8e61f858120bcb800b97af589f46"
DATA_ID, DATA_REV = "emozilla/pg19-test", "c5e39bf32e33f9111323aa68d7d9000d22722035"
DATA_FILE = "data/test-00000-of-00001-29a571947c0b5ccc.parquet"
NUM_BOOKS = 4
CACHE_SIZES = [256, 512, 1024]
SINK_SIZES = [1, 4, 8]
BUCKET = 512
RECOMPUTE_STRIDE = 16
RECOMPUTE_BATCH = 16
PROBE_TOKENS = 1024
PROBE_FROM = 256


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Pythia:
    def __init__(self, weights: dict, cfg: dict):
        self.w = {k: v.float().cuda() for k, v in weights.items()}
        self.layers = cfg["num_hidden_layers"]
        self.hidden = cfg["hidden_size"]
        self.heads = cfg["num_attention_heads"]
        self.head_dim = self.hidden // self.heads
        self.rot = int(self.head_dim * cfg["rotary_pct"])
        self.eps = cfg["layer_norm_eps"]
        base = cfg["rotary_emb_base"]
        self.inv_freq = (1.0 / base ** (torch.arange(0, self.rot, 2, dtype=torch.int64).float() / self.rot)).cuda()

    def cos_sin(self, pos):
        freqs = pos.float()[:, None] * self.inv_freq[None]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def rope(self, x, cos, sin):
        xr, xp = x[..., : self.rot], x[..., self.rot:]
        return torch.cat((xr * cos + rotate_half(xr) * sin, xp), dim=-1)

    def ln(self, x, name):
        return F.layer_norm(x, (self.hidden,), self.w[name + ".weight"], self.w[name + ".bias"], self.eps)

    def lin(self, x, name):
        return F.linear(x, self.w[name + ".weight"], self.w[name + ".bias"])

    def forward(self, ids, pos, cache=None, pos_shift=False, probe=None, last_only=False, keep_cache=True):
        """ids [B, T]; pos [T] RoPE positions of the new tokens.

        pos_shift=False: keys are cached already rotated at their text position (HF behaviour).
        pos_shift=True:  keys are cached un-rotated and rotated at their cache slot index
                         on every step (upstream modify_gpt_neox.py).
        """
        B, T = ids.shape
        assert T == 1 or cache is None, "multi-token input only without a cache (causal mask)"
        x = self.w["gpt_neox.embed_in.weight"][ids]
        cos, sin = self.cos_sin(pos)
        past = 0 if cache is None else cache[0][0].shape[2]
        if pos_shift:
            kcos, ksin = self.cos_sin(torch.arange(past + T, device=ids.device))
        new_cache = []
        for i in range(self.layers):
            p = f"gpt_neox.layers.{i}."
            h = self.ln(x, p + "input_layernorm")
            qkv = self.lin(h, p + "attention.query_key_value").view(B, T, self.heads, 3 * self.head_dim).transpose(1, 2)
            q, k, v = qkv.split(self.head_dim, dim=-1)
            q = self.rope(q, cos, sin)
            if not pos_shift:
                k = self.rope(k, cos, sin)
            if cache is not None:
                k = torch.cat((cache[i][0], k), dim=2)
                v = torch.cat((cache[i][1], v), dim=2)
            if keep_cache:
                new_cache.append([k, v])
            k_att = self.rope(k, kcos, ksin) if pos_shift else k
            if probe is not None:
                scores = (q @ k_att.transpose(-1, -2)) / math.sqrt(self.head_dim)
                mask = torch.ones(T, T, dtype=torch.bool, device=ids.device).tril()
                probs = scores.masked_fill(~mask, float("-inf")).softmax(-1)
                late = probs[:, :, PROBE_FROM:]
                last4 = sum(probs.diagonal(offset=-d, dim1=-2, dim2=-1)[..., PROBE_FROM - d:] for d in range(4))
                probe.append({"layer": i,
                              "mass_first1": round(late[..., :1].sum(-1).mean().item(), 4),
                              "mass_first4": round(late[..., :4].sum(-1).mean().item(), 4),
                              "mass_last4": round(last4.mean().item(), 4)})
                a = probs @ v
            else:
                a = F.scaled_dot_product_attention(q, k_att, v, is_causal=T > 1)
            a = self.lin(a.transpose(1, 2).reshape(B, T, self.hidden), p + "attention.dense")
            m = self.lin(F.gelu(self.lin(self.ln(x, p + "post_attention_layernorm"), p + "mlp.dense_h_to_4h")),
                         p + "mlp.dense_4h_to_h")
            x = x + a + m
        if last_only:
            x = x[:, -1:]
        x = self.ln(x, "gpt_neox.final_layer_norm")
        return F.linear(x, self.w["embed_out.weight"]), new_cache


def load_books(tok: Tokenizer, n_tokens: int):
    path = hf_hub_download(DATA_ID, DATA_FILE, repo_type="dataset", revision=DATA_REV)
    table = pq.read_table(path, columns=["short_book_title", "text"])
    titles, rows = [], []
    for title, text in zip(table.column("short_book_title").to_pylist(), table.column("text").to_pylist()):
        ids = tok.encode(text[: 12 * n_tokens]).ids
        if len(ids) >= n_tokens:
            titles.append(title)
            rows.append(ids[:n_tokens])
        if len(rows) == NUM_BOOKS:
            break
    return titles, torch.tensor(rows, dtype=torch.long).cuda()


@torch.inference_mode()
def stream(model: Pythia, ids, policy: dict):
    """Upstream eval_long_ppl.py loop: feed one token, score the next, then evict."""
    B, n = ids.shape
    T = n - 1
    cache_size, sinks, shift = policy["cache"], policy["sinks"], policy["pos_shift"]
    kv = None if cache_size is None else StartRecentKVCache(start_size=sinks, recent_size=cache_size - sinks)
    positions = torch.arange(n, device=ids.device)
    nll = torch.empty(B, T, device=ids.device)
    events = []
    cache = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for t in range(T):
        past = 0 if cache is None else cache[0][0].shape[2]
        pos = positions[past: past + 1] if shift else positions[t: t + 1]
        logits, cache = model.forward(ids[:, t: t + 1], pos, cache, shift)
        nll[:, t] = F.cross_entropy(logits[:, 0], ids[:, t + 1], reduction="none")
        if kv is not None and past + 1 > cache_size:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            cache = kv(cache)
            e1.record()
            events.append((e0, e1))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    evict_ms = sum(a.elapsed_time(b) for a, b in events)
    final_entries = cache[0][0].shape[2]
    return nll, {"ms_per_token": round(1000 * elapsed / T, 3),
                 "evict_ms_per_evicting_step": round(evict_ms / len(events), 3) if events else None,
                 "evicting_steps": len(events),
                 "final_cache_tokens": final_entries,
                 "kv_mib_per_sequence": round(final_entries * model.layers * 2 * model.hidden * 4 / 2**20, 1)}


@torch.inference_mode()
def recompute(model: Pythia, ids, cache_size: int, targets: list[int]):
    """NLL of token t+1 from a fresh forward over tokens [t-cache_size, t] (C+1 tokens, same span as the cache)."""
    B = ids.shape[0]
    jobs = [(b, t) for t in targets for b in range(B)]
    out = {}
    pos = torch.arange(cache_size + 1, device=ids.device)
    for i in range(0, len(jobs), RECOMPUTE_BATCH):
        chunk = jobs[i: i + RECOMPUTE_BATCH]
        x = torch.stack([ids[b, t - cache_size: t + 1] for b, t in chunk])
        logits, _ = model.forward(x, pos, last_only=True, keep_cache=False)
        y = torch.stack([ids[b, t + 1] for b, t in chunk])
        for (b, t), v in zip(chunk, F.cross_entropy(logits[:, -1], y, reduction="none").tolist()):
            out[(b, t)] = v
    return out


def ppl(x) -> float:
    return round(math.exp(float(x)), 3)


def upstream_commit() -> str:
    return subprocess.run(["git", "-C", str(ROOT / "upstream" / "streaming-llm"), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


@torch.inference_mode()
def validate(model: Pythia, ids, model_dir: str) -> dict:
    from transformers import GPTNeoXForCausalLM

    x = ids[:2, :512]
    ours, _ = model.forward(x, torch.arange(512, device=x.device))
    hf = GPTNeoXForCausalLM.from_pretrained(model_dir, dtype=torch.float32).cuda().eval()
    ref = hf(x).logits
    vs_hf = (ours - ref).abs().max().item()
    scale = ref.abs().max().item()
    del hf, ref
    gc.collect()
    torch.cuda.empty_cache()

    # token-by-token with a growing cache must equal the parallel forward
    cache, steps = None, []
    for t in range(256):
        logits, cache = model.forward(x[:, t: t + 1], torch.arange(t, t + 1, device=x.device), cache)
        steps.append(logits)
    vs_parallel = (torch.cat(steps, 1) - ours[:, :256]).abs().max().item()

    # RoPE is relative, so with no sinks the position shift must not change anything
    kv = StartRecentKVCache(start_size=0, recent_size=128)
    diffs = []
    caches = {False: None, True: None}
    for t in range(384):
        outs = {}
        for shift in (False, True):
            c = caches[shift]
            past = 0 if c is None else c[0][0].shape[2]
            p = past if shift else t
            outs[shift], c = model.forward(x[:, t: t + 1], torch.arange(p, p + 1, device=x.device), c, shift)
            caches[shift] = kv(c)
        diffs.append((outs[False] - outs[True]).abs().max().item())
    return {"max_abs_logit_transformers": round(scale, 3),
            "max_abs_logit_diff_vs_transformers": round(vs_hf, 6),
            "max_abs_logit_diff_streaming_vs_parallel": round(vs_parallel, 6),
            "max_abs_logit_diff_window_shift_vs_noshift": round(max(diffs), 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=4096, help="scored tokens per book")
    args = ap.parse_args()

    gpu = gpu_guard.acquire()
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    files = {f: hf_hub_download(MODEL_ID, f, revision=MODEL_REV) for f in ("config.json", "tokenizer.json", "model.safetensors")}
    cfg = json.loads(Path(files["config.json"]).read_text())
    tok = Tokenizer.from_file(files["tokenizer.json"])
    model = Pythia(load_file(files["model.safetensors"]), cfg)
    titles, ids = load_books(tok, args.tokens + 1)
    T = args.tokens
    print(f"books: {titles}", flush=True)

    t_start = time.perf_counter()
    checks = validate(model, ids, str(Path(files["config.json"]).parent))
    print(f"validation: {checks}", flush=True)

    probe = []
    with torch.inference_mode():
        model.forward(ids[:, :PROBE_TOKENS], torch.arange(PROBE_TOKENS, device=ids.device), probe=probe,
                      last_only=True, keep_cache=False)
    uniform = sum(1 / (t + 1) for t in range(PROBE_FROM, PROBE_TOKENS)) / (PROBE_TOKENS - PROBE_FROM)

    policies = [{"name": "dense", "cache": None, "sinks": 0, "pos_shift": False}]
    for c in CACHE_SIZES:
        policies.append({"name": f"window C={c}", "cache": c, "sinks": 0, "pos_shift": False})
        policies.append({"name": f"sink4 no-shift C={c}", "cache": c, "sinks": 4, "pos_shift": False})
        for s in SINK_SIZES:
            policies.append({"name": f"sink{s} shift C={c}", "cache": c, "sinks": s, "pos_shift": True})

    sub = list(range(RECOMPUTE_STRIDE - 1, T, RECOMPUTE_STRIDE))
    n_buckets = math.ceil(T / BUCKET)
    rows, nlls = [], {}
    for pol in policies:
        nll, perf = stream(model, ids, pol)
        nlls[pol["name"]] = nll
        row = {**pol, "ppl": ppl(nll.mean()), "ppl_subsample": ppl(nll[:, sub].mean()),
               "ppl_by_bucket": [ppl(nll[:, i * BUCKET:(i + 1) * BUCKET].mean()) for i in range(n_buckets)],
               "ppl_per_book": [ppl(nll[b].mean()) for b in range(ids.shape[0])], **perf}
        rows.append(row)
        print(f"{pol['name']}: ppl {row['ppl']}  buckets {row['ppl_by_bucket']}  {perf}", flush=True)

    dense = nlls["dense"]
    for c in CACHE_SIZES:
        t0 = time.perf_counter()
        far = recompute(model, ids, c, [t for t in sub if t >= c])
        vals = torch.tensor([[far.get((b, t), dense[b, t].item()) for t in sub] for b in range(ids.shape[0])])
        rows.append({"name": f"recompute C={c}", "cache": c, "sinks": 0, "pos_shift": False, "ppl": None,
                     "ppl_subsample": ppl(vals.mean()), "ppl_by_bucket": None, "ppl_per_book": None,
                     "ms_per_token": round(1000 * (time.perf_counter() - t0) / (len(far) / ids.shape[0]), 3),
                     "evict_ms_per_evicting_step": None, "evicting_steps": None, "final_cache_tokens": c + 1,
                     "kv_mib_per_sequence": None})
        print(f"recompute C={c}: ppl_subsample {rows[-1]['ppl_subsample']}", flush=True)
    total_s = round(time.perf_counter() - t_start, 1)

    result = {
        "experiment": HERE.name,
        "upstream": "mit-han-lab/streaming-llm",
        "upstream_commit": upstream_commit(),
        "component": "streaming_llm/kv_cache.py StartRecentKVCache (+ pos_shift/modify_gpt_neox.py, re-implemented)",
        "model": f"{MODEL_ID}@{MODEL_REV}",
        "model_train_context": cfg["max_position_embeddings"],
        "data": f"{DATA_ID}@{DATA_REV} first {len(titles)} books with >= {T + 1} tokens",
        "books": titles,
        "scored_tokens_per_book": T,
        "dtype": "fp32 (TF32 off)",
        "recompute_stride": RECOMPUTE_STRIDE,
        "gpu": gpu,
        "validation": checks,
        "attention_probe": {"tokens": PROBE_TOKENS, "queries_from": PROBE_FROM,
                            "uniform_mass_first1": round(uniform, 4), "layers": probe},
        "policies": rows,
        "runtime_s": total_s,
        "peak_memory_mib": gpu_guard.peak_memory_mib(),
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        f"Pythia-160M fp32, torch {torch.__version__}, {gpu.get('name', 'cuda')}; "
        f"{len(titles)} PG19 test books x {T} scored tokens; trained context {cfg['max_position_embeddings']}",
        "",
        "## Validation (max |logit diff|)",
        "",
        "| check | value |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in checks.items()]
    lines += [
        "",
        f"## Attention mass on the first tokens (dense, {PROBE_TOKENS} tokens, queries t >= {PROBE_FROM}; "
        f"uniform would put {uniform:.4f} on token 0)",
        "",
        "| layer | token 0 | tokens 0-3 | last 4 tokens |",
        "|---|---|---|---|",
    ]
    lines += [f"| {r['layer']} | {r['mass_first1']} | {r['mass_first4']} | {r['mass_last4']} |" for r in probe]
    lines += [
        "",
        f"## Perplexity (C = KV entries kept after eviction; subsample = every {RECOMPUTE_STRIDE}th token)",
        "",
        "| policy | ppl | ppl (subsample) | per book | ms/token | evict ms/evicting step | KV MiB/seq at end |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [f"| {r['name']} | {r['ppl']} | {r['ppl_subsample']} | {r['ppl_per_book']} | {r['ms_per_token']} | "
              f"{r['evict_ms_per_evicting_step']} | {r['kv_mib_per_sequence']} |" for r in rows]
    header = " | ".join(f"{i * BUCKET}-{min((i + 1) * BUCKET, T) - 1}" for i in range(n_buckets))
    lines += [
        "",
        f"## Perplexity by position ({BUCKET}-token buckets of the scored position)",
        "",
        f"| policy | {header} |",
        "|---|" + "---|" * n_buckets,
    ]
    lines += [f"| {r['name']} | " + " | ".join(str(v) for v in r["ppl_by_bucket"]) + " |"
              for r in rows if r["ppl_by_bucket"]]
    lines += ["", f"Total runtime {total_s} s, peak GPU memory {result['peak_memory_mib']} MiB."]
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
