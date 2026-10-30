"""minbpe BPE training / encoding cost vs vocab size, compared with tiktoken.

Trains karpathy/minbpe's `RegexTokenizer` (GPT-4 split pattern) on the first half
of the bundled `tests/taylorswift.txt`, then encodes held-out prose and code with:

- `minbpe`: upstream `encode_ordinary` (pure Python, re-merges every chunk),
- `minbpe+cache`: the same merges with a per-chunk memo (local variant, upstream untouched),
- `tiktoken`: tiktoken's Rust `CoreBPE` loaded with the *same* merges and regex.

Token ids must match across all three. Finally compares minbpe's `GPT4Tokenizer`
(cl100k merges recovered in Python) against tiktoken `cl100k_base`.
CPU only, no model downloads apart from the ~1.7 MB cl100k_base ranks file.
"""
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MINBPE_ROOT = HERE.parents[1] / "upstream" / "minbpe"
sys.path.insert(0, str(MINBPE_ROOT))

import tiktoken  # noqa: E402
from minbpe import GPT4Tokenizer, RegexTokenizer  # noqa: E402
from minbpe.regex import GPT4_SPLIT_PATTERN  # noqa: E402

SEED = 0
VOCAB_SIZES = [512, 1024, 2048, 4096, 8192]
CODE_FILES = ["minbpe/base.py", "minbpe/basic.py", "minbpe/regex.py", "minbpe/gpt4.py", "train.py"]


def bench(fn, reps):
    best, out = float("inf"), None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


class CachedRegexTokenizer(RegexTokenizer):
    """minbpe RegexTokenizer that memoizes `_encode_chunk` by chunk bytes."""

    def __init__(self, base: RegexTokenizer):
        super().__init__(pattern=base.pattern)
        self.merges, self.vocab = base.merges, base.vocab
        self._cache = {}

    def _encode_chunk(self, text_bytes):
        ids = self._cache.get(text_bytes)
        if ids is None:
            ids = self._cache[text_bytes] = super()._encode_chunk(text_bytes)
        return ids


def to_tiktoken(tok: RegexTokenizer, name: str):
    """Load minbpe merges into tiktoken: rank of a byte string = minbpe token id."""
    ranks = {}
    duplicates = 0
    for idx in sorted(tok.vocab):
        b = tok.vocab[idx]
        if b in ranks:
            duplicates += 1
            continue
        ranks[b] = idx
    enc = tiktoken.Encoding(name=name, pat_str=tok.pattern, mergeable_ranks=ranks, special_tokens={})
    return enc, duplicates


def encode_row(label, n_bytes, seconds, ids):
    return {"impl": label, "seconds": round(seconds, 5), "MB_per_s": round(n_bytes / seconds / 1e6, 4),
            "tokens": len(ids), "bytes_per_token": round(n_bytes / len(ids), 4)}


def compare_encoders(text, minbpe_tok, enc, reps_py=3, reps_rs=20):
    n_bytes = len(text.encode("utf-8"))
    t_py, ids_py = bench(lambda: minbpe_tok.encode_ordinary(text), reps_py)
    cached = CachedRegexTokenizer(minbpe_tok) if type(minbpe_tok) is RegexTokenizer else None
    rows = [encode_row("minbpe", n_bytes, t_py, ids_py)]
    if cached is not None:
        t_c, ids_c = bench(lambda: (cached._cache.clear(), cached.encode_ordinary(text))[1], reps_py)
        rows.append(encode_row("minbpe+cache", n_bytes, t_c, ids_c))
        assert ids_c == ids_py, "chunk cache changed the output"
    t_rs, ids_rs = bench(lambda: enc.encode_ordinary(text), reps_rs)
    rows.append(encode_row("tiktoken", n_bytes, t_rs, ids_rs))
    mismatch = next((i for i, (a, b) in enumerate(zip(ids_py, ids_rs)) if a != b), None)
    if mismatch is None and len(ids_py) != len(ids_rs):
        mismatch = min(len(ids_py), len(ids_rs))
    assert minbpe_tok.decode(ids_py) == text, "minbpe round-trip failed"
    return rows, {"n_bytes": n_bytes, "ids_identical": mismatch is None, "first_mismatch": mismatch,
                  "tiktoken_speedup_vs_minbpe": round(t_py / t_rs, 1)}


def git_rev(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def main():
    random.seed(SEED)
    prose = (MINBPE_ROOT / "tests" / "taylorswift.txt").read_text(encoding="utf-8")
    half = len(prose) // 2
    train_text, heldout_prose = prose[:half], prose[half:]
    code = "\n".join((MINBPE_ROOT / f).read_text(encoding="utf-8") for f in CODE_FILES)
    corpora = {"heldout_prose": heldout_prose, "code": code}

    train_rows, encode_rows, checks = [], [], []
    prev_merges = None
    for vocab_size in VOCAB_SIZES:
        tok = RegexTokenizer()
        t0 = time.perf_counter()
        tok.train(train_text, vocab_size)
        train_s = time.perf_counter() - t0
        merges = sorted(tok.merges.items(), key=lambda kv: kv[1])
        if prev_merges is not None:
            assert merges[:len(prev_merges)] == prev_merges, "smaller vocab is not a prefix"
        prev_merges = merges
        enc, dups = to_tiktoken(tok, f"minbpe_{vocab_size}")
        n_merges = vocab_size - 256
        train_rows.append({"vocab_size": vocab_size, "merges": n_merges, "train_seconds": round(train_s, 2),
                           "ms_per_merge": round(1000 * train_s / n_merges, 2),
                           "duplicate_byte_tokens": dups})
        print(f"vocab {vocab_size}: trained in {train_s:.1f}s ({dups} duplicate tokens)", flush=True)
        for corpus, text in corpora.items():
            rows, chk = compare_encoders(text, tok, enc)
            for r in rows:
                encode_rows.append({"tokenizer": f"trained_{vocab_size}", "corpus": corpus, **r})
            checks.append({"tokenizer": f"trained_{vocab_size}", "corpus": corpus, **chk})

    t0 = time.perf_counter()
    gpt4 = GPT4Tokenizer()
    gpt4_init_s = time.perf_counter() - t0
    cl100k = tiktoken.get_encoding("cl100k_base")
    print(f"GPT4Tokenizer init (recover_merges) {gpt4_init_s:.1f}s", flush=True)
    for corpus, text in {"full_prose": prose, "code": code}.items():
        rows, chk = compare_encoders(text, gpt4, cl100k, reps_py=2)
        for r in rows:
            encode_rows.append({"tokenizer": "gpt4_cl100k", "corpus": corpus, **r})
        checks.append({"tokenizer": "gpt4_cl100k", "corpus": corpus, **chk})

    result = {
        "experiment": HERE.name,
        "upstream": "karpathy/minbpe",
        "upstream_commit": git_rev(MINBPE_ROOT),
        "component": "minbpe.RegexTokenizer train/encode_ordinary (+ GPT4Tokenizer)",
        "baseline": f"tiktoken {tiktoken.__version__} (Rust CoreBPE, single thread)",
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "seed": SEED,
        "train_text_bytes": len(train_text.encode("utf-8")),
        "corpus_bytes": {k: len(v.encode("utf-8")) for k, v in {**corpora, "full_prose": prose}.items()},
        "gpt4_init_seconds": round(gpt4_init_s, 2),
        "training": train_rows,
        "encoding": encode_rows,
        "checks": checks,
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    def table(rows, cols):
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        return lines + ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]

    md = [f"# {HERE.name}", "",
          f"minbpe `{result['upstream_commit'][:7]}`, {result['baseline']}, Python {result['python']}.",
          f"Training text: first half of taylorswift.txt ({result['train_text_bytes']} bytes). "
          f"GPT4Tokenizer init: {result['gpt4_init_seconds']} s.", "",
          "## Training", ""]
    md += table(train_rows, ["vocab_size", "merges", "train_seconds", "ms_per_merge", "duplicate_byte_tokens"])
    md += ["", "## Encoding (best of N runs)", ""]
    md += table(encode_rows, ["tokenizer", "corpus", "impl", "seconds", "MB_per_s", "tokens", "bytes_per_token"])
    md += ["", "## minbpe vs tiktoken token ids", ""]
    md += table(checks, ["tokenizer", "corpus", "n_bytes", "ids_identical", "first_mismatch",
                         "tiktoken_speedup_vs_minbpe"])
    (HERE / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
