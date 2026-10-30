# 0006 — minbpe BPE training/encoding vs vocab size, compared with tiktoken

- **Upstream**: [karpathy/minbpe](https://github.com/karpathy/minbpe) (pinned by the `upstream/minbpe` submodule)
- **Component**: `minbpe/regex.py` `RegexTokenizer.train` / `encode_ordinary` (+ `minbpe/gpt4.py` `GPT4Tokenizer`)
- **Baseline**: tiktoken 0.14.0 (Rust `CoreBPE`, single-threaded `encode_ordinary`)
- **Hardware**: CPU only (Intel Core, Python 3.13.3); the only download is the ~1.7 MB `cl100k_base` ranks file
- **Run**: `.venv\Scripts\python.exe experiments\0006-minbpe-vs-tiktoken\run.py` (~3 min; set `HTTPS_PROXY` if tiktoken needs a proxy to fetch `cl100k_base`)

## Question

How do minbpe's pure-Python BPE training and encoding costs scale with vocab size, how far is minbpe from tiktoken when both use exactly the same merges, and how much of that gap comes from re-merging repeated chunks?

## Setup

- Training text: the first half of minbpe's bundled `tests/taylorswift.txt` (92,874 bytes), GPT-4 split pattern, vocab sizes 512 / 1024 / 2048 / 4096 / 8192. Each vocab size is trained from scratch with the unmodified upstream `train`.
- Encoding corpora: the held-out second half of the file (prose, 92,894 bytes) and minbpe's own Python sources (code, 23,804 bytes, out of domain).
- Three encoders per trained vocab:
  - `minbpe`: upstream `encode_ordinary`.
  - `minbpe+cache`: a subclass in `run.py` that memoizes `_encode_chunk` by chunk bytes. The cache is cleared before every timed run.
  - `tiktoken`: a `tiktoken.Encoding` built from the same merges (rank = minbpe token id) and the same regex.
- `gpt4_cl100k`: minbpe `GPT4Tokenizer` (cl100k merges recovered in Python) vs tiktoken `cl100k_base`, on the full prose file and the code corpus.
- Timing is the best of 3 runs for minbpe (2 runs for GPT4Tokenizer) and the best of 20 runs for tiktoken. Every configuration asserts that the cached encoder's ids equal minbpe's, that minbpe round-trips the text, and compares token ids with tiktoken.

## Findings

1. **minbpe and tiktoken produce identical token ids in all 10 configurations**, including GPT4Tokenizer vs `cl100k_base`. No trained vocab contained duplicate byte strings, so minbpe merges load directly into tiktoken as ranks; this is a cheap way to get a fast encoder for a minbpe-trained vocab.
2. **Training is O(merges × corpus size).** Every merge re-counts all pairs over the whole corpus: 2.9 s for vocab 512, then 7.5 s, 15.7 s, and 29.8 s for vocab 4096 on 93 KB. The cost per merge falls only slowly, from 11.4 to 7.8 ms, as the sequences shrink. Extrapolating linearly (about 8 ms per merge per 93 KB), a 100k vocab on 1 GB of text would take months with this loop.
3. **Upstream `train` crashes when the text runs out of pairs.** With vocab 8192, every regex chunk had become a single token after 7,252 merges, so `max(stats)` raised `ValueError: max() iterable argument is empty` after 51 s and the merges were lost. The largest vocab this text supports is 7,508. `train` has no early stop or clear error for this case.
4. **tiktoken encodes 12–29× faster, and the gap grows with vocab size.** minbpe's prose throughput drops from 1.00 MB/s at vocab 512 to 0.88 MB/s at vocab 4096 and 0.75 MB/s with cl100k, because each chunk runs more merge rounds, each costing `get_stats` plus a `min` over pairs. tiktoken stays at 18–22 MB/s on prose. The speedup on prose is 18× at vocab 512, 24× at 4096, and 29× with cl100k.
5. **Re-encoding repeated chunks is 60–67% of minbpe's encode time.** A per-chunk memo, with identical output, gives 2.5–3.0× on prose and 2.5× on code. tiktoken is still 5–8× faster than the cached Python version.
6. **Compression returns diminish quickly on this small corpus.** On held-out prose, bytes per token rise from 2.11 (vocab 512) to 2.54, 2.80, and 2.99 (vocab 4096). On out-of-domain code they only reach 1.71, compared with 3.77 (prose) and 4.16 (code) for cl100k. `GPT4Tokenizer.__init__` alone takes 7.9 s to recover the 100k merges in Python.

Full numbers: [results.md](results.md) / [results.json](results.json).
