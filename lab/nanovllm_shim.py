"""Import single nano-vllm components without pulling in the full engine.

`nanovllm/__init__.py` imports the LLM engine (flash-attn, transformers, a model
checkpoint). Registering a bare package object lets us import submodules such
as `nanovllm.engine.block_manager` directly from the pinned submodule source.
"""
import sys
import types
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NANOVLLM_SRC = ROOT / "upstream" / "nano-vllm" / "nanovllm"


@dataclass
class LiteConfig:
    """Fields of `nanovllm.config.Config` that the Scheduler reads."""
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = 1024


def install() -> None:
    if not NANOVLLM_SRC.is_dir():
        raise FileNotFoundError(f"{NANOVLLM_SRC} missing; run scripts/setup.ps1 first")
    if "nanovllm" in sys.modules:
        return
    for name, path in [("nanovllm", NANOVLLM_SRC),
                       ("nanovllm.engine", NANOVLLM_SRC / "engine"),
                       ("nanovllm.layers", NANOVLLM_SRC / "layers")]:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules[name] = pkg
    config = types.ModuleType("nanovllm.config")
    config.Config = LiteConfig
    sys.modules["nanovllm.config"] = config


def upstream_commit() -> str:
    import subprocess

    return subprocess.run(["git", "-C", str(NANOVLLM_SRC.parent), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
