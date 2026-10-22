"""Keep experiments within the local GPU budget (RTX 4060 Laptop, 8 GB).

Every GPU experiment must call `acquire()` before touching CUDA. It refuses to
run when the GPU is already busy and caps this process's memory share.
"""
import shutil
import subprocess

MAX_MEMORY_FRACTION = 0.6
MAX_BUSY_UTIL_PERCENT = 50
MIN_FREE_MIB = 4096


def query_nvidia_smi() -> dict | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    out = subprocess.run(
        [exe, "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30,
    ).stdout.strip().splitlines()[0]
    name, total, used, util, temp = [s.strip() for s in out.split(",")]
    return {
        "name": name,
        "memory_total_mib": int(total),
        "memory_used_mib": int(used),
        "utilization_percent": int(util),
        "temperature_c": int(temp),
    }


def acquire(max_fraction: float = MAX_MEMORY_FRACTION) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; GPU experiments need the CUDA build of torch")
    info = query_nvidia_smi() or {}
    free = info.get("memory_total_mib", 0) - info.get("memory_used_mib", 0)
    if info and free < MIN_FREE_MIB:
        raise RuntimeError(f"GPU has only {free} MiB free (< {MIN_FREE_MIB}); refusing to run")
    if info and info["utilization_percent"] > MAX_BUSY_UTIL_PERCENT:
        raise RuntimeError(f"GPU is {info['utilization_percent']}% busy; refusing to run")
    torch.cuda.set_per_process_memory_fraction(max_fraction, 0)
    info["memory_fraction_cap"] = max_fraction
    info["torch"] = torch.__version__
    info["cuda"] = torch.version.cuda
    return info


def peak_memory_mib() -> float:
    import torch

    return round(torch.cuda.max_memory_allocated() / 2**20, 1)
