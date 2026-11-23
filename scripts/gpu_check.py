"""Refuse to start a GPU step on hardware that cannot run it, with a short message and no traceback.

The training recipes use batch size 32, which needs a CUDA GPU with at least MIN_GB of memory. Smaller GPUs run out of
memory partway through a step, so the check happens before any work starts.
"""
import sys

MIN_GB = 24


def gpu_memory_gb():
    """Total memory of GPU 0 in GiB; 0.0 when there is no CUDA device; None when PyTorch is not installed."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 2 ** 30


def describe():
    gb = gpu_memory_gb()
    if gb is None:
        return 'PyTorch is not installed'
    if gb == 0.0:
        return 'no CUDA GPU was found'
    return f'the GPU has {gb:.1f} GiB'


def require_gpu(step, min_gb=MIN_GB):
    """Exit with status 2 and a one-line reason unless a CUDA GPU with at least min_gb GiB is present."""
    gb = gpu_memory_gb()
    if gb is not None and gb >= min_gb:
        return
    print(f'error: {step} needs a CUDA GPU with at least {min_gb} GiB of memory, but {describe()}.\n'
          f'       Run this step on a larger GPU (for example with --backend slurm on a cluster), or run the CPU-only '
          f'stages instead: scripts/reproduce.py --stages check data evaluate', file=sys.stderr, flush=True)
    sys.exit(2)
