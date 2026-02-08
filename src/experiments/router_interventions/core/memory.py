"""GPU memory reporting for router intervention experiments."""
from __future__ import annotations

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def _bytes_to_gib(b: float) -> float:
    return b / (1024**3)


def get_gpu_memory_gib() -> Tuple[float, float, float]:
    """Return (total_gib, allocated_gib, free_gib) for the current device, or (0, 0, 0) if not CUDA."""
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    # "Free" as in not allocated; total - reserved is a better proxy for what we can still allocate
    free = total - reserved
    return _bytes_to_gib(total), _bytes_to_gib(allocated), _bytes_to_gib(free)


def log_gpu_memory(label: str = "GPU memory") -> None:
    """Log current GPU memory usage (total, allocated, free in GiB)."""
    total, allocated, free = get_gpu_memory_gib()
    if total == 0:
        logger.info("%s: CUDA not available", label)
        return
    logger.info(
        "%s: total=%.2f GiB, allocated=%.2f GiB, free≈%.2f GiB",
        label, total, allocated, free,
    )


def require_gpu_memory_gib(need_gib: float, label: str = "Operation") -> bool:
    """
    Check if at least need_gib GiB is free on GPU. Log and return True if OK, False otherwise.
    Use before model.to("cuda") or similar to avoid OOM.
    """
    total, allocated, free = get_gpu_memory_gib()
    if total == 0:
        logger.warning("CUDA not available; cannot check memory")
        return True
    if free < need_gib:
        logger.warning(
            "%s needs ~%.2f GiB but only ~%.2f GiB free (total=%.2f GiB, allocated=%.2f GiB). "
            "Try smaller batch_size/seq_len/num_samples.",
            label, need_gib, free, total, allocated,
        )
        return False
    logger.info("%s: need ~%.2f GiB, free ~%.2f GiB", label, need_gib, free)
    return True
