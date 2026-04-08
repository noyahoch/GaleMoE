"""GPU memory reporting."""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def log_gpu_memory(label: str = "GPU") -> None:
    """Log current GPU memory usage."""
    if not torch.cuda.is_available():
        return
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    alloc = torch.cuda.memory_allocated(dev) / 1024**3
    total = props.total_memory / 1024**3
    logger.info("%s: %.1f / %.1f GiB allocated", label, alloc, total)
    if torch.cuda.device_count() > 1:
        total_alloc = sum(torch.cuda.memory_allocated(d) for d in range(torch.cuda.device_count())) / 1024**3
        logger.info("%s: %.1f GiB total across %d devices", label, total_alloc, torch.cuda.device_count())
