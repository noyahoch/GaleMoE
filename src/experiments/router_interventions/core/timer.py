"""Timing utilities."""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)


@contextmanager
def timer(label: str) -> Generator[None, None, None]:
    """Context manager that logs elapsed time for a block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.info("%s took %.2f s", label, time.perf_counter() - start)


class Timer:
    """Accumulates named timings and prints a report."""

    def __init__(self) -> None:
        self._starts: dict[str, float] = {}
        self._elapsed: dict[str, float] = {}

    def start(self, label: str) -> None:
        self._starts[label] = time.perf_counter()

    def stop(self, label: str) -> float:
        dt = time.perf_counter() - self._starts.pop(label)
        self._elapsed[label] = dt
        return dt

    def report(self) -> None:
        total = sum(self._elapsed.values()) or 1
        for label, dt in sorted(self._elapsed.items(), key=lambda x: -x[1]):
            logger.info("  %s: %.2fs (%.0f%%)", label, dt, dt / total * 100)
        logger.info("  TOTAL: %.2fs", total)
