"""Batch loaders for evaluation: wikitext, wiki titles, and plain text lists."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


class BatchLoader(ABC):
    """Abstract base: produce [batch_size, seq_len] token-ID batches."""

    @abstractmethod
    def get_batches(self) -> List[torch.Tensor]: ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _hf_cache_dir() -> str:
    """Ensure HF dataset cache dir exists and return its path."""
    if not os.environ.get("HF_DATASETS_CACHE") and not os.environ.get("HF_HOME"):
        d = Path.home() / ".cache" / "huggingface" / "datasets"
        d.mkdir(parents=True, exist_ok=True)
        os.environ["HF_DATASETS_CACHE"] = str(d)
    return os.environ.get("HF_DATASETS_CACHE") or str(
        Path.home() / ".cache" / "huggingface" / "datasets"
    )


def _chunk_to_batches(ids: torch.Tensor, seq_len: int, batch_size: int,
                      max_rows: int | None = None) -> List[torch.Tensor]:
    """Reshape flat token IDs into [batch_size, seq_len] batches."""
    n = (ids.numel() // seq_len) * seq_len
    if n == 0:
        raise ValueError("Not enough tokens for a single batch")
    rows = ids[:n].view(-1, seq_len)
    if max_rows is not None:
        rows = rows[:max_rows]
    return [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]


# ---------------------------------------------------------------------------
# Wikitext (long passages)
# ---------------------------------------------------------------------------

class WikitextBatchLoader(BatchLoader):
    """Wikipedia / wikitext-2 passages, tokenized into fixed-length chunks."""

    def __init__(self, tokenizer: AutoTokenizer, num_samples: int = 200,
                 seq_len: int = 512, batch_size: int = 4, min_len: int = 100) -> None:
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.min_len = min_len
        self._cache = _hf_cache_dir()

    def get_batches(self) -> List[torch.Tensor]:
        texts = self._collect()
        ids = torch.cat([
            self.tokenizer(t, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)
            for t in texts
        ])
        return _chunk_to_batches(ids, self.seq_len, self.batch_size, max_rows=self.num_samples)

    def _collect(self) -> List[str]:
        for name, cfg in [("wikipedia", "20220301.en"), ("wikitext", "wikitext-2-raw-v1")]:
            try:
                ds = load_dataset(name, cfg, split="train", streaming=True, cache_dir=self._cache)
                return [x["text"] for x, _ in zip(ds, range(self.num_samples)) if len(x["text"]) >= self.min_len]
            except Exception:
                continue
        raise RuntimeError("Could not load wikitext dataset")


# ---------------------------------------------------------------------------
# Wiki titles (short strings, pad to seq_len)
# ---------------------------------------------------------------------------

class WikiTitlesBatchLoader(BatchLoader):
    """Short Wikipedia titles padded to seq_len (like gate-hook experiments)."""

    def __init__(self, tokenizer: AutoTokenizer, num_samples: int = 500,
                 seq_len: int = 32, batch_size: int = 64) -> None:
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.batch_size = batch_size
        self._cache = _hf_cache_dir()

    def get_batches(self) -> List[torch.Tensor]:
        titles = self._load()
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        rows = torch.stack([
            self.tokenizer(t, return_tensors="pt", padding="max_length",
                           truncation=True, max_length=self.seq_len)["input_ids"].squeeze(0)
            for t in titles
        ])
        return [rows[i:i + self.batch_size] for i in range(0, len(rows), self.batch_size)]

    def _load(self) -> List[str]:
        for cfg in ("20220301.en", "20231101.en"):
            try:
                ds = load_dataset("wikimedia/wikipedia", cfg, split="train",
                                  streaming=True, cache_dir=self._cache, trust_remote_code=True)
                titles = []
                for x in ds:
                    if len(titles) >= self.num_samples:
                        break
                    t = (x.get("title") or (x.get("text") or "")[:80]).strip()
                    if t:
                        titles.append(t)
                if titles:
                    return titles
            except Exception:
                continue
        # Fall back to wikitext short lines
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=self._cache)
        titles = [r["text"].strip() for r in ds
                  if r["text"].strip() and not r["text"].startswith("=") and len(r["text"]) < 200]
        if not titles:
            raise RuntimeError("Could not load wiki titles from any source")
        return titles[:self.num_samples]


# ---------------------------------------------------------------------------
# Plain text list
# ---------------------------------------------------------------------------

class TextListBatchLoader(BatchLoader):
    """Fixed list of texts, tokenized into chunks."""

    def __init__(self, tokenizer: AutoTokenizer, texts: List[str],
                 seq_len: int = 512, batch_size: int = 4) -> None:
        self.tokenizer = tokenizer
        self.texts = texts
        self.seq_len = seq_len
        self.batch_size = batch_size

    def get_batches(self) -> List[torch.Tensor]:
        ids = torch.cat([
            self.tokenizer(t, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)
            for t in self.texts
        ])
        return _chunk_to_batches(ids, self.seq_len, self.batch_size)
