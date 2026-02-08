"""Data loading: abstract batch loader + implementations."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


class BatchLoader(ABC):
    """Abstract base for loaders that produce batches of token IDs for LM evaluation.

    Each batch is a tensor of shape [batch_size, seq_len] (input_ids; labels = input_ids for causal LM).
    """

    @abstractmethod
    def get_batches(self) -> List[torch.Tensor]:
        """Return a list of batches, each of shape [batch_size, seq_len]."""
        ...


def _tokenize_to_batches(
    tokenizer: AutoTokenizer,
    texts: List[str],
    seq_len: int,
    batch_size: int,
) -> List[torch.Tensor]:
    """Shared helper: tokenize texts and chunk into fixed-size batches.
    Concatenates all text then splits into seq_len chunks (good for long doc loss).
    """
    if not texts:
        raise ValueError("No texts to tokenize.")
    all_ids: List[torch.Tensor] = []
    for text in texts:
        enc = tokenizer(
            text, return_tensors="pt", padding=False, truncation=False
        )
        all_ids.append(enc["input_ids"].squeeze(0))
    ids = torch.cat(all_ids, dim=0)
    n_tokens = (ids.numel() // seq_len) * seq_len
    if n_tokens == 0:
        raise ValueError("Not enough data to create a single batch.")
    ids = ids[:n_tokens].view(-1, seq_len)
    return [
        ids[i : i + batch_size]
        for i in range(0, len(ids), batch_size)
    ]


def _tokenize_per_sample_to_batches(
    tokenizer: AutoTokenizer,
    texts: List[str],
    seq_len: int,
    batch_size: int,
    pad_token_id: int | None = None,
) -> List[torch.Tensor]:
    """Tokenize so each text is one sequence (truncate/pad to seq_len), then batch.
    Use this for 'titles' / short-sequence mode: allows large batch_size with small seq_len
    (e.g. 2000 wiki titles × 32 tokens → much lower memory than 4 × 512).
    """
    if not texts:
        raise ValueError("No texts to tokenize.")
    if pad_token_id is None and tokenizer.pad_token_id is not None:
        pad_token_id = tokenizer.pad_token_id
    elif pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id or 0
    rows: List[torch.Tensor] = []
    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            pad_to_multiple_of=1,
        )
        row = enc["input_ids"].squeeze(0)  # [seq_len]
        if row.shape[0] != seq_len:
            # pad to seq_len if tokenizer didn't (e.g. no pad_token_id set)
            if row.shape[0] < seq_len:
                pad = torch.full(
                    (seq_len - row.shape[0],),
                    pad_token_id,
                    dtype=row.dtype,
                )
                row = torch.cat([row, pad], dim=0)
            else:
                row = row[:seq_len]
        rows.append(row)
    ids = torch.stack(rows, dim=0)  # [num_samples, seq_len]
    return [
        ids[i : i + batch_size]
        for i in range(0, len(ids), batch_size)
    ]


def _resolve_cache_dir() -> str:
    if not os.environ.get("HF_DATASETS_CACHE") and not os.environ.get("HF_HOME"):
        cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    return os.environ.get("HF_DATASETS_CACHE") or str(
        Path.home() / ".cache" / "huggingface" / "datasets"
    )


def _load_wikipedia_streaming(cache_dir: str):
    """Load long-form wiki for chunked batching (wikitext/wikipedia)."""
    try:
        return load_dataset(
            "wikipedia", "20220301.en",
            split="train", streaming=True, cache_dir=cache_dir,
        )
    except Exception:
        try:
            return load_dataset(
                "wikitext", "wikitext-2-raw-v1",
                split="train", streaming=True, cache_dir=cache_dir,
            )
        except Exception as e:
            raise RuntimeError(f"Could not load dataset: {e}") from e


def _load_wiki_titles_streaming(cache_dir: str, prefer_gate_hook: bool = True):
    """Load Wikipedia for titles-only batching. If prefer_gate_hook=True (default),
    uses same source as gate-hook: wikimedia/wikipedia 20231101.en (title column).
    Falls back to wikipedia 20220301.en or wikitext otherwise.
    """
    if prefer_gate_hook:
        try:
            return load_dataset(
                "wikimedia/wikipedia", "20231101.en",
                split="train", streaming=True, cache_dir=cache_dir,
            )
        except Exception:
            pass
    try:
        return load_dataset(
            "wikipedia", "20220301.en",
            split="train", streaming=True, cache_dir=cache_dir,
        )
    except Exception:
        try:
            return load_dataset(
                "wikitext", "wikitext-2-raw-v1",
                split="train", streaming=True, cache_dir=cache_dir,
            )
        except Exception as e:
            raise RuntimeError(f"Could not load wiki titles dataset: {e}") from e


class WikitextBatchLoader(BatchLoader):
    """Streams Wikipedia (or wikitext-2), tokenizes, and returns fixed-size batches."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        num_samples: int = 200,
        seq_len: int = 512,
        batch_size: int = 4,
        min_text_length: int = 100,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.min_text_length = min_text_length
        self._cache_dir = _resolve_cache_dir()

    def _load_streaming_dataset(self):
        return _load_wikipedia_streaming(self._cache_dir)

    def _collect_texts(self) -> List[str]:
        ds = self._load_streaming_dataset()
        texts: List[str] = []
        for x in ds:
            if len(texts) >= self.num_samples:
                break
            if len(x["text"]) >= self.min_text_length:
                texts.append(x["text"])
        return texts

    def get_batches(self) -> List[torch.Tensor]:
        texts = self._collect_texts()
        return _tokenize_to_batches(
            self.tokenizer, texts, self.seq_len, self.batch_size
        )


class WikitextTitlesBatchLoader(BatchLoader):
    """One sequence per Wikipedia title (or per short text). Enables large batch_size with small seq_len.
    Use e.g. num_samples=2000, seq_len=32, batch_size=2000 to match gate-hook style runs.
    With prefer_gate_hook=True (default), uses same dataset as gate-hook: wikimedia/wikipedia 20231101.en (title column).
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        num_samples: int = 2000,
        seq_len: int = 32,
        batch_size: int = 2000,
        min_text_length: int = 1,
        prefer_gate_hook: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.min_text_length = min_text_length
        self.prefer_gate_hook = prefer_gate_hook
        self._cache_dir = _resolve_cache_dir()

    def _load_streaming_dataset(self):
        return _load_wiki_titles_streaming(
            self._cache_dir, prefer_gate_hook=self.prefer_gate_hook
        )

    def _collect_titles_or_short_texts(self) -> List[str]:
        ds = self._load_streaming_dataset()
        texts: List[str] = []
        for x in ds:
            if len(texts) >= self.num_samples:
                break
            # Gate-hook uses only "title"; wikimedia/wikipedia and wikipedia have "title"
            if "title" in x and (x["title"] or "").strip():
                texts.append((x["title"] or "").strip())
            elif len((x.get("text") or "").strip()) >= self.min_text_length:
                # Fallback: first line or first 200 chars (e.g. wikitext has no title)
                t = (x["text"] or "").strip()
                first_line = t.split("\n")[0].strip() or t[:200]
                texts.append(first_line)
        return texts

    def get_batches(self) -> List[torch.Tensor]:
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        texts = self._collect_titles_or_short_texts()
        return _tokenize_per_sample_to_batches(
            self.tokenizer, texts, self.seq_len, self.batch_size
        )


class TextListBatchLoader(BatchLoader):
    """Builds batches from an explicit list of text strings (e.g. eval set or custom corpus)."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        texts: List[str],
        seq_len: int = 512,
        batch_size: int = 4,
    ) -> None:
        self.tokenizer = tokenizer
        self.texts = texts
        self.seq_len = seq_len
        self.batch_size = batch_size

    def get_batches(self) -> List[torch.Tensor]:
        return _tokenize_to_batches(
            self.tokenizer, self.texts, self.seq_len, self.batch_size
        )
