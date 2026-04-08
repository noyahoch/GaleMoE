"""
Mixed pretraining dataloader for Experiment 1.

Training mixture and eval datasets are fully driven by the config —
no dataset names are hardcoded here.

Training sources (configured in data: section of base.yaml):
  general  : allenai/c4
  code     : bigcode/the-stack-dedup (data/<code_language>/ on the Hub)
  science  : allenai/peS2o (requires datasets<4 — see pyproject.toml)

Eval datasets (configured in data.indist_eval / data.ood_eval):
  indist   : any HF dataset with a split (default: allenai/c4 validation)
  ood      : any HF dataset with a split (default: ccdv/pubmed-summarization test)

All datasets are streamed — nothing downloaded upfront.
"""

from dataclasses import dataclass
from typing import Optional

from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset, interleave_datasets
from transformers import PreTrainedTokenizerBase

# ---------------------------------------------------------------------------
# Training mixture config
# ---------------------------------------------------------------------------

@dataclass
class MixtureConfig:
    general_prob: float = 0.60
    code_prob: float = 0.20
    science_prob: float = 0.20
    code_language: str = "python"
    extra_dataset: Optional[str] = None
    extra_prob: float = 0.0
    seed: int = 42

    def probabilities(self) -> list[float]:
        base = [self.general_prob, self.code_prob, self.science_prob]
        if self.extra_dataset and self.extra_prob > 0:
            scale = 1.0 - self.extra_prob
            return [p * scale for p in base] + [self.extra_prob]
        return base


# ---------------------------------------------------------------------------
# Build training mixture
# ---------------------------------------------------------------------------

def build_train_dataset(cfg: MixtureConfig):
    general = load_dataset("allenai/c4", "en", split="train", streaming=True)
    code    = load_dataset(
        "bigcode/the-stack-dedup",
        data_dir=f"data/{cfg.code_language}",
        split="train",
        streaming=True,
    )
    science = load_dataset(
        "allenai/peS2o",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    sources = [general, code, science]
    probs   = cfg.probabilities()

    if cfg.extra_dataset and cfg.extra_prob > 0:
        extra = load_dataset(
            cfg.extra_dataset,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        sources.append(extra)

    return interleave_datasets(sources, probabilities=probs,
                               seed=cfg.seed, stopping_strategy="all_exhausted")


# ---------------------------------------------------------------------------
# Build eval dataset — driven entirely by a config node
# ---------------------------------------------------------------------------

def build_eval_dataset_from_cfg(eval_cfg):
    """
    eval_cfg is a Namespace with fields: dataset, subset, split, text_col
    e.g. from config:
      indist_eval:
        dataset: allenai/c4
        subset: en
        split: validation
        text_col: text
    """
    kwargs = dict(split=eval_cfg.split, streaming=True, trust_remote_code=True)
    if eval_cfg.subset:
        return load_dataset(eval_cfg.dataset, eval_cfg.subset, **kwargs)
    return load_dataset(eval_cfg.dataset, **kwargs)


# ---------------------------------------------------------------------------
# Tokenised IterableDataset wrapper
# ---------------------------------------------------------------------------

class TokenisedDataset(IterableDataset):
    def __init__(self, hf_dataset, tokenizer: PreTrainedTokenizerBase,
                 max_length: int, text_col: str = "text"):
        self.hf_dataset = hf_dataset
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.text_col   = text_col

    def __iter__(self):
        for example in self.hf_dataset:
            text = example.get(self.text_col, "")
            if not text:
                continue
            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            yield {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_train_dataloader(
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_length: int = 512,
    mixture: Optional[MixtureConfig] = None,
    num_workers: int = 2,
) -> DataLoader:
    if mixture is None:
        mixture = MixtureConfig()
    # training sources use different column names — fall back gracefully
    class _MultiColDataset(TokenisedDataset):
        def __iter__(self):
            for example in self.hf_dataset:
                text = (example.get("text") or example.get("content")
                        or example.get("article") or "")
                if not text:
                    continue
                enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                     padding="max_length", return_tensors="pt")
                yield {"input_ids":      enc["input_ids"].squeeze(0),
                       "attention_mask": enc["attention_mask"].squeeze(0)}

    ds = _MultiColDataset(build_train_dataset(mixture), tokenizer, max_length)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)


def build_eval_dataloader(
    eval_cfg,                        # Namespace from config (indist_eval or ood_eval)
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_length: int = 512,
    num_workers: int = 1,
) -> DataLoader:
    hf_ds = build_eval_dataset_from_cfg(eval_cfg)
    ds    = TokenisedDataset(hf_ds, tokenizer, max_length, text_col=eval_cfg.text_col)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
