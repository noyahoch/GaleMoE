"""Configuration for router intervention experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class ExperimentConfig:
    """Config for router intervention experiments (project-out, interventions).

    Requires precomputed expert SVD vectors in svd_dir, named
    ``{model_tag}_layer{layer_idx}_expert{i}.pkl``.
    """

    svd_dir: str
    layer_idx: int
    num_experts: int = 8
    model_id: str = "mistralai/Mixtral-8x7B-v0.1"
    model_tag: str = "mistralai_Mixtral_8x7B_v0.1"
    output_file: str = "results_project_out.json"
    num_samples: int = 200
    seq_len: int = 64
    batch_size: int = 4
    seed: int = 42
    variations: Sequence[str] = ("svd", "orthogonal", "random", "zero", "shuffle")
    dataset: str = "wikitext"
    text_file: str | None = None
    top_k: Sequence[int] = (1,)
    load_in_8bit: bool = False  # If True, use BitsAndBytes 8-bit quantization (gate-hook style; halves weight memory)
