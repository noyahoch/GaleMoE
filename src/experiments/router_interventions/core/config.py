"""Experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class ExperimentConfig:
    """Config for router intervention experiments.

    SVD files: ``{svd_dir}/{model_tag}_layer{layer_idx}_expert{i}.pkl``
    Results:   ``{output_dir}/project_out_L{layer_idx}_k{...}_{dataset}_q{...}.json``
    """

    svd_dir: str
    layer_idx: int
    output_dir: str = "results"
    num_experts: int = 8
    model_id: str = "mistralai/Mixtral-8x7B-v0.1"
    model_tag: str = "mistralai_Mixtral_8x7B_v0.1"
    num_samples: int = 500
    seq_len: int = 32
    batch_size: int = 64
    seed: int = 42
    variations: Sequence[str] = ("svd", "orthogonal", "random", "zero", "shuffle")
    dataset: str = "wiki_titles"
    text_file: str | None = None
    top_k: Sequence[int] = (1,)
    quantization: str | None = None
    scale: float = 1.0  # Projection scale multiplier
    replace_weight_decay: float = 1.0  # For "replace" variant: w_i ∝ 1/(i+1)^decay
