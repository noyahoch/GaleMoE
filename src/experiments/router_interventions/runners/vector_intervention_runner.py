"""Runner for inject/subtract (and project-out) interventions with token distribution comparison."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..core import (
    ExperimentConfig,
    BatchLoader,
    LossEvaluator,
    RouterManager,
    ExpertVectors,
    VectorIntervention,
    TokenDistributionComparator,
    confusion_matrix_top_k,
)
from ..core.data import TextListBatchLoader, WikitextBatchLoader, WikitextTitlesBatchLoader

logger = logging.getLogger(__name__)

VALID_VARIATIONS = ("svd", "orthogonal", "random")
VALID_INTERVENTIONS = ("project_out", "inject", "subtract")


def _make_batch_loader(config: ExperimentConfig, tokenizer: Any) -> BatchLoader:
    if config.dataset == "wikitext":
        return WikitextBatchLoader(
            tokenizer,
            num_samples=config.num_samples,
            seq_len=config.seq_len,
            batch_size=config.batch_size,
        )
    if config.dataset == "wikitext_titles":
        return WikitextTitlesBatchLoader(
            tokenizer,
            num_samples=config.num_samples,
            seq_len=config.seq_len,
            batch_size=config.batch_size,
        )
    if config.dataset == "text":
        if not config.text_file:
            raise ValueError("config.dataset is 'text' but config.text_file is not set")
        path = Path(config.text_file)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {path}")
        texts = path.read_text().strip().splitlines()
        texts = [t.strip() for t in texts if t.strip()]
        return TextListBatchLoader(
            tokenizer,
            texts=texts,
            seq_len=config.seq_len,
            batch_size=config.batch_size,
        )
    raise ValueError(f"Unknown dataset: {config.dataset}")


def _get_direction(
    variant: str,
    v_svd: Any,
    exp_idx: int,
    intervention: VectorIntervention,
    seed: int,
) -> Any:
    """Get direction vector(s) for intervention.
    
    Returns:
        - For variant="svd": returns v_svd as-is (may be [dim] or [k, dim])
        - For variant="orthogonal" or "random": returns [dim] tensor (single vector)
    """
    if variant not in VALID_VARIATIONS:
        raise ValueError(
            f"Unknown variant: {variant!r}. Must be one of {VALID_VARIATIONS}"
        )
    s = seed + exp_idx
    if variant == "svd":
        return v_svd
    dim = int(v_svd.shape[-1])
    if variant == "orthogonal":
        # Orthogonal to subspace (single vector is the k=1 case)
        subspace = v_svd.unsqueeze(0) if v_svd.ndim == 1 else v_svd
        return intervention.make_orthogonal_to_subspace(subspace, seed=s)
    elif variant == "random":
        return intervention.make_random(dim, seed=s)
    else:
        raise ValueError(f"Unknown variant: {variant!r}. Must be one of {VALID_VARIATIONS}")


def _apply_intervention(
    intervention: VectorIntervention,
    kind: str,
    row: Any,
    direction: Any,
    scale: float,
) -> Any:
    if kind not in VALID_INTERVENTIONS:
        raise ValueError(
            f"Unknown intervention: {kind!r}. Must be one of {VALID_INTERVENTIONS}"
        )
    if kind == "project_out":
        return intervention.project_out(row, direction)
    if kind == "inject":
        return intervention.inject(row, direction, scale=scale)
    return intervention.subtract(row, direction, scale=scale)


def run_vector_intervention_experiment(
    config: ExperimentConfig,
    *,
    interventions: Sequence[str] = ("project_out", "inject", "subtract"),
    inject_subtract_scale: float = 1.0,
    distribution_metric_name: str = "kl",
    confusion_top_k: int = 2,
    batch_loader: Optional[BatchLoader] = None,
) -> Dict[str, Any]:
    """Run inject/subtract/project_out interventions and compute loss + token distribution metric.

    Results keys are ``{variant}_{intervention}`` e.g. ``svd_inject``, ``orthogonal_subtract``.
    ``confusion_top_k`` controls the size of the saved confusion matrix (default 2, Mixtral's expert top-k).
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    load_kwargs: dict = {"low_cpu_mem_usage": False}
    if config.load_in_8bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(config.model_id, **load_kwargs)
    if torch.cuda.is_available():
        model = model.to("cuda")

    if batch_loader is not None:
        data_loader = batch_loader
    else:
        data_loader = _make_batch_loader(config, tokenizer)
    batches = data_loader.get_batches()

    # Materialize lazy (meta) parameters: run one forward so gate weights are materialized
    with torch.no_grad():
        model.eval()
        device = next(model.parameters()).device
        _ = model(batches[0].to(device))

    expert_vectors = ExpertVectors(
        config.svd_dir,
        config.layer_idx,
        config.num_experts,
        config.model_tag,
        top_k=config.top_k[0] if config.top_k else 1,
    )
    expert_vectors.load()
    if len(expert_vectors) == 0:
        raise RuntimeError("No SVD vectors loaded. Check svd_dir and file names.")

    evaluator = LossEvaluator(model, pad_token_id=tokenizer.pad_token_id)
    results: Dict[str, Any] = {
        "config": {
            **vars(config),
            "interventions": list(interventions),
            "inject_subtract_scale": inject_subtract_scale,
            "distribution_metric": distribution_metric_name,
            "confusion_top_k": confusion_top_k,
        },
        "results": {},
    }

    intervention = VectorIntervention()
    with RouterManager(model, config.layer_idx) as router:
        logger.info("Evaluating baseline...")
        base_loss = evaluator.evaluate(batches)
        results["baseline_loss"] = base_loss
        logger.info("Baseline loss: %.4f", base_loss)

        baseline_logits = evaluator.get_logits(batches)
        logger.info("Baseline logits shape: %s", tuple(baseline_logits.shape))
        dist_comparator = TokenDistributionComparator(
            metric=distribution_metric_name
        )

        for variant in config.variations:
            for intervent_kind in interventions:
                key = f"{variant}_{intervent_kind}"
                logger.info("--- %s ---", key.replace("_", " ").title())
                modified = router.original_weights.clone()

                for exp_idx, v_svd in expert_vectors.items():
                    direction = _get_direction(
                        variant, v_svd, exp_idx, intervention, config.seed
                    )
                    row = modified[exp_idx]
                    modified[exp_idx] = _apply_intervention(
                        intervention, intervent_kind, row, direction, inject_subtract_scale
                    )

                router.apply_weights(modified)
                loss = evaluator.evaluate(batches)
                delta = loss - base_loss
                modified_logits = evaluator.get_logits(batches)
                cm = dist_comparator.compare(baseline_logits, modified_logits)
                conf_matrix, conf_token_ids = confusion_matrix_top_k(
                    baseline_logits, modified_logits, top_k=confusion_top_k
                )
                results["results"][key] = {
                    "loss": loss,
                    "delta": delta,
                    "distribution_metric": cm,
                    "distribution_metric_name": distribution_metric_name,
                    "confusion_matrix": conf_matrix.tolist(),
                    "confusion_token_ids": conf_token_ids,
                }
                logger.info(
                    "Result [%s]: loss=%.4f, delta=%+.4f, %s=%.4f",
                    key, loss, delta, distribution_metric_name, cm,
                )

    out_path = Path(config.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving results to %s", out_path)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results
