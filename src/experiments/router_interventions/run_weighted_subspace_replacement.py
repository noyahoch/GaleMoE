"""
Replace each router row with a weighted subspace of that expert's top-k SVD vectors,
then compute the routing migration matrix (baseline vs modified).

For loss-only comparison with other ablation variants (svd, random, zero, etc.),
use run_project_out with --variations including "replace" instead; results will
be in the same JSON with by_k and k_independent.

This script is for when you need the expert migration matrix (routing before/after)
for the replace intervention.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from .core import (
    ExperimentConfig,
    LossEvaluator,
    ModelLoader,
    RouterManager,
    ExpertVectors,
    VectorIntervention,
    register_routing_capture,
    routing_captures_to_selections,
    expert_confusion_matrix,
    Timer,
    timer,
)
from .core.data import TextListBatchLoader, WikitextBatchLoader

logger = logging.getLogger(__name__)


def _make_batches(config: ExperimentConfig, tokenizer):
    if config.dataset == "wikitext":
        return WikitextBatchLoader(
            tokenizer, config.num_samples, config.seq_len, config.batch_size
        ).get_batches()
    if config.dataset == "text":
        if not config.text_file:
            raise ValueError("dataset='text' requires text_file")
        texts = [
            t for t in Path(config.text_file).read_text().splitlines() if t.strip()
        ]
        return TextListBatchLoader(
            tokenizer, texts, config.seq_len, config.batch_size
        ).get_batches()
    raise ValueError(f"Unknown dataset: {config.dataset!r}")


def run_experiment(
    config: ExperimentConfig,
    scale: float = 1.0,
    weight_decay: float = 1.0,
    k_values: Sequence[int] | None = None,
) -> Dict[str, Any]:
    """
    For each k in k_values: replace each expert's router row with
    scale * unit(weighted_sum of that expert's top-k SVD vectors).
    Capture baseline and modified routing, compute migration matrix.
    """
    k_values = k_values or list(config.top_k) if config.top_k else [1]
    if not isinstance(k_values, list):
        k_values = [k_values]
    clock = Timer()
    clock.start("total")

    loader = ModelLoader(config, priority_layers=[config.layer_idx], max_gpu_layers=15)
    with timer("Loading model"):
        tokenizer = loader.load_tokenizer()
        model = loader.load_model()

    with timer("Loading data"):
        batches = _make_batches(config, tokenizer)
        logger.info(
            "Batches: %d x %d = %d samples",
            len(batches),
            batches[0].shape[0],
            len(batches) * batches[0].shape[0],
        )

    with timer("Loading SVD (all experts)"):
        expert_vecs = ExpertVectors(
            config.svd_dir,
            config.layer_idx,
            config.num_experts,
            config.model_tag,
            top_k=max(k_values),
        )
        expert_vecs.load()
    if len(expert_vecs) == 0:
        raise RuntimeError("No SVD vectors loaded. Check svd_dir and file names.")
    if len(expert_vecs) < config.num_experts:
        logger.warning(
            "Loaded %d experts, expected %d; missing experts will be skipped",
            len(expert_vecs),
            config.num_experts,
        )

    evaluator = LossEvaluator(model)

    results = {
        "config": {
            **{k: v for k, v in vars(config).items() if k != "top_k"},
            "scale": scale,
            "weight_decay": weight_decay,
            "k_values": k_values,
        },
        "baseline": {},
        "by_k": {},
    }

    with RouterManager(model, config.layer_idx) as router:
        # Baseline: capture loss, logits, routing in one forward
        with timer("Baseline"):
            logger.info("Evaluating baseline...")
            routing_handle, baseline_routing_captured = register_routing_capture(
                model, config.layer_idx
            )
            base_loss, baseline_logits = evaluator.evaluate_and_get_logits(batches)
            routing_handle.remove()
            baseline_routing = routing_captures_to_selections(baseline_routing_captured)
            results["baseline"] = {
                "loss": base_loss,
                "routing_shape": list(baseline_routing.shape),
            }
            logger.info("Baseline loss: %.4f, routing tokens: %d", base_loss, baseline_routing.numel())

        # Per-k: replace each row with weighted subspace, then capture routing
        for top_k in k_values:
            clock.start(f"k_{top_k}")
            logger.info("--- top_k=%d ---", top_k)
            modified = router.original_weights.clone()
            device = modified.device
            dtype = modified.dtype

            for exp_idx in range(config.num_experts):
                if exp_idx not in expert_vecs:
                    continue
                v_full = expert_vecs[exp_idx]
                if v_full.ndim == 1:
                    v_k = v_full.unsqueeze(0)
                else:
                    v_k = v_full[:top_k]
                if v_k.shape[0] == 0:
                    continue
                v_k = v_k.to(device=device, dtype=dtype)
                new_row = VectorIntervention.weighted_subspace_row(
                    v_k, scale=scale, weight_decay=weight_decay
                )
                modified[exp_idx] = new_row.to(device=device, dtype=dtype)

            router.apply_weights(modified)
            routing_handle, modified_routing_captured = register_routing_capture(
                model, config.layer_idx
            )
            loss, _ = evaluator.evaluate_and_get_logits(batches)
            routing_handle.remove()
            modified_routing = routing_captures_to_selections(modified_routing_captured)

            expert_conf_matrix, expert_stats = expert_confusion_matrix(
                baseline_routing, modified_routing, config.num_experts
            )

            results["by_k"][top_k] = {
                "loss": loss,
                "delta": loss - base_loss,
                "expert_confusion_matrix": expert_conf_matrix.tolist(),
                "expert_routing_stats": expert_stats,
            }
            logger.info(
                "k=%d: loss=%.4f, delta=%+.4f, migration_rate=%.1f%%",
                top_k,
                loss,
                loss - base_loss,
                expert_stats["migration_rate"] * 100,
            )
            router.restore()
            clock.stop(f"k_{top_k}")

    clock.stop("total")
    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    p = argparse.ArgumentParser(
        description="Replace each router row with weighted subspace of that expert's top-k SVD; report migration matrix.",
    )
    p.add_argument("--svd-dir", required=True, help="Directory with expert SVD pickle files")
    p.add_argument("--layer-idx", type=int, required=True, help="MoE layer index")
    p.add_argument("--output-dir", default="results", help="Output directory for JSON")
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--model-id", default="mistralai/Mixtral-8x7B-v0.1")
    p.add_argument("--model-tag", default="mistralai_Mixtral_8x7B_v0.1")
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--dataset", default="wikitext", choices=("wikitext", "text"))
    p.add_argument("--text-file", default=None)
    p.add_argument("--top-k", default="1,2,4,8", help="Comma-separated k values")
    p.add_argument("--scale", type=float, default=1.0, help="Scale for weighted subspace row")
    p.add_argument("--weight-decay", type=float, default=1.0, help="Weight decay for weighted sum: w_i ∝ 1/(i+1)^decay")
    p.add_argument("--quantization", default=None, choices=("8bit", "4bit"))
    args = p.parse_args()

    k_values = [int(x.strip()) for x in args.top_k.split(",") if x.strip()] or [1]
    if args.dataset == "text" and not args.text_file:
        p.error("--dataset=text requires --text-file")

    config = ExperimentConfig(
        svd_dir=args.svd_dir,
        layer_idx=args.layer_idx,
        output_dir=args.output_dir,
        num_experts=args.num_experts,
        model_id=args.model_id,
        model_tag=args.model_tag,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        dataset=args.dataset,
        text_file=args.text_file,
        top_k=k_values,
        quantization=args.quantization,
    )

    results = run_experiment(
        config,
        scale=args.scale,
        weight_decay=args.weight_decay,
        k_values=k_values,
    )

    out_name = f"weighted_subspace_L{config.layer_idx}_k{'-'.join(map(str, k_values))}_S{args.scale}_wd{args.weight_decay}_{config.dataset}.json"
    if config.quantization:
        out_name = out_name.replace(".json", f"_q{config.quantization}.json")
    out_path = Path(config.output_dir) / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving results to %s", out_path)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
