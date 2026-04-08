"""Project out expert SVD directions from router rows and compare loss.

Usage::

    python -m src.experiments.router_interventions.run_project_out \\
        --svd-dir svd_cache \\
        --layer-idx 0 1 15 31 \\
        --output-dir results \\
        --top-k 1,2,4,8,16,32,64

Results format:
    - "baseline_loss": float
    - "k_independent": dict of {variant: {"loss": ..., "delta": ...}}
      Contains zero, shuffle, orthogonal (same for all k)
    - "by_k": dict of {k: {variant: {"loss": ..., "delta": ...}}}
      Contains svd and random per k (both are k-dependent)
      
Variants:
    - svd: Project out top-k SVD from each expert row
    - orthogonal: Random direction orthogonal to v1 (k-independent)
    - random: Project out k random orthonormal vectors (k-dependent)
    - zero: Zero out entire router (k-independent)
    - shuffle: Permute expert rows (k-independent)
    - replace: Replace each router row with scale * unit(weighted_sum of that expert's top-k SVD) (k-dependent)
      
Old format (still supported for backward compatibility):
    All variants were duplicated for each k value in "by_k"
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Sequence

import torch

from .core import (
    ExperimentConfig,
    LossEvaluator,
    ModelLoader,
    RouterManager,
    ExpertVectors,
    VectorIntervention,
    Timer,
    timer,
)
from .core.data import WikitextBatchLoader, WikiTitlesBatchLoader, TextListBatchLoader
from .core.memory import log_gpu_memory

logger = logging.getLogger(__name__)


def _orthogonalize_rows_gs(R: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # R: [E, d]
    Rf = R.float()
    Q = []
    for i in range(Rf.shape[0]):
        v = Rf[i].clone()
        for q in Q:
            v = v - (v @ q) * q
        n = v.norm()
        if n < eps:
            # fallback: keep original (or random) if it collapses
            v = Rf[i].clone()
            n = v.norm() + eps
        Q.append(v / n)
    return torch.stack(Q, dim=0).to(dtype=R.dtype, device=R.device)


# Ablation variants:
# - svd: Project out top-k SVD right singular vectors
# - orthogonal: Project out random direction orthogonal to v1
# - random: Project out k random orthonormal vectors in full R^dim (QR decomposition)
# - zero: Zero out entire router layer
# - shuffle: Random permutation of expert rows
# - replace: Replace each router row with scale * unit(weighted_sum of that expert's top-k SVD)
VARIATIONS = ("svd", "orthogonal", "random", "zero", "shuffle", "replace")


def _get_model_device(model: torch.nn.Module) -> torch.device:
    """Get the first non-meta device from model parameters."""
    for param in model.parameters():
        if not param.is_meta:
            return param.device
    # Fallback to CPU if all parameters are on meta (shouldn't happen)
    logger.warning("All parameters on meta device, defaulting to CPU")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanity_check(router: RouterManager, vectors: ExpertVectors) -> None:
    """Log diagnostics: project-out of self ~ 0, loaded SVD alignment."""
    row = router.original_weights[0].float()
    norm = row.norm().item() + 1e-12
    po = VectorIntervention.project_out

    ratio = po(row, row / norm).norm().item() / norm
    logger.info("Sanity: project-out self -> ratio=%.6f (expect ~0)", ratio)
    if ratio > 0.01:
        logger.warning("Project-out self ratio %.4f; check logic", ratio)

    v = vectors.get_single(0).to(device=row.device, dtype=row.dtype)
    ratio_svd = po(row, v).norm().item() / norm
    logger.info("Sanity: loaded SVD -> ratio=%.6f (expect <1 if aligned)", ratio_svd)
    if ratio_svd > 0.99:
        logger.warning(
            "SVD almost orthogonal to gate (ratio=%.4f). Regenerate SVD cache.",
            ratio_svd,
        )


def _make_batches(config: ExperimentConfig, tokenizer):
    if config.dataset == "wikitext":
        return WikitextBatchLoader(
            tokenizer, config.num_samples, config.seq_len, config.batch_size
        ).get_batches()
    if config.dataset == "wiki_titles":
        return WikiTitlesBatchLoader(
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
    raise ValueError(f"Unknown dataset: {config.dataset}")


def _vectors_for_k(v_full: torch.Tensor, k: int) -> torch.Tensor:
    """First *k* vectors: [k, dim] or [dim] when k=1."""
    if v_full.ndim == 1:
        return v_full
    v = v_full[:k]
    return v[0] if v.shape[0] == 1 else v


def _remove_vector(variant: str, v_svd: torch.Tensor, seed: int) -> torch.Tensor:
    """Direction to project out for a given variant."""
    if variant == "svd":
        return v_svd
    v1 = v_svd[0] if v_svd.ndim == 2 else v_svd
    if variant == "orthogonal":
        return VectorIntervention.make_orthogonal(v1, seed)
    if variant == "random":
        # Random k orthonormal vectors in full R^dim (not constrained to SVD span)
        # Matches main2.py implementation
        k = v_svd.shape[0] if v_svd.ndim == 2 else 1
        dim = v_svd.shape[-1]
        v_rand = VectorIntervention.make_random_subspace(
            k, dim, seed, device=v_svd.device
        )
        return v_rand[0] if k == 1 else v_rand
    raise ValueError(f"Unknown variant: {variant}")


def _svd_dir_tag(svd_dir: str) -> str:
    """Short tag for svd_dir so w1 vs w3 (and paths) don't share the same cache file."""
    path = svd_dir.rstrip("/")
    if "/w3" in path or path.endswith("w3"):
        return "w3"
    if "/w1" in path or path.endswith("w1"):
        return "w1"
    # Use last path component (e.g. "float") so different cache dirs don't collide
    return Path(path).name or "svd"


def _result_path(config: ExperimentConfig, layer_idx: int) -> Path:
    """Build unique filename capturing all config params (including svd source so w1/w3 don't share cache)."""
    k_part = "-".join(str(k) for k in config.top_k)
    v_part = "-".join(config.variations)
    q = config.quantization or "none"
    svd_tag = _svd_dir_tag(config.svd_dir)
    # Format scale: if integer (1.0, 10.0), show without decimals; otherwise show full value
    scale_str = (
        f"{int(config.scale)}"
        if config.scale == int(config.scale)
        else f"{config.scale}"
    )
    name = (
        f"project_out_L{layer_idx}_k{k_part}_{config.dataset}_"
        f"n{config.num_samples}_s{config.seq_len}_b{config.batch_size}_"
        f"v{v_part}_q{q}_svd{svd_tag}_scale{scale_str}_seed{config.seed}.json"
    )
    return Path(config.output_dir) / name


def _save_results(
    config: ExperimentConfig, layer_idx: int, results: Dict[str, Any]
) -> None:
    path = _result_path(config, layer_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving results to %s", path)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment(
    config: ExperimentConfig, layer_indices: Sequence[int]
) -> Dict[int, Dict[str, Any]]:
    """Run project-out experiment on one or more layers."""
    clock = Timer()
    clock.start("total")

    log_gpu_memory("Before model load")
    # Pass layer_indices as priority layers - ensures they're on GPU
    loader = ModelLoader(config, priority_layers=list(layer_indices), max_gpu_layers=15)
    with timer("Loading model"):
        tokenizer = loader.load_tokenizer()
        model = loader.load_model()
    log_gpu_memory("After model load")

    with timer("Loading data"):
        batches = _make_batches(config, tokenizer)
        logger.info(
            "Batches: %d x %d = %d samples",
            len(batches),
            batches[0].shape[0],
            len(batches) * batches[0].shape[0],
        )

    with timer("First forward"):
        with torch.no_grad():
            model.eval()
            device = _get_model_device(model)
            _ = model(batches[0].to(device))

    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(
        tokenizer, "eos_token_id", None
    )
    evaluator = LossEvaluator(model, pad_token_id=pad_id)

    all_layer_results = {}
    for layer_idx in layer_indices:
        # Check if results already exist (resume capability)
        result_path = _result_path(config, layer_idx)
        if result_path.exists():
            logger.info(
                "Layer %d: SKIPPING (result file exists: %s)",
                layer_idx,
                result_path.name,
            )
            with open(result_path) as f:
                all_layer_results[layer_idx] = json.load(f)
            continue

        logger.info("\n" + "=" * 70)
        logger.info("LAYER %d", layer_idx)
        logger.info("=" * 70)
        clock.start(f"layer_{layer_idx}")
        results = _run_single_layer(config, layer_idx, model, batches, evaluator)
        elapsed = clock.stop(f"layer_{layer_idx}")
        logger.info("Layer %d completed in %.1f minutes", layer_idx, elapsed / 60)
        all_layer_results[layer_idx] = results
        _save_results(config, layer_idx, results)

    clock.stop("total")
    logger.info("\n" + "=" * 50)
    logger.info("Completed %d layer(s)", len(layer_indices))
    clock.report()
    return all_layer_results


def _run_single_layer(
    config: ExperimentConfig,
    layer_idx: int,
    model,
    batches,
    evaluator: LossEvaluator,
) -> Dict[str, Any]:
    """Run experiments for a single layer."""
    k_values = list(config.top_k)
    intervention = VectorIntervention

    results: Dict[str, Any] = {
        "config": {**vars(config), "layer_idx": layer_idx},
        "by_k": {},
        "k_independent": {},  # zero, shuffle, orthogonal (random is per-k in by_k)
    }

    with RouterManager(model, layer_idx) as router:
        # Baseline
        with timer(f"Baseline (L{layer_idx})"):
            base_loss = evaluator.evaluate(batches)
            results["baseline_loss"] = base_loss
            logger.info(
                "Baseline loss=%.4f (ppl=%.1f)", base_loss, math.exp(min(base_loss, 20))
            )
            logger.info("Projection scale: %.1f", config.scale)

        orig = router.original_weights

        # k-independent interventions
        fixed: Dict[str, Dict[str, Any]] = {}

        if "zero" in config.variations:
            with timer("Zero"):
                router.apply_weights(torch.zeros_like(orig))
                loss = evaluator.evaluate(batches)
                fixed["zero"] = {"loss": loss, "delta": loss - base_loss}
                logger.info("Zero -> loss=%.4f, delta=%+.4f", loss, loss - base_loss)
                router.restore()

        if "shuffle" in config.variations:
            with timer("Shuffle"):
                perm = torch.randperm(
                    orig.shape[0], generator=torch.Generator().manual_seed(config.seed)
                )
                router.apply_weights(orig[perm])
                loss = evaluator.evaluate(batches)
                fixed["shuffle"] = {"loss": loss, "delta": loss - base_loss}
                logger.info("Shuffle -> loss=%.4f, delta=%+.4f", loss, loss - base_loss)
                router.restore()

        # Load expert vectors
        with timer(f"Loading SVD vectors (L{layer_idx})"):
            expert_vecs = ExpertVectors(
                config.svd_dir,
                layer_idx,
                config.num_experts,
                config.model_tag,
                top_k=max(k_values),
            )
            expert_vecs.load()
        _sanity_check(router, expert_vecs)

        first_vec = {i: _vectors_for_k(v, 1) for i, v in expert_vecs.items()}

        # Orthogonal only: random unit vector orthogonal to v1 (does not depend on k)
        for variant in ("orthogonal",):
            if variant not in config.variations:
                continue
            with timer(f"{variant.title()} (once)"):
                logger.info("Computing %s (k-independent)", variant.upper())
                modified = router.original_weights.clone()
                for i in expert_vecs.keys():
                    v_rm = _remove_vector(variant, first_vec[i], config.seed + i)
                    modified[i] = intervention.project_out(
                        modified[i], v_rm, scale=config.scale
                    )
                router.apply_weights(modified)
                loss = evaluator.evaluate(batches)
                fixed[variant] = {"loss": loss, "delta": loss - base_loss}
                logger.info(
                    "%s -> loss=%.4f, delta=%+.4f",
                    variant.upper(),
                    loss,
                    loss - base_loss,
                )
                router.restore()

        # Store k-independent results (only once, not for each k)
        # (random is k-dependent: computed per top_k in the per-k loop below)
        results["k_independent"] = fixed

        # Per-k loop (only for k-dependent variants: svd, random, replace)
        for top_k in k_values:
            logger.info("\n--- top_k=%d ---", top_k)
            k_results: Dict[str, Any] = {}

            for variant in config.variations:
                # Skip k-independent variants - they're already in results["k_independent"]
                if variant in fixed:
                    continue

                with timer(f"{variant} k={top_k}"):
                    logger.info("Computing %s (k=%d)", variant.upper(), top_k)
                    modified = router.original_weights.clone()

                    if variant == "replace":
                        # Replace each row with weighted subspace of that expert's top-k SVD
                        orig_norms = modified.norm(dim=1)
                        for i, v_full in expert_vecs.items():
                            v_k = _vectors_for_k(v_full, top_k)
                            if v_k.ndim == 1:
                                v_k = v_k.unsqueeze(0)
                            v_k = v_k.to(device=modified.device, dtype=modified.dtype)
                            orig_norm = modified[i].norm()
                            new_row = intervention.weighted_subspace_row(
                                v_k,
                                scale=config.scale,
                                weight_decay=config.replace_weight_decay,
                            )
                            new_norm = new_row.norm()
                            if new_norm > 0:
                                new_row = new_row * (orig_norm / new_norm)
                            modified[i] = new_row
                        modified = _orthogonalize_rows_gs(modified)
                        new_norms = modified.norm(dim=1) + 1e-12
                        modified = modified * (orig_norms / new_norms).unsqueeze(1)

                    else:
                        # Project out: svd, random
                        for i, v_full in expert_vecs.items():
                            v = _vectors_for_k(v_full, top_k)
                            v_rm = _remove_vector(variant, v, config.seed + i)
                            modified[i] = intervention.project_out(
                                modified[i], v_rm, scale=config.scale
                            )

                    router.apply_weights(modified)
                    loss = evaluator.evaluate(batches)
                    k_results[variant] = {"loss": loss, "delta": loss - base_loss}
                    logger.info(
                        "%s -> loss=%.4f, delta=%+.4f",
                        variant.upper(),
                        loss,
                        loss - base_loss,
                    )
                    router.restore()

            results["by_k"][top_k] = k_results

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    p = argparse.ArgumentParser(
        description="Project out SVD directions from router rows, compare loss."
    )
    p.add_argument(
        "--svd-dir", required=True, help="Directory with expert SVD pickle files"
    )
    p.add_argument(
        "--layer-idx",
        type=int,
        nargs="+",
        required=True,
        help="MoE layer index (or indices, e.g. 0 15 31 or 0 1 2 3 ... 31)",
    )
    p.add_argument(
        "--output-dir", default="results", help="Folder for result JSON files"
    )
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--model-id", default="mistralai/Mixtral-8x7B-v0.1")
    p.add_argument("--model-tag", default="mistralai_Mixtral_8x7B_v0.1")
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--variations",
        default="svd,orthogonal,random,zero,shuffle",
        help="Comma-separated: svd,orthogonal,random,zero,shuffle,replace",
    )
    p.add_argument(
        "--dataset", default="wiki_titles", choices=("wikitext", "wiki_titles", "text")
    )
    p.add_argument("--text-file", default=None, help="Text file for --dataset=text")
    p.add_argument(
        "--top-k",
        nargs="*",
        default=["1"],
        metavar="K [K ...]",
        help="Singular-vector count(s) to project out (e.g. 1 2 4 8 or 1,2,4,8,16,32,64)",
    )
    p.add_argument(
        "--quantization",
        default=None,
        choices=("8bit", "4bit"),
        help="Model quantization (omit for float)",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Projection scale multiplier (1.0=realistic, 10.0=amplified). Also used for replace variant.",
    )
    p.add_argument(
        "--replace-weight-decay",
        type=float,
        default=1.0,
        help="For 'replace' variant only: w_i ∝ 1/(i+1)^decay (default 1.0)",
    )

    args = p.parse_args()

    variations = [v.strip() for v in args.variations.split(",") if v.strip()]
    # Support both: --top-k 2 4 8 16 and --top-k 2,4,8,16
    if len(args.top_k) == 1 and "," in args.top_k[0]:
        top_k = [int(x) for x in args.top_k[0].split(",") if x.strip()]
    else:
        top_k = [int(x) for x in args.top_k if str(x).strip()]
    if not top_k:
        p.error("--top-k must contain at least one integer")
    if args.dataset == "text" and not args.text_file:
        p.error("--dataset=text requires --text-file")

    config = ExperimentConfig(
        svd_dir=args.svd_dir,
        layer_idx=args.layer_idx[0],  # Just for config; actual layers passed separately
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        model_id=args.model_id,
        model_tag=args.model_tag,
        num_experts=args.num_experts,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        variations=variations,
        dataset=args.dataset,
        text_file=args.text_file,
        top_k=top_k,
        quantization=args.quantization,
        scale=args.scale,
        replace_weight_decay=args.replace_weight_decay,
    )

    run_experiment(config, layer_indices=args.layer_idx)


if __name__ == "__main__":
    main()
