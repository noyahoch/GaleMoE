"""Runner for project-out experiment: remove direction from router rows, compare loss."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..core import (
    ExperimentConfig,
    BatchLoader,
    LossEvaluator,
    RouterManager,
    ExpertVectors,
    VectorIntervention,
)
from ..core.data import TextListBatchLoader, WikitextBatchLoader, WikitextTitlesBatchLoader
from ..core.memory import log_gpu_memory

logger = logging.getLogger(__name__)

VALID_VARIATIONS = ("svd", "orthogonal", "random", "zero", "shuffle")
PROJECT_OUT_VARIATIONS = ("svd", "orthogonal", "random")  # require expert vectors


class ProjectOutRunner:
    """Project out a direction from each expert's router row; compare loss (baseline + variants including zero/shuffle)."""

    def __init__(
        self,
        config: ExperimentConfig,
        batch_loader: Optional[BatchLoader] = None,
    ) -> None:
        self.config = config
        self._batch_loader = batch_loader

    def _make_batch_loader(self, tokenizer: "AutoTokenizer") -> BatchLoader:
        if self.config.dataset == "wikitext":
            return WikitextBatchLoader(
                tokenizer,
                num_samples=self.config.num_samples,
                seq_len=self.config.seq_len,
                batch_size=self.config.batch_size,
            )
        if self.config.dataset == "wikitext_titles":
            return WikitextTitlesBatchLoader(
                tokenizer,
                num_samples=self.config.num_samples,
                seq_len=self.config.seq_len,
                batch_size=self.config.batch_size,
            )
        if self.config.dataset == "text":
            if not self.config.text_file:
                raise ValueError("config.dataset is 'text' but config.text_file is not set")
            path = Path(self.config.text_file)
            if not path.exists():
                raise FileNotFoundError(f"Text file not found: {path}")
            texts = path.read_text().strip().splitlines()
            texts = [t.strip() for t in texts if t.strip()]
            return TextListBatchLoader(
                tokenizer,
                texts=texts,
                seq_len=self.config.seq_len,
                batch_size=self.config.batch_size,
            )
        raise ValueError(f"Unknown dataset: {self.config.dataset}")

    def _get_remove_vector(
        self,
        variant: str,
        v_svd: torch.Tensor,
        exp_idx: int,
        intervention: VectorIntervention,
    ) -> torch.Tensor:
        """Return the direction(s) to project out for this variant. Raises on unknown variant.
        
        Returns:
            - For variant="svd": returns v_svd as-is (may be [dim] or [k, dim])
            - For variant="orthogonal" or "random": returns [dim] tensor (single vector)
        """
        if variant not in PROJECT_OUT_VARIATIONS:
            raise ValueError(
                f"Unknown project-out variant: {variant!r}. Must be one of {PROJECT_OUT_VARIATIONS}"
            )
        seed = self.config.seed + exp_idx
        if variant == "svd":
            return v_svd
        # For orthogonal and random variants, we need a single vector
        # Get the first vector if v_svd is [k, dim], otherwise use as-is
        if v_svd.ndim == 2:
            v_first = v_svd[0]  # [dim]
        else:
            v_first = v_svd  # [dim]
        if variant == "orthogonal":
            return intervention.make_orthogonal(v_first, seed=seed)
        # variant == "random"
        return intervention.make_random(v_first.shape[0], seed=seed)

    def _k_values_to_run(self) -> list[int]:
        """Return list of top_k values to run."""
        return list(self.config.top_k)

    def run(self) -> Dict[str, Any]:
        """Load model and data, run project-out variants, save results to config.output_file.
        Runs over each k in config.top_k. If len(top_k) > 1, stores results in results['by_k'][k];
        otherwise stores in results['results'] (backward compatible).
        """
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        log_gpu_memory("Before model load")

        load_kwargs: dict = {"low_cpu_mem_usage": False}
        if self.config.load_in_8bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **load_kwargs)
        if torch.cuda.is_available():
            model = model.to("cuda")
        log_gpu_memory("After model load")

        if self._batch_loader is not None:
            data_loader = self._batch_loader
        else:
            data_loader = self._make_batch_loader(tokenizer)
        batches = data_loader.get_batches()

        # Materialize lazy (meta) parameters: run one forward so gate weights are materialized
        with torch.no_grad():
            model.eval()
            first_batch = batches[0]
            device = next(model.parameters()).device
            _ = model(first_batch.to(device))
        log_gpu_memory("After first forward (batches loaded)")

        evaluator = LossEvaluator(model, pad_token_id=tokenizer.pad_token_id)
        k_values = self._k_values_to_run()
        multi_k = len(k_values) > 1

        results: Dict[str, Any] = {
            "config": vars(self.config),
            "results": {},
        }
        if multi_k:
            results["by_k"] = {}

        # Validate variations
        for v in self.config.variations:
            if v not in VALID_VARIATIONS:
                raise ValueError(
                    f"Unknown variation: {v!r}. Must be one of {VALID_VARIATIONS}"
                )

        intervention = VectorIntervention()
        with RouterManager(model, self.config.layer_idx) as router:
            logger.info("Evaluating baseline...")
            base_loss = evaluator.evaluate(batches)
            results["baseline_loss"] = base_loss
            logger.info("Baseline loss: %.4f", base_loss)

            orig = router.original_weights
            num_experts = orig.shape[0]

            # Precompute zero and shuffle once (they don't depend on k)
            baseline_interventions: Dict[str, Dict[str, Any]] = {}
            if "zero" in self.config.variations:
                router.apply_weights(torch.zeros_like(orig))
                loss_zero = evaluator.evaluate(batches)
                baseline_interventions["zero"] = {"loss": loss_zero, "delta": loss_zero - base_loss}
                logger.info("Result [zero]: loss=%.4f, delta=%+.4f", loss_zero, baseline_interventions["zero"]["delta"])
                router.restore()
            if "shuffle" in self.config.variations:
                g = torch.Generator().manual_seed(self.config.seed)
                perm = torch.randperm(num_experts, generator=g)
                router.apply_weights(orig[perm].clone())
                loss_shuffle = evaluator.evaluate(batches)
                baseline_interventions["shuffle"] = {"loss": loss_shuffle, "delta": loss_shuffle - base_loss}
                logger.info("Result [shuffle]: loss=%.4f, delta=%+.4f", loss_shuffle, baseline_interventions["shuffle"]["delta"])
                router.restore()

            # Load expert vectors once for max(k) and reuse for each k (avoids repeated disk I/O)
            max_k = max(k_values)
            logger.info("Loading expert vectors once (top_k=%d) for all k values", max_k)
            expert_vectors = ExpertVectors(
                self.config.svd_dir,
                self.config.layer_idx,
                self.config.num_experts,
                self.config.model_tag,
                top_k=max_k,
            )
            expert_vectors.load()
            if len(expert_vectors) == 0:
                raise RuntimeError("No SVD vectors loaded. Check svd_dir and file names.")

            for top_k in k_values:
                if multi_k:
                    logger.info("=== top_k=%d ===", top_k)
                    results["by_k"][top_k] = {}

                for variant in self.config.variations:
                    if variant in baseline_interventions:
                        entry = baseline_interventions[variant]
                        logger.info("--- Variant: %s (baseline intervention, k=%d) ---", variant.upper(), top_k)
                    else:
                        logger.info("--- Variant: %s (k=%d) ---", variant.upper(), top_k)
                        modified = router.original_weights.clone()
                        for exp_idx, v_svd_full in expert_vectors.items():
                            # Use first top_k vectors (slice of pre-loaded max_k vectors)
                            if v_svd_full.ndim == 2:
                                v_svd = v_svd_full[:top_k]  # [k, dim]
                                if v_svd.shape[0] == 1:
                                    v_svd = v_svd[0]  # [dim] for project_out compat
                            else:
                                v_svd = v_svd_full
                            v_remove = self._get_remove_vector(
                                variant, v_svd, exp_idx, intervention
                            )
                            row = modified[exp_idx]
                            modified[exp_idx] = intervention.project_out(row, v_remove)
                        router.apply_weights(modified)
                        loss = evaluator.evaluate(batches)
                        entry = {"loss": loss, "delta": loss - base_loss}
                        logger.info(
                            "Result [%s] k=%d: loss=%.4f, delta=%+.4f",
                            variant, top_k, entry["loss"], entry["delta"],
                        )

                    if multi_k:
                        results["by_k"][top_k][variant] = entry
                    else:
                        results["results"][variant] = entry

        self._save_results(results)
        return results

    def _save_results(self, results: Dict[str, Any]) -> None:
        out_path = Path(self.config.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving results to %s", out_path)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)


def run_project_out_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Run project-out experiment: build runner and run."""
    return ProjectOutRunner(cfg).run()
