"""Compute SVD of each expert's w1 or w3 and cache the full V basis for project-out.

For each expert, w1/w3 have shape [d_ff, hidden_dim].  SVD gives V of shape
(hidden_dim, hidden_dim) with columns = right singular vectors.  We save this
matrix so run_project_out can use the first top_k columns.

Use --weight w1 or --weight w3.  For w3, output goes under a "w3" subdir so you
can run project_out with --svd_dir pointing at that dir (e.g. svd_cache/float/w3).

Works for float or quantized models.  Output is always float32.
Files: {cache_dir}/{float|8bit}[/{w3}]/{model_tag}_layer{L}_expert{i}.pkl
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any

import torch

from .core import ExperimentConfig
from .core.inference import ModelLoader
from .core.router_manager import _dequantize_int8

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_QUANT_SUBDIR = {"8bit": "8bit", "4bit": "4bit", None: "float"}


def _get_model_device(model: torch.nn.Module) -> torch.device:
    """Get the first non-meta device from model parameters."""
    for param in model.parameters():
        if not param.is_meta:
            return param.device
    logger.warning("All parameters on meta device, defaulting to CPU")
    return torch.device("cpu")


def _weight_float(linear: Any, quantized: bool) -> torch.Tensor:
    """Extract weight as float (dequantize int8 if needed)."""
    w = getattr(linear.weight, "data", linear.weight)
    if quantized and w.dtype == torch.int8:
        return _dequantize_int8(linear.weight)
    return w.float()


def run_svd_from_expert(
    model_id: str,
    model_tag: str,
    cache_dir: str,
    layer_indices: list[int],
    num_experts: int = 8,
    quantization: str | None = None,
    weight: str = "w1",
    force: bool = False,
) -> str:
    """Compute SVD of each expert w1 or w3 → save V (columns = vectors) per expert.

    Skips (layer, expert) if the output .pkl already exists unless force=True.
    weight: "w1" or "w3" (expert linear to run SVD on).
    Returns the output directory path (use as --svd_dir in run_project_out).
    """
    if weight not in ("w1", "w3"):
        raise ValueError("weight must be 'w1' or 'w3'")
    quantized = quantization in ("8bit", "4bit")
    out_path = Path(cache_dir).resolve() / _QUANT_SUBDIR.get(quantization, "float")
    if weight == "w3":
        out_path = out_path / "w3"
    out_path.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig(
        svd_dir="",
        layer_idx=layer_indices[0],
        model_id=model_id,
        model_tag=model_tag,
        num_experts=num_experts,
        quantization=quantization,
    )
    # Pass layer_indices as priority layers - ensures they're on GPU
    # max_gpu_layers=12 leaves ~45GB headroom for activations (avoids OOM)
    loader = ModelLoader(config, priority_layers=layer_indices, max_gpu_layers=12)
    logger.info("Loading model (%s)...", quantization or "float")
    loader.load_tokenizer()
    model = loader.load_model()

    # Materialize lazy params
    try:
        dev = _get_model_device(model)
        if dev.type == "cuda":
            with torch.no_grad():
                model.eval()
                model(torch.zeros(1, 4, dtype=torch.long, device=dev))
    except Exception as exc:
        logger.warning("Skip materialization: %s", exc)

    for layer_idx in layer_indices:
        try:
            n_saved = 0
            for expert_idx in range(num_experts):
                path = out_path / f"{model_tag}_layer{layer_idx}_expert{expert_idx}.pkl"
                if not force and path.exists():
                    logger.debug("Layer %d expert %d: cached, skipping", layer_idx, expert_idx)
                    continue
                linear = getattr(
                    model.model.layers[layer_idx].block_sparse_moe.experts[expert_idx],
                    weight,
                )
                w = _weight_float(linear, quantized).cpu().float()
                _, _, Vh = torch.linalg.svd(w, full_matrices=False)
                V = Vh.T  # (dim, dim), columns = right singular vectors
                with open(path, "wb") as f:
                    pickle.dump(V.numpy().astype("float32"), f)
                n_saved += 1
            if n_saved > 0:
                logger.info("Layer %d: saved %d expert V matrices to %s", layer_idx, n_saved, out_path)
            else:
                logger.info("Layer %d: all cached, skipped", layer_idx)
        except (NotImplementedError, RuntimeError, OSError) as e:
            logger.warning("Layer %d failed (%s: %s), skipping layer", layer_idx, type(e).__name__, e)
            continue

    svd_dir = str(out_path)
    logger.info("Use --svd_dir %s with run_project_out", svd_dir)
    return svd_dir


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute SVD of each expert w1 or w3; save V for project-out on the router.",
    )
    p.add_argument("--model-id", default="mistralai/Mixtral-8x7B-v0.1")
    p.add_argument("--model-tag", default="mistralai_Mixtral_8x7B_v0.1")
    p.add_argument(
        "--cache-dir",
        default="svd_cache",
        help="Base cache dir (subdirs: float/, 8bit/, float/w3/)",
    )
    p.add_argument(
        "--layer_idx", type=int, nargs="+", required=True, help="Layer(s) to export"
    )
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--quantization", default=None, choices=("8bit", "4bit"))
    p.add_argument(
        "--weight",
        default="w1",
        choices=("w1", "w3"),
        help="Expert weight to run SVD on (w1 or w3)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite even if output .pkl already exists",
    )
    args = p.parse_args()

    run_svd_from_expert(
        model_id=args.model_id,
        model_tag=args.model_tag,
        cache_dir=args.cache_dir,
        layer_indices=args.layer_idx,
        num_experts=args.num_experts,
        quantization=args.quantization,
        weight=args.weight,
        force=args.force,
    )


if __name__ == "__main__":
    main()
