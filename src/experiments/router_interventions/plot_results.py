"""CLI: generate plots from project-out result JSONs.

Usage examples::

    # Delta vs k for layer 15
    python -m src.experiments.router_interventions.plot_results \\
        --results-dir results --mode delta-vs-k --layer 15

    # Delta vs layer for k=1
    python -m src.experiments.router_interventions.plot_results \\
        --results-dir results --mode delta-vs-layers --k 1

    # Both at once
    python -m src.experiments.router_interventions.plot_results \\
        --results-dir results --mode all --k 1 --layer 15
"""
from __future__ import annotations

import argparse
import logging

from .viz import load_results_dir, plot_delta_vs_k, plot_delta_vs_layers

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    p = argparse.ArgumentParser(description="Plot project-out experiment results.")
    p.add_argument("--results-dir", required=True, help="Folder with project_out_*.json files")
    p.add_argument("--mode", required=True, choices=("delta-vs-k", "delta-vs-layers", "all"),
                   help="Which plot(s) to generate")
    p.add_argument("--layer", type=int, default=None,
                   help="Layer index (required for delta-vs-k and 'all')")
    p.add_argument("--k", type=int, default=None,
                   help="k value (required for delta-vs-layers and 'all')")
    p.add_argument("--variants", default=None,
                   help="Comma-separated variant filter (default: all)")
    p.add_argument("--save-dir", default=None,
                   help="Directory to save PNGs (default: show interactively)")
    args = p.parse_args()

    variants = [v.strip() for v in args.variants.split(",")] if args.variants else None
    all_results = load_results_dir(args.results_dir)
    logger.info("Loaded %d result files from %s", len(all_results), args.results_dir)

    do_k = args.mode in ("delta-vs-k", "all")
    do_layer = args.mode in ("delta-vs-layers", "all")

    if do_k:
        if args.layer is None:
            p.error("--layer is required for delta-vs-k")
        # Find the result for this layer
        match = [r for r in all_results if r.get("config", {}).get("layer_idx") == args.layer]
        if not match:
            p.error(f"No result file found for layer {args.layer}")
        save = f"{args.save_dir}/delta_vs_k_L{args.layer}.png" if args.save_dir else None
        plot_delta_vs_k(match[0], variants=variants, save_path=save)
        logger.info("Plotted delta-vs-k for layer %d", args.layer)

    if do_layer:
        if args.k is None:
            p.error("--k is required for delta-vs-layers")
        save = f"{args.save_dir}/delta_vs_layers_k{args.k}.png" if args.save_dir else None
        plot_delta_vs_layers(all_results, args.k, variants=variants, save_path=save)
        logger.info("Plotted delta-vs-layers for k=%d", args.k)

    if not args.save_dir:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
