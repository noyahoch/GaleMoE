"""Plot hijack/transplant result JSONs (delta vs k, expert migration heatmaps).

Usage::

    # Delta vs k + migration heatmaps for one k/variant, save to dir
    python -m src.experiments.router_interventions.plot_vector_intervention_results \\
        alignment_ablations_results/results_hijack_try_1/hijack_L5_Src5_k1-4-8-256_S3.0_wikitext_qnone.json \\
        --save-dir alignment_ablations_results/results_hijack_try_1/plots

    # Only delta vs k, show interactively
    python -m src.experiments.router_interventions.plot_vector_intervention_results \\
        alignment_ablations_results/results_hijack_try_1/hijack_L5_Src5_k1-4-8-256_S3.0_wikitext_qnone.json

    # Migration heatmap for k=8, variant svd_sum_normalized_hijack
    python -m src.experiments.router_interventions.plot_vector_intervention_results \\
        alignment_ablations_results/results_hijack_try_1/hijack_L5_Src5_k1-4-8-256_S3.0_wikitext_qnone.json \\
        --migration --k 8 --variant svd_sum_normalized_hijack --save-dir alignment_ablations_results/results_hijack_try_1/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .viz import (
    load_results,
    plot_delta_vs_k,
    plot_expert_migration_heatmap,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot vector intervention (hijack/transplant) result JSON."
    )
    p.add_argument(
        "json_path",
        type=Path,
        help="Path to hijack_*.json or transplant_*.json",
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory to save figures (default: show only)",
    )
    p.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated variant keys for delta-vs-k (default: all)",
    )
    p.add_argument(
        "--migration",
        action="store_true",
        help="Also plot expert migration heatmap(s)",
    )
    p.add_argument(
        "--k",
        type=int,
        default=None,
        help="k value for migration heatmap (required if --migration)",
    )
    p.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant key for migration heatmap (default: first available)",
    )
    args = p.parse_args()

    results = load_results(args.json_path)
    config = results.get("config", {})
    layer = config.get("layer_idx", "?")
    source_expert = config.get("source_expert")
    num_experts = config.get("num_experts", 8)

    variants = None
    if args.variants:
        variants = [v.strip() for v in args.variants.split(",")]

    save_dir = args.save_dir
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # 1) Delta vs k
    try:
        fig = plot_delta_vs_k(
            results,
            variants=variants,
            title=f"Layer {layer}, source expert {source_expert}",
            save_path=save_dir / "delta_vs_k.png" if save_dir else None,
        )
        if not save_dir:
            import matplotlib.pyplot as plt

            plt.show()
    except Exception as e:
        print(f"Delta vs k: {e}")

    # 2) Migration heatmap(s)
    if args.migration:
        by_k = results.get("by_k", {})
        if not by_k:
            print("No by_k in results, skipping migration plot.")
        else:
            k_val = args.k
            if k_val is None:
                k_val = int(min(by_k.keys(), key=lambda x: int(x)))
                print(f"Using k={k_val} (use --k to override)")
            k_str = str(k_val)
            if k_str not in by_k:
                print(f"k={k_val} not in results. Available: {list(by_k.keys())}")
            else:
                entries = by_k[k_str]
                variant_key = args.variant
                if not variant_key:
                    for k in entries:
                        if (
                            isinstance(entries.get(k), dict)
                            and "expert_confusion_matrix" in entries[k]
                        ):
                            variant_key = k
                            break
                if not variant_key or variant_key not in entries:
                    print(
                        f"Variant {variant_key or '?'} not found. Available: {list(entries.keys())}"
                    )
                else:
                    entry = entries[variant_key]
                    mat = entry.get("expert_confusion_matrix")
                    if mat is None:
                        print("No expert_confusion_matrix in entry.")
                    else:
                        out_path = None
                        if save_dir:
                            out_path = (
                                save_dir / f"migration_k{k_val}_{variant_key}.png"
                            )
                        plot_expert_migration_heatmap(
                            mat,
                            num_experts=num_experts,
                            source_expert=source_expert,
                            title=f"k={k_val} — {variant_key}",
                            save_path=out_path,
                        )
                        if not save_dir:
                            import matplotlib.pyplot as plt

                            plt.show()

    if save_dir:
        print(f"Saved to {save_dir}")


if __name__ == "__main__":
    main()
