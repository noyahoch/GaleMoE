"""Experiment 2: Vector interventions (inject, subtract, project_out); compare loss and token distributions.

Apply different vector operations to router rows (project out, inject, subtract) using
SVD/orthogonal/random directions. Measure loss, delta, token distribution metric (KL/CE),
and save confusion matrices.
"""
from __future__ import annotations

import argparse
import logging

from .core import ExperimentConfig
from .runners import run_vector_intervention_experiment


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Experiment 2: Vector interventions (inject, subtract, project_out); compare loss and token distributions.",
    )
    parser.add_argument("--svd_dir", type=str, required=True, help="Directory with expert SVD pickle files")
    parser.add_argument("--layer_idx", type=int, required=True, help="MoE layer index")
    parser.add_argument("--output_file", type=str, default="results_vector_interventions.json")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--model-id", type=str, default="mistralai/Mixtral-8x7B-v0.1")
    parser.add_argument("--model-tag", type=str, default="mistralai_Mixtral_8x7B_v0.1")
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variations",
        type=str,
        default="svd,orthogonal,random",
        help="Comma-separated variants: svd, orthogonal, random",
    )
    parser.add_argument(
        "--interventions",
        type=str,
        default="project_out,inject,subtract",
        help="Comma-separated: project_out, inject, subtract",
    )
    parser.add_argument(
        "--inject-subtract-scale",
        type=float,
        default=1.0,
        help="Scale for inject/subtract (default 1.0)",
    )
    parser.add_argument(
        "--distribution-metric",
        type=str,
        default="kl",
        choices=("kl", "ce"),
        help="Token distribution metric: kl or ce",
    )
    parser.add_argument(
        "--confusion-top-k",
        type=int,
        default=2,
        help="Top-k tokens for confusion matrix (default 2)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        choices=("wikitext", "wikitext_titles", "text"),
        help="Dataset: wikitext (long chunks), wikitext_titles (one short seq per title; use with --seq-len 32 --batch-size 2000), or text (--text-file)",
    )
    parser.add_argument(
        "--text-file",
        type=str,
        default=None,
        help="Path to text file when --dataset=text",
    )
    parser.add_argument(
        "--top-k",
        type=str,
        default="1",
        metavar="K",
        help="Number of top singular vectors to project out (default: 1). For project-out experiment use run_project_out with comma-separated list.",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit (BitsAndBytes). Matches gate-hook; halves weight memory. Requires: pip install bitsandbytes",
    )

    args = parser.parse_args()

    variations = [v.strip() for v in args.variations.split(",") if v.strip()]
    interventions = [i.strip() for i in args.interventions.split(",") if i.strip()]
    if args.dataset == "text" and not args.text_file:
        parser.error("--dataset=text requires --text-file")

    top_k = [int(x.strip()) for x in args.top_k.split(",") if x.strip()]
    if not top_k:
        top_k = [1]

    cfg = ExperimentConfig(
        svd_dir=args.svd_dir,
        layer_idx=args.layer_idx,
        output_file=args.output_file,
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
        load_in_8bit=args.load_in_8bit,
    )

    run_vector_intervention_experiment(
        cfg,
        interventions=interventions,
        inject_subtract_scale=args.inject_subtract_scale,
        distribution_metric_name=args.distribution_metric,
        confusion_top_k=args.confusion_top_k,
    )


if __name__ == "__main__":
    main()
