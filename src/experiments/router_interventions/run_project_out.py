"""Experiment 1: Project out SVDs from all experts and compare loss.

Project the SVD (or orthogonal/random) direction out of each expert's router row,
then measure loss and delta vs baseline. No inject/subtract, no token distribution metrics.
"""
from __future__ import annotations

import argparse
import logging

from .core import ExperimentConfig
from .runners import run_project_out_experiment


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Experiment 1: Project out SVDs from all experts, compare loss and delta.",
    )
    parser.add_argument("--svd_dir", type=str, required=True, help="Directory with expert SVD pickle files")
    parser.add_argument("--layer_idx", type=int, required=True, help="MoE layer index")
    parser.add_argument("--output_file", type=str, default="results_project_out.json")
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
        default="svd,orthogonal,random,zero,shuffle",
        help="Comma-separated: svd, orthogonal, random, zero, shuffle",
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
        metavar="K1[,K2,...]",
        help="Top singular vector count(s) to project out: single int or comma-separated list (e.g. 1,2,4,8). Default: 1",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit (BitsAndBytes). Matches gate-hook; halves weight memory. Requires: pip install bitsandbytes",
    )

    args = parser.parse_args()

    variations = [v.strip() for v in args.variations.split(",") if v.strip()]
    if args.dataset == "text" and not args.text_file:
        parser.error("--dataset=text requires --text-file")

    top_k = [int(x.strip()) for x in args.top_k.split(",") if x.strip()]
    if not top_k:
        parser.error("--top-k must contain at least one integer")

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

    run_project_out_experiment(cfg)


if __name__ == "__main__":
    main()
