"""
Evaluate perplexity for each condition on the two held-out sets defined in the config.

Usage:
  python eval.py --config configs/baseline.yaml \\
      --run_dirs runs/baseline/final runs/gated_random/final runs/gated_svd/final
"""

import argparse
import math
import os
import sys

import torch
from transformers import AutoTokenizer, OlmoeForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from config import load_config
from data import build_eval_dataloader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/baseline.yaml",
                   help="Any condition config — used only for eval dataset definitions")
    p.add_argument("--run_dirs", nargs="+", required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batches", type=int, default=200)
    p.add_argument("--dtype", type=str, default="bfloat16")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, dataloader, device, n_batches: int):
    model.eval()
    total_loss = total_tokens = 0
    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = input_ids.clone()
        labels[attention_mask == 0] = -100
        out     = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        n_tok   = (labels != -100).sum().item()
        total_loss   += out.loss.item() * n_tok
        total_tokens += n_tok
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)


def main():
    args  = parse_args()
    cfg   = load_config(args.config)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"In-dist eval : {cfg.data.indist_eval.dataset} ({cfg.data.indist_eval.split})")
    print(f"OOD eval     : {cfg.data.ood_eval.dataset} ({cfg.data.ood_eval.split})")

    indist_loader = build_eval_dataloader(cfg.data.indist_eval, tokenizer,
                                          args.batch_size, cfg.training.max_length)
    ood_loader    = build_eval_dataloader(cfg.data.ood_eval, tokenizer,
                                          args.batch_size, cfg.training.max_length)

    results = {}
    for run_dir in args.run_dirs:
        label = os.path.basename(run_dir.rstrip("/"))
        print(f"\nLoading {run_dir} ...")
        model = OlmoeForCausalLM.from_pretrained(run_dir, torch_dtype=dtype).to(device)

        loss_id,  ppl_id  = evaluate(model, indist_loader, device, args.eval_batches)
        loss_ood, ppl_ood = evaluate(model, ood_loader,    device, args.eval_batches)

        results[label] = dict(indist_ppl=ppl_id, ood_ppl=ppl_ood)
        print(f"  in-dist  loss={loss_id:.4f}  ppl={ppl_id:.2f}")
        print(f"  ood      loss={loss_ood:.4f}  ppl={ppl_ood:.2f}")
        del model
        torch.cuda.empty_cache()

    print("\n=== Summary ===")
    print(f"{'condition':22s}  {'in-dist ppl':>12s}  {'ood ppl':>10s}  {'delta':>8s}")
    print("-" * 60)
    for label, r in results.items():
        delta = r["ood_ppl"] - r["indist_ppl"]
        print(f"  {label:20s}  {r['indist_ppl']:12.2f}  {r['ood_ppl']:10.2f}  {delta:+8.2f}")


if __name__ == "__main__":
    main()
