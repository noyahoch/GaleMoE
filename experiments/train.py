"""
Experiment 1: Gated Router on OLMoE-1B-7B

Usage:
  python train.py --config configs/baseline.yaml
  python train.py --config configs/gated_svd.yaml
  python train.py --config configs/gated_svd.yaml training.lr=2e-4   # override
"""

import argparse
import os
import sys

import torch
try:
    import wandb as _wandb
except ImportError:
    _wandb = None
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    OlmoeForCausalLM,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, os.path.dirname(__file__))
from config import load_config
from replace_router import replace_routers
from freeze import apply_freezing
from data import MixtureConfig, build_train_dataloader, build_eval_dataloader
from metrics import RouterMonitor, MetricsLogger, build_log_record
from train_diagnostics import (
    gated_gate_grad_norm,
    snapshot_w_gate_weights,
    w_gate_delta_rms,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to condition yaml (e.g. configs/gated_svd.yaml)")
    p.add_argument("overrides", nargs="*", help="key=value overrides, e.g. training.lr=2e-4")
    return p.parse_args()


def main():
    args = parse_args()

    overrides = {}
    for item in args.overrides:
        k, _, v = item.partition("=")
        # cast to float/int if possible
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        overrides[k] = v

    cfg = load_config(args.config, overrides or None)

    condition = getattr(cfg, "condition", "baseline")
    output_dir = cfg.checkpointing.output_dir or f"runs/{condition}"
    os.makedirs(output_dir, exist_ok=True)

    wandb_cfg = getattr(cfg, "wandb", None)
    wb_run = None
    if wandb_cfg is not None and getattr(wandb_cfg, "enabled", False):
        if _wandb is None:
            print("WARNING: wandb not installed — skipping wandb logging.", flush=True)
        else:
            wb_run = _wandb.init(
                project=getattr(wandb_cfg, "project", "GaleMoE"),
                name=getattr(wandb_cfg, "run_name", None) or condition,
                tags=list(getattr(wandb_cfg, "tags", []) or []),
                config=cfg.to_dict(),
            )
            print(f"[wandb] run: {wb_run.url}", flush=True)

    dtype  = torch.bfloat16 if cfg.model.dtype == "bfloat16" else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # One optimizer step consumes grad_accum micro-batches, each of size batch_size × max_length tokens.
    tokens_per_opt_step = (
        cfg.training.batch_size * cfg.training.grad_accum * cfg.training.max_length
    )
    total_opt_steps = cfg.training.total_tokens // tokens_per_opt_step
    total_micro_steps = total_opt_steps * cfg.training.grad_accum
    print(
        f"[{condition}] device={device}  "
        f"{cfg.training.total_tokens / 1e9:.1f}B tokens  →  "
        f"{total_opt_steps:,} optimizer steps  ({tokens_per_opt_step:,} tokens/opt-step),  "
        f"{total_micro_steps:,} micro-batches"
    )

    # model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = getattr(cfg.model, "attn_implementation", "sdpa")
    try:
        model = OlmoeForCausalLM.from_pretrained(
            cfg.model.model_id,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
        )
    except TypeError:
        model = OlmoeForCausalLM.from_pretrained(cfg.model.model_id, torch_dtype=dtype)
        print("  (attn_implementation not supported by this transformers; using default)", flush=True)
    else:
        print(f"  attn_implementation={attn_impl}", flush=True)
    model = replace_routers(model, condition, cfg)
    if condition != "baseline":
        g0 = model.model.layers[0].mlp.gate
        if hasattr(g0, "W_gate"):
            wgn = g0.W_gate.weight.detach().float().norm().item()
            print(f"Layer0 gate: {type(g0).__name__}  ||W_gate||={wgn:.4f}", flush=True)
        else:
            print(
                f"WARNING: layer0 gate is {type(g0).__name__} (expected GatedRouter for {condition})",
                flush=True,
            )
    apply_freezing(model, cfg.training.trainable)
    model = model.to(device)
    model.train()

    train_diagnostics = bool(getattr(cfg.metrics, "train_diagnostics", True))
    w_gate_init_snap = snapshot_w_gate_weights(model) if train_diagnostics else {}
    last_gate_grads: dict[str, float] = {"w_gate_grad_norm": 0.0, "w_score_grad_norm": 0.0}
    if train_diagnostics and w_gate_init_snap:
        print(
            "[train_diagnostics] Logging W_gate drift vs init + gate grad norms on each log_every "
            "(see metrics JSONL: w_gate_delta_rms, w_gate_grad_norm, w_score_grad_norm).",
            flush=True,
        )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        "[speed] Only "
        f"{n_trainable:,} trainable params, but each step still runs a full forward+backward "
        "through the whole OLMoE stack (attention, all MoE experts used in routing, LM head). "
        "Freezing does not skip those layers—it only stops most weights from updating. "
        "Eval every eval_every is also many extra full forwards.",
        flush=True,
    )

    # data
    mixture = MixtureConfig(
        general_prob=cfg.data.general_prob,
        code_prob=cfg.data.code_prob,
        science_prob=cfg.data.science_prob,
        code_language=cfg.data.code_language,
        extra_dataset=cfg.data.extra_dataset,
        extra_prob=cfg.data.extra_prob,
        seed=cfg.data.seed,
    )
    print(f"Mixture: general={mixture.general_prob}  code={mixture.code_prob}  science={mixture.science_prob}")

    train_loader  = build_train_dataloader(tokenizer, cfg.training.batch_size,
                                           cfg.training.max_length, mixture,
                                           num_workers=cfg.data.num_workers)
    eval_bs = getattr(cfg.metrics, "eval_batch_size", 4)
    indist_loader = build_eval_dataloader(cfg.data.indist_eval, tokenizer,
                                          batch_size=eval_bs, max_length=cfg.training.max_length)
    ood_loader    = build_eval_dataloader(cfg.data.ood_eval, tokenizer,
                                          batch_size=eval_bs, max_length=cfg.training.max_length)

    extra_eval_loader = None
    extra_eval_tag    = None
    raw_extra         = getattr(cfg.data, "extra_eval", None)
    if raw_extra is not None and getattr(raw_extra, "dataset", None):
        extra_eval_loader = build_eval_dataloader(
            raw_extra, tokenizer, batch_size=eval_bs, max_length=cfg.training.max_length
        )
        extra_eval_tag = getattr(raw_extra, "tag", None) or "extra"

    train_eval_batches = int(getattr(cfg.metrics, "train_eval_batches", 0) or 0)
    train_eval_loader  = None
    if train_eval_batches > 0:
        train_eval_loader = build_train_dataloader(
            tokenizer, cfg.training.batch_size, cfg.training.max_length, mixture,
            num_workers=cfg.data.num_workers,
        )

    # metrics
    monitor     = RouterMonitor(model)
    freq_layers = cfg.metrics.freq_layers
    logger      = MetricsLogger(os.path.join(output_dir, "metrics.jsonl"))

    # optimiser
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=total_opt_steps,
    )

    # training loop
    data_iter    = iter(train_loader)
    global_step  = 0
    running_loss = 0.0
    optimizer.zero_grad()

    print(
        f"Training for {total_micro_steps:,} micro-batches "
        f"({total_opt_steps:,} optimizer steps)..."
    )
    while global_step < total_micro_steps:
        batch          = next(data_iter)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss / cfg.training.grad_accum
        loss.backward()
        if train_diagnostics and (global_step + 1) % cfg.training.grad_accum == 0:
            last_gate_grads = gated_gate_grad_norm(model)
        running_loss += loss.item() * cfg.training.grad_accum

        if (global_step + 1) % cfg.training.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1

        if global_step % cfg.metrics.log_every == 0:
            avg_loss = running_loss / cfg.metrics.log_every
            running_loss = 0.0
            do_eval = (global_step % cfg.metrics.eval_every == 0)

            if do_eval:
                print(
                    f"  running eval (≤{cfg.metrics.eval_batches} batches per loader) …",
                    flush=True,
                )

            record = build_log_record(
                step=global_step,
                train_loss=avg_loss,
                monitor=monitor,
                model=model if do_eval else None,
                indist_loader=indist_loader if do_eval else None,
                ood_loader=ood_loader if do_eval else None,
                extra_eval_loader=extra_eval_loader if do_eval else None,
                extra_eval_tag=extra_eval_tag,
                train_eval_loader=train_eval_loader if do_eval else None,
                train_eval_batches=train_eval_batches if do_eval else 0,
                device=device,
                eval_batches=cfg.metrics.eval_batches,
                log_freq_layers=freq_layers if do_eval else None,
                moe_geometry=bool(getattr(cfg.metrics, "moe_geometry", False)),
                moe_geometry_max_tokens_per_expert=int(
                    getattr(cfg.metrics, "moe_geometry_max_tokens_per_expert", 128) or 128
                ),
            )
            if train_diagnostics and w_gate_init_snap:
                record["w_gate_delta_rms"] = w_gate_delta_rms(model, w_gate_init_snap)
                record["w_gate_grad_norm"] = last_gate_grads["w_gate_grad_norm"]
                record["w_score_grad_norm"] = last_gate_grads["w_score_grad_norm"]
            logger.log(record)
            if wb_run is not None:
                wb_scalars = {
                    k: v for k, v in record.items()
                    if isinstance(v, (int, float))
                }
                wb_run.log(wb_scalars, step=global_step)
            monitor.reset()

            msg = (f"  step {global_step}/{total_micro_steps}  "
                   f"loss={avg_loss:.4f}  cv={record['expert_cv_mean']:.3f}")
            if train_diagnostics and w_gate_init_snap:
                msg += (
                    f"  dWg={record.get('w_gate_delta_rms', 0):.2e}  "
                    f"grad_wg={record.get('w_gate_grad_norm', 0):.2e}"
                )
            if do_eval:
                msg += (f"  ppl_id={record.get('ppl_indist', 0):.2f}"
                        f"  ppl_ood={record.get('ppl_ood', 0):.2f}")
                if "ppl_train_mix" in record:
                    msg += f"  ppl_tr={record['ppl_train_mix']:.2f}"
                if "ppl_extra_eval" in record:
                    msg += f"  ppl_x={record['ppl_extra_eval']:.2f}"
                if "moe_same_expert_cos_mean" in record:
                    msg += (
                        f"  cos_ex={record['moe_same_expert_cos_mean']:.3f}"
                        f"  pe={record['moe_router_proj_energy_mean']:.3f}"
                    )
            print(msg)

        if global_step % cfg.checkpointing.save_every == 0:
            ckpt = os.path.join(output_dir, f"step_{global_step}")
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"  saved -> {ckpt}")

    monitor.remove()
    if wb_run is not None:
        wb_run.finish()
    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Done. Final model -> {final_path}")
    print(f"Metrics -> {os.path.join(output_dir, 'metrics.jsonl')}")


if __name__ == "__main__":
    main()
