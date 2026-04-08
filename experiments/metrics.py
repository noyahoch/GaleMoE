"""
Training metrics for Experiment 1.

Three metrics tracked during training:
  1. perplexity_indist / perplexity_ood  — held-out PPL as a function of steps
  2. expert_cv   — coefficient of variation (std/mean) of tokens-per-expert,
                   averaged across layers; measures routing collapse
  3. expert_freq — per-expert selection frequency on the last training window;
                   on eval steps also *_indist / *_ood / *_extra_eval / *_train_mix
                   from the same eval batches as each PPL (layers in metrics.freq_layers)
  4. moe_same_expert_cos_* — optional; mean pairwise cosine of hiddens co-routed to the
                   same expert (metrics.moe_geometry), on indist eval batches only
  5. moe_router_proj_energy_* — optional; mean cos²(h, w_e) for score-router rows w_e
                   on selected experts (same batches)

RouterMonitor registers forward hooks on every moe.gate so we capture
expert selections without modifying the model or transformers source.
All stats are written as JSONL to a log file, one record per log step.
"""

import json
import math
import os
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import OlmoeForCausalLM

from moe_geometry import MoEGeometryCollector


# ---------------------------------------------------------------------------
# RouterMonitor — hooks into every MoE gate
# ---------------------------------------------------------------------------

class RouterMonitor:
    """
    Accumulates token-per-expert counts across all MoE layers during forward passes.
    Call .reset() after each logging step.
    """

    def __init__(self, model: OlmoeForCausalLM):
        cfg = model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_tok
        # counts[layer_idx] = (n_experts,) float tensor (accumulated token counts)
        self.counts: dict[int, torch.Tensor] = {}
        self._hooks = []
        self._register(model)

    def _register(self, model: OlmoeForCausalLM):
        for layer_idx, layer in enumerate(model.model.layers):
            def _hook(module, inp, out, _idx=layer_idx):
                # HF OlmoeTopKRouter / GatedRouter: (router_probs, scores, indices)
                with torch.no_grad():
                    if isinstance(out, tuple) and len(out) >= 3:
                        selected = out[2]
                        dev = selected.device
                    else:
                        probs = F.softmax(out.float(), dim=-1)
                        _, selected = torch.topk(probs, self.top_k, dim=-1)  # (T, top_k)
                        dev = out.device
                    flat = selected.flatten()
                    counts = torch.zeros(self.n_experts, device=dev)
                    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
                    if _idx in self.counts:
                        self.counts[_idx] += counts.cpu()
                    else:
                        self.counts[_idx] = counts.cpu()

            h = layer.mlp.gate.register_forward_hook(_hook)
            self._hooks.append(h)

    def reset(self):
        self.counts.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    def expert_cv_per_layer(self) -> dict[int, float]:
        """CV = std/mean of token counts per expert, per layer."""
        cvs = {}
        for layer_idx, c in self.counts.items():
            mean = c.mean().item()
            std = c.std().item()
            cvs[layer_idx] = std / mean if mean > 0 else 0.0
        return cvs

    def mean_cv(self) -> float:
        """Average CV across all layers."""
        cvs = self.expert_cv_per_layer()
        return sum(cvs.values()) / len(cvs) if cvs else 0.0

    def expert_freq_per_layer(self) -> dict[int, list[float]]:
        """Normalised selection frequency per expert per layer."""
        freq = {}
        for layer_idx, c in self.counts.items():
            total = c.sum().item()
            freq[layer_idx] = (c / total).tolist() if total > 0 else c.tolist()
        return freq


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_perplexity_and_routing(
    model: OlmoeForCausalLM,
    dataloader,
    device,
    monitor: RouterMonitor,
    n_batches: int = 50,
    geometry_collector: MoEGeometryCollector | None = None,
) -> tuple[float, dict[int, list[float]]]:
    """
    Clears monitor, runs eval forwards, returns (PPL, full-layer expert_freq).
    Use for a clean routing histogram on one split (e.g. OOD only).
    If geometry_collector is set, its accumulators are reset here; gate hooks
    update it during the same forwards (indist-only recommended).
    """
    if geometry_collector is not None:
        geometry_collector.reset()
    monitor.reset()
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        n_tok = (labels != -100).sum().item()
        total_loss += out.loss.item() * n_tok
        total_tokens += n_tok
    model.train()
    ppl = math.exp(total_loss / total_tokens)
    freq = monitor.expert_freq_per_layer()
    return ppl, freq


# ---------------------------------------------------------------------------
# Logger — writes JSONL
# ---------------------------------------------------------------------------

class MetricsLogger:
    """Appends one JSON record per call to log()."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, record: dict):
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Convenience: build one record per log step
# ---------------------------------------------------------------------------

def build_log_record(
    step: int,
    train_loss: float,
    monitor: RouterMonitor,
    model: Optional[OlmoeForCausalLM] = None,
    indist_loader=None,
    ood_loader=None,
    extra_eval_loader=None,
    extra_eval_tag: Optional[str] = None,
    train_eval_loader=None,
    train_eval_batches: int = 0,
    device=None,
    eval_batches: int = 50,
    log_freq_layers: Optional[list[int]] = None,  # which layers to log full freq hist
    moe_geometry: bool = False,
    moe_geometry_max_tokens_per_expert: int = 128,
) -> dict:
    record: dict = {
        "step": step,
        "train_loss": train_loss,
        "expert_cv_mean": monitor.mean_cv(),
        "expert_cv_per_layer": monitor.expert_cv_per_layer(),
    }

    # expert frequency distribution for selected layers
    if log_freq_layers is not None:
        freq = monitor.expert_freq_per_layer()
        record["expert_freq"] = {str(l): freq[l] for l in log_freq_layers if l in freq}

    geometry: MoEGeometryCollector | None = None
    if moe_geometry and model is not None and indist_loader is not None:
        geometry = MoEGeometryCollector(
            model, max_tokens_per_expert=moe_geometry_max_tokens_per_expert
        )

    try:
        # held-out perplexity + routing on that split only (monitor.reset inside each call)
        if model is not None and indist_loader is not None:
            ppl_i, freq_i = eval_perplexity_and_routing(
                model,
                indist_loader,
                device,
                monitor,
                eval_batches,
                geometry_collector=geometry,
            )
            record["ppl_indist"] = ppl_i
            if log_freq_layers is not None:
                record["expert_freq_indist"] = {
                    str(l): freq_i[l] for l in log_freq_layers if l in freq_i
                }
            if geometry is not None:
                record.update(geometry.finalize_dict())
    finally:
        if geometry is not None:
            geometry.remove()

    if model is not None and ood_loader is not None:
        ppl_o, freq_o = eval_perplexity_and_routing(
            model, ood_loader, device, monitor, eval_batches
        )
        record["ppl_ood"] = ppl_o
        if log_freq_layers is not None:
            record["expert_freq_ood"] = {
                str(l): freq_o[l] for l in log_freq_layers if l in freq_o
            }

    if model is not None and extra_eval_loader is not None:
        ppl_x, freq_x = eval_perplexity_and_routing(
            model, extra_eval_loader, device, monitor, eval_batches
        )
        record["ppl_extra_eval"] = ppl_x
        if extra_eval_tag is not None:
            record["extra_eval_tag"] = extra_eval_tag
        if log_freq_layers is not None:
            record["expert_freq_extra_eval"] = {
                str(l): freq_x[l] for l in log_freq_layers if l in freq_x
            }

    if (
        model is not None
        and train_eval_loader is not None
        and train_eval_batches > 0
    ):
        ppl_tr, freq_tr = eval_perplexity_and_routing(
            model, train_eval_loader, device, monitor, train_eval_batches
        )
        record["ppl_train_mix"] = ppl_tr
        if log_freq_layers is not None:
            record["expert_freq_train_mix"] = {
                str(l): freq_tr[l] for l in log_freq_layers if l in freq_tr
            }

    return record
