"""
Optional training health stats for GatedRouter experiments.

Tracks whether W_gate is actually moving and whether gradients reach it.
"""

from __future__ import annotations

import torch
from transformers import OlmoeForCausalLM


def snapshot_w_gate_weights(model: OlmoeForCausalLM) -> dict[int, torch.Tensor]:
    """CPU float32 clones of W_gate.weight per layer (only layers with GatedRouter)."""
    out: dict[int, torch.Tensor] = {}
    for i, layer in enumerate(model.model.layers):
        gate = layer.mlp.gate
        if hasattr(gate, "W_gate"):
            out[i] = gate.W_gate.weight.detach().float().cpu().clone()
    return out


def w_gate_delta_rms(model: OlmoeForCausalLM, init_snap: dict[int, torch.Tensor]) -> float:
    """RMS of (W_gate - W_gate_init) over all gated layers."""
    if not init_snap:
        return 0.0
    sq_sum = 0.0
    n_el = 0
    for i, w0 in init_snap.items():
        w = model.model.layers[i].mlp.gate.W_gate.weight.detach().float().cpu()
        d = (w - w0).flatten()
        sq_sum += float((d * d).sum().item())
        n_el += d.numel()
    return (sq_sum / max(n_el, 1)) ** 0.5


def gated_gate_grad_norm(model: OlmoeForCausalLM) -> dict[str, float]:
    """
    L2 norm of gradients on GatedRouter weights after backward (before clip/step).
    Keys: w_gate, w_score (if present and grad exists).
    """
    sq_gate = sq_score = 0.0
    for layer in model.model.layers:
        gate = layer.mlp.gate
        if hasattr(gate, "W_gate") and gate.W_gate.weight.grad is not None:
            g = gate.W_gate.weight.grad
            sq_gate += float(g.float().detach().square().sum().cpu().item())
        if hasattr(gate, "W_score") and gate.W_score.weight.grad is not None:
            g = gate.W_score.weight.grad
            sq_score += float(g.float().detach().square().sum().cpu().item())
    return {
        "w_gate_grad_norm": sq_gate ** 0.5,
        "w_score_grad_norm": sq_score ** 0.5,
    }
