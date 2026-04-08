"""
MoE geometry metrics on pre-router hidden states (eval hooks on each layer gate).

1) same_expert_cos_* — For each layer, among token positions that route to expert e
   (e appears in any top-k slot), mean pairwise cosine similarity of normalized hiddens.
   Layer value = mean over experts with ≥2 tokens (subsampled). Scalar = mean over layers.

2) router_proj_energy_* — Mean cos²(h, w_e) over tokens and top-k slots, where w_e is the
   score-router row for the selected expert (nn.Linear.weight[e] or GatedRouter.W_score).
   Interpretable as squared cosine / directional energy in router weight directions.

Both are computed in float32 during eval; enable via metrics.moe_geometry in base.yaml.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import OlmoeForCausalLM


def _router_score_weight(gate: torch.nn.Module) -> torch.Tensor:
    """(num_experts, hidden_size) used as rows w_e in logits = h @ w_e^T (score path)."""
    if hasattr(gate, "W_score"):
        return gate.W_score.weight
    return gate.weight


class MoEGeometryCollector:
    """
    Forward hooks on each layer's mlp.gate. Call reset() before an eval loop, then
    finalize_dict() after forwards; remove() to unregister hooks.
    """

    def __init__(self, model: OlmoeForCausalLM, max_tokens_per_expert: int = 128):
        self.model = model
        self.max_tokens_per_expert = max_tokens_per_expert
        self.top_k = model.config.num_experts_per_tok
        self.n_experts = model.config.num_experts
        n_layers = model.config.num_hidden_layers
        self._proj_sum: dict[int, float] = {i: 0.0 for i in range(n_layers)}
        self._proj_n: dict[int, int] = {i: 0 for i in range(n_layers)}
        self._cos_sum: dict[int, float] = {i: 0.0 for i in range(n_layers)}
        self._cos_n: dict[int, int] = {i: 0 for i in range(n_layers)}
        self._hooks: list = []
        for layer_idx, layer in enumerate(model.model.layers):
            hook = layer.mlp.gate.register_forward_hook(
                lambda mod, inp, out, lid=layer_idx: self._on_gate(mod, inp, out, lid)
            )
            self._hooks.append(hook)

    def reset(self) -> None:
        for i in self._proj_sum:
            self._proj_sum[i] = 0.0
            self._proj_n[i] = 0
            self._cos_sum[i] = 0.0
            self._cos_n[i] = 0

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _on_gate(
        self,
        gate_module: torch.nn.Module,
        inp: tuple[torch.Tensor, ...],
        out: torch.Tensor,
        layer_idx: int,
    ) -> None:
        x = inp[0]
        if x.dim() != 2:
            return
        if isinstance(out, tuple) and len(out) >= 3:
            probs = out[0].float()
            sel = out[2]
            if probs.dim() != 2:
                return
        else:
            logits = out
            if logits.dim() != 2:
                return
            probs = F.softmax(logits.float(), dim=-1)
            _, sel = torch.topk(probs, self.top_k, dim=-1)
        x_f = x.float()
        T, _ = probs.shape
        W = _router_score_weight(gate_module).float()
        Wn = F.normalize(W, dim=1, eps=1e-8)
        xn = F.normalize(x_f, dim=1, eps=1e-8)

        w_pick = Wn[sel]
        cos_slot = (xn.unsqueeze(1) * w_pick).sum(dim=-1)
        pe = cos_slot.pow(2).mean()
        self._proj_sum[layer_idx] += pe.item()
        self._proj_n[layer_idx] += 1

        expert_cos_means: list[float] = []
        for e in range(self.n_experts):
            mask = (sel == e).any(dim=1)
            n = int(mask.sum().item())
            if n < 2:
                continue
            Hsub = xn[mask]
            if n > self.max_tokens_per_expert:
                idx = torch.randperm(n, device=Hsub.device)[: self.max_tokens_per_expert]
                Hsub = Hsub[idx]
                n = Hsub.shape[0]
            if n < 2:
                continue
            sim = Hsub @ Hsub.T
            tri = torch.triu_indices(n, n, offset=1, device=sim.device)
            expert_cos_means.append(sim[tri[0], tri[1]].mean().item())

        if expert_cos_means:
            self._cos_sum[layer_idx] += sum(expert_cos_means) / len(expert_cos_means)
            self._cos_n[layer_idx] += 1

    def finalize_dict(self) -> dict:
        """JSON-serializable keys; means skip layers with no data."""
        proj_layer: dict[str, float] = {}
        cos_layer: dict[str, float] = {}
        for i in self._proj_sum:
            if self._proj_n[i] > 0:
                proj_layer[str(i)] = self._proj_sum[i] / self._proj_n[i]
            if self._cos_n[i] > 0:
                cos_layer[str(i)] = self._cos_sum[i] / self._cos_n[i]

        proj_vals = list(proj_layer.values())
        cos_vals = list(cos_layer.values())
        out: dict = {
            "moe_router_proj_energy_per_layer": proj_layer,
            "moe_router_proj_energy_mean": sum(proj_vals) / len(proj_vals) if proj_vals else 0.0,
            "moe_same_expert_cos_per_layer": cos_layer,
            "moe_same_expert_cos_mean": sum(cos_vals) / len(cos_vals) if cos_vals else 0.0,
        }
        return out
