"""Evaluation: loss and token-distribution comparison."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F


class LossEvaluator:
    """Mean causal-LM loss over batches (ignores padding when pad_token_id is set)."""

    def __init__(self, model: torch.nn.Module, pad_token_id: int | None = None) -> None:
        self.model = model
        # Find first non-meta device
        self._device = next(
            (p.device for p in model.parameters() if not p.is_meta), torch.device("cpu")
        )
        self.pad_token_id = pad_token_id

    @torch.no_grad()
    def evaluate(self, batches: List[torch.Tensor]) -> float:
        self.model.eval()
        losses: List[float] = []
        for batch in batches:
            batch = batch.to(self._device)
            labels = batch.clone()
            if self.pad_token_id is not None:
                labels[batch == self.pad_token_id] = -100
            losses.append(self.model(input_ids=batch, labels=labels).loss.item())
        return sum(losses) / len(losses) if losses else 0.0

    @torch.no_grad()
    def get_logits(self, batches: List[torch.Tensor]) -> torch.Tensor:
        """Concatenated logits [total_tokens, vocab] for all batches."""
        self.model.eval()
        parts = []
        for b in batches:
            out = self.model(input_ids=b.to(self._device))
            parts.append(out.logits.flatten(0, 1))
        return torch.cat(parts, dim=0)

    @torch.no_grad()
    def evaluate_and_get_logits(
        self, batches: List[torch.Tensor]
    ) -> Tuple[float, torch.Tensor]:
        """One forward per batch; returns (mean_loss, concatenated logits). Use for single-pass baseline/variation."""
        self.model.eval()
        losses: List[float] = []
        parts: List[torch.Tensor] = []
        for batch in batches:
            batch = batch.to(self._device)
            labels = batch.clone()
            if self.pad_token_id is not None:
                labels[batch == self.pad_token_id] = -100
            out = self.model(input_ids=batch, labels=labels)
            losses.append(out.loss.item())
            parts.append(out.logits.flatten(0, 1))
        mean_loss = sum(losses) / len(losses) if losses else 0.0
        return mean_loss, torch.cat(parts, dim=0)


class TokenDistributionComparator:
    """Compare next-token distributions before/after intervention (KL or CE)."""

    def __init__(self, metric: str = "kl") -> None:
        if metric not in ("kl", "ce"):
            raise ValueError(f"metric must be 'kl' or 'ce', got {metric!r}")
        self.metric = metric

    def compare(self, logits_before: torch.Tensor, logits_after: torch.Tensor) -> float:
        p_after = F.softmax(logits_after.float(), dim=-1)
        log_p_before = F.log_softmax(logits_before.float(), dim=-1)
        if self.metric == "kl":
            log_p_after = F.log_softmax(logits_after.float(), dim=-1)
            return (p_after * (log_p_after - log_p_before)).sum(-1).mean().item()
        # CE
        return (
            -(
                F.softmax(logits_before.float(), dim=-1)
                * F.log_softmax(logits_after.float(), dim=-1)
            )
            .sum(-1)
            .mean()
            .item()
        )


def confusion_matrix_top_k(
    logits_before: torch.Tensor,
    logits_after: torch.Tensor,
    top_k: int = 2,
) -> Tuple[torch.Tensor, List[int]]:
    """Confusion matrix of top-1 predictions before vs after, restricted to top_k tokens."""
    pred_b = logits_before.argmax(-1).cpu()
    pred_a = logits_after.argmax(-1).cpu()
    all_ids = torch.cat([pred_b, pred_a])
    uniq, counts = torch.unique(all_ids, return_counts=True)
    k = min(top_k, len(uniq))
    token_ids = uniq[counts.argsort(descending=True)[:k]].tolist()
    idx_map = {tid: i for i, tid in enumerate(token_ids)}
    mat = torch.zeros(k, k, dtype=torch.long)
    for pb, pa in zip(pred_b.view(-1).tolist(), pred_a.view(-1).tolist()):
        if pb in idx_map and pa in idx_map:
            mat[idx_map[pb], idx_map[pa]] += 1
    return mat, token_ids


def register_routing_capture(model: torch.nn.Module, layer_idx: int) -> Tuple[Any, List[torch.Tensor]]:
    """
    Register a forward hook to capture router logits during the next forward(s).
    Use this so routing is captured as part of your existing forward pass (e.g. get_logits).

    Returns:
        (hook_handle, captured_list). Each forward appends one [batch, seq, num_experts]
        tensor to captured_list. Call hook_handle.remove() when done, then
        routing_captures_to_selections(captured_list) to get [num_tokens] expert indices.
    """
    captured: List[torch.Tensor] = []

    def router_hook(module: torch.nn.Module, input: Any, output: Any) -> None:
        # Gate may return a single tensor [batch, seq, num_experts] or (logits, ...)
        if isinstance(output, tuple) and len(output) >= 1:
            logits = output[0]
        elif isinstance(output, torch.Tensor):
            logits = output
        else:
            return
        captured.append(logits.detach().cpu())  # [batch, seq, num_experts]

    layer_module = model.model.layers[layer_idx].block_sparse_moe
    handle = layer_module.gate.register_forward_hook(router_hook)
    return handle, captured


def routing_captures_to_selections(captured: List[torch.Tensor]) -> torch.Tensor:
    """Convert list of router logits [batch, seq, num_experts] to flat [num_tokens] expert indices."""
    
    return torch.cat([t.argmax(dim=-1).reshape(-1) for t in captured], dim=0)


def expert_routing_matrix(
    model: torch.nn.Module,
    layer_idx: int,
    batches: List[torch.Tensor],
    num_experts: int = 8,
) -> torch.Tensor:
    """
    Capture which experts are selected by the router for each token (does its own forwards).
    Prefer register_routing_capture + routing_captures_to_selections during your existing
    forward pass to avoid extra forwards.

    Args:
        model: The MoE model
        layer_idx: Which MoE layer to track
        batches: Input token batches
        num_experts: Number of experts (default 8)

    Returns:
        [num_tokens] tensor of expert indices selected for each token
    """
    handle, captured = register_routing_capture(model, layer_idx)
    try:
        with torch.no_grad():
            device = next(model.parameters()).device
            for batch in batches:
                _ = model(batch.to(device))
        return routing_captures_to_selections(captured)
    finally:
        handle.remove()


def expert_confusion_matrix(
    selections_before: torch.Tensor,
    selections_after: torch.Tensor,
    num_experts: int = 8,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Build confusion matrix showing expert migration: before (rows) vs after (columns).
    
    Args:
        selections_before: [num_tokens] expert indices before intervention
        selections_after: [num_tokens] expert indices after intervention
        num_experts: Number of experts
        
    Returns:
        matrix: [num_experts, num_experts] confusion matrix
        stats: Dictionary with statistics (stolen rate, migration counts, etc.)
    """
    mat = torch.zeros(num_experts, num_experts, dtype=torch.long)
    
    for before, after in zip(selections_before.tolist(), selections_after.tolist()):
        mat[before, after] += 1
    
    # Compute statistics
    total_tokens = len(selections_before)
    diagonal = torch.diag(mat).sum().item()  # Tokens that stayed with same expert
    off_diagonal = mat.sum().item() - diagonal  # Tokens that migrated
    
    stats = {
        "total_tokens": total_tokens,
        "stayed": diagonal,
        "migrated": off_diagonal,
        "migration_rate": off_diagonal / total_tokens if total_tokens > 0 else 0.0,
    }
    
    # Per-expert statistics
    for exp_idx in range(num_experts):
        original_count = mat[exp_idx, :].sum().item()  # Tokens originally routed to this expert
        retained = mat[exp_idx, exp_idx].item()  # Tokens that stayed
        lost = original_count - retained  # Tokens lost to other experts
        gained = mat[:, exp_idx].sum().item() - retained  # Tokens gained from other experts
        
        stats[f"expert_{exp_idx}"] = {
            "original": original_count,
            "retained": retained,
            "lost": lost,
            "gained": gained,
            "retention_rate": retained / original_count if original_count > 0 else 0.0,
        }
    
    return mat, stats
