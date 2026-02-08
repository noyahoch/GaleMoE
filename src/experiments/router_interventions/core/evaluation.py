"""Loss and token-distribution evaluation for router interventions."""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F


class LossEvaluator:
    """Evaluates mean causal LM loss over batches (labels = input_ids).
    If pad_token_id is set, positions with that id are ignored in the loss (set to -100).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        pad_token_id: int | None = None,
    ) -> None:
        self.model = model
        self._device = next(model.parameters()).device
        self._pad_token_id = pad_token_id

    @torch.no_grad()
    def evaluate(self, batches: List[torch.Tensor]) -> float:
        """Return mean loss over the given batches."""
        self.model.eval()
        losses: List[float] = []
        for batch in batches:
            batch = batch.to(self._device)
            labels = batch.clone()
            if self._pad_token_id is not None:
                labels[batch == self._pad_token_id] = -100
            out = self.model(input_ids=batch, labels=labels)
            losses.append(out.loss.item())
        return sum(losses) / len(losses) if losses else 0.0

    @torch.no_grad()
    def get_logits(self, batches: List[torch.Tensor]) -> torch.Tensor:
        """Return logits for all batches, concatenated on the batch dimension.
        Shape: [total_tokens, vocab_size] where total_tokens = sum(batch.shape[0] * batch.shape[1]).
        """
        self.model.eval()
        logits_list: List[torch.Tensor] = []
        for batch in batches:
            batch = batch.to(self._device)
            out = self.model(input_ids=batch)
            # [B, S, V] -> [B*S, V]
            logits_list.append(out.logits.flatten(0, 1))
        return torch.cat(logits_list, dim=0)


class TokenDistributionComparator:
    """Compares next-token distributions before vs after an intervention (KL or CE)."""

    VALID_METRICS = ("kl", "ce")

    def __init__(self, metric: str = "kl") -> None:
        if metric not in self.VALID_METRICS:
            raise ValueError(
                f"metric must be one of {self.VALID_METRICS}, got {metric!r}"
            )
        self.metric = metric

    def compare(
        self,
        logits_before: torch.Tensor,
        logits_after: torch.Tensor,
    ) -> float:
        """Return mean comparison over N positions.

        Args:
            logits_before: [N, vocab_size] logits with original model.
            logits_after: [N, vocab_size] logits with modified model.

        Returns:
            Scalar: mean KL(after || before) or mean cross-entropy, depending on self.metric.
        """
        if logits_before.shape != logits_after.shape:
            raise ValueError(
                "logits_before and logits_after must have the same shape"
            )
        log_p_before = F.log_softmax(logits_before.float(), dim=-1)
        p_after = F.softmax(logits_after.float(), dim=-1)
        if self.metric == "kl":
            log_p_after = F.log_softmax(logits_after.float(), dim=-1)
            kl = (p_after * (log_p_after - log_p_before)).sum(dim=-1)
            return kl.mean().item()
        # metric == "ce"
        p_before = F.softmax(logits_before.float(), dim=-1)
        ce_cross = -(
            p_before * F.log_softmax(logits_after.float(), dim=-1)
        ).sum(dim=-1)
        return ce_cross.mean().item()


def confusion_matrix_top_k(
    logits_before: torch.Tensor,
    logits_after: torch.Tensor,
    top_k: int = 2,
) -> Tuple[torch.Tensor, List[int]]:
    """Build a confusion matrix of top-1 predictions before vs after.

    Rows = predicted token (argmax) before, cols = predicted token after.
    Restricted to the top-K most frequently predicted token IDs (by combined count)
    so the matrix is plottable. Default K=2 matches Mixtral's top-k experts per token.

    Args:
        logits_before: [N, vocab_size] logits with original model.
        logits_after: [N, vocab_size] logits with modified model.
        top_k: Number of token IDs to keep (by frequency). Matrix shape is
               min(top_k, num_unique_tokens). Default 2 (Mixtral's expert top-k).
               Use a larger k for finer detail.

    Returns:
        matrix: [K, K] count matrix (K = min(top_k, num_unique)).
        token_ids: length-K list of token IDs corresponding to rows/columns.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if logits_before.shape != logits_after.shape:
        raise ValueError("logits_before and logits_after must have the same shape")
    pred_before = logits_before.argmax(dim=-1).cpu()  # [N]
    pred_after = logits_after.argmax(dim=-1).cpu()    # [N]
    # Combined counts for each token ID
    all_ids = torch.cat([pred_before, pred_after])
    unique, counts = torch.unique(all_ids, return_counts=True)
    # Top-K by count (k is adaptable)
    k = min(top_k, len(unique))
    _, idx = counts.sort(descending=True)
    token_ids = unique[idx[:k]].tolist()
    id_to_idx = {tid: i for i, tid in enumerate(token_ids)}
    # Build KxK matrix (only count pairs where both in top-K)
    matrix = torch.zeros((k, k), dtype=torch.long)
    for i in range(pred_before.numel()):
        pb, pa = pred_before.view(-1)[i].item(), pred_after.view(-1)[i].item()
        if pb in id_to_idx and pa in id_to_idx:
            matrix[id_to_idx[pb], id_to_idx[pa]] += 1
    return matrix, token_ids
