"""Context manager for temporarily modifying and restoring Mixtral router weights."""
from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


class RouterManager:
    """Locate, apply, and restore the gate (router) weights for one Mixtral MoE layer.
    Gate weights are cloned on first use (original_weights or apply_weights) so that
    a forward pass can be run first to materialize lazy (meta) parameters.
    """

    def __init__(self, model: nn.Module, layer_idx: int) -> None:
        self.model = model
        self.layer_idx = layer_idx
        self._gate = self._find_gate()
        self._original: Optional[torch.Tensor] = None
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None

    def _find_gate(self) -> nn.Linear:
        try:
            return self.model.model.layers[self.layer_idx].block_sparse_moe.gate
        except AttributeError as e:
            raise ValueError(f"Could not find Mixtral gate at layer {self.layer_idx}") from e

    def _ensure_materialized(self) -> None:
        """Clone gate weights on first use. If still on meta, materialize from state_dict."""
        if self._original is not None:
            return
        w = self._gate.weight.data
        if w.is_meta:
            # Lazy loading: gate was never materialized. Load from state_dict (triggers load).
            key = f"model.layers.{self.layer_idx}.block_sparse_moe.gate.weight"
            state = self.model.state_dict()
            if key not in state:
                for k in state:
                    if f"layers.{self.layer_idx}" in k and "block_sparse_moe" in k and "gate" in k:
                        key = k
                        break
                else:
                    raise RuntimeError(
                        f"Gate weights on meta and key {key!r} not found in state_dict. "
                        "Cannot materialize."
                    )
            loaded = state[key]
            if loaded.is_meta:
                raise RuntimeError(
                    "Gate weights still on meta after state_dict(). "
                    "Try loading the model with low_cpu_mem_usage=False."
                )
            # Put onto same device as the rest of the model
            target_device = next(self.model.parameters()).device
            self._gate.weight.data = loaded.clone().to(target_device, dtype=loaded.dtype)
            w = self._gate.weight.data
        self._original = w.clone().cpu()
        self._device = w.device
        self._dtype = w.dtype

    @property
    def original_weights(self) -> torch.Tensor:
        """Copy of the original gate weights [num_experts, dim]."""
        self._ensure_materialized()
        return self._original.clone()

    def apply_weights(self, new_weights: torch.Tensor) -> None:
        """Apply new gate weights (shape must match)."""
        self._ensure_materialized()
        if new_weights.shape != self._original.shape:
            raise ValueError(
                f"Shape mismatch: {new_weights.shape} vs {self._original.shape}"
            )
        self._gate.weight.data = new_weights.to(self._device, dtype=self._dtype)

    def restore(self) -> None:
        """Restore original weights."""
        if self._original is None:
            return
        logger.debug("Restoring original router weights.")
        self._gate.weight.data = self._original.to(self._device, dtype=self._dtype)

    def __enter__(self) -> "RouterManager":
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[object]) -> None:
        self.restore()
