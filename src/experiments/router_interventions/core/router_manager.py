"""Temporarily modify and restore Mixtral router (gate) weights."""
from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _dequantize_int8(weight_param: Any) -> torch.Tensor:
    """Dequantize bitsandbytes Int8Params → float.  weight_param is gate.weight."""
    w = weight_param.data
    if w.dtype != torch.int8:
        return w.float()
    scb = getattr(weight_param, "SCB", None)
    if scb is None:
        logger.warning("int8 weight without SCB; raw conversion (may be inaccurate)")
        return w.float() / 127.0
    scale = scb.unsqueeze(1) if scb.dim() == 1 else scb
    return (w.float() * scale.to(w.device)) / 127.0


class RouterManager:
    """Locate, swap, and restore the gate (router) weights for one MoE layer.

    For quantized (int8) gates the original is dequantized to float and replaced
    with an ``nn.Linear`` so that baseline and interventions both run in float and
    deltas are comparable.
    """

    def __init__(self, model: nn.Module, layer_idx: int) -> None:
        self.model = model
        self.layer_idx = layer_idx
        self._gate = self._find_gate()
        self._original: Optional[torch.Tensor] = None
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None
        self._quantized = False
        self._replacement: Optional[nn.Linear] = None
        self._moe_block: Optional[Any] = None

    # -- init helpers --------------------------------------------------------

    def _find_gate(self) -> nn.Module:
        try:
            return self.model.model.layers[self.layer_idx].block_sparse_moe.gate
        except AttributeError as exc:
            raise ValueError(f"No Mixtral gate at layer {self.layer_idx}") from exc

    def _materialize(self) -> None:
        """Ensure gate weights are on a real device (not meta) and cache original."""
        if self._original is not None:
            return
        w = self._gate.weight.data
        if w.is_meta:
            w = self._resolve_meta()
        self._quantized = (
            getattr(self._gate.weight, "data", w).dtype == torch.int8
            or getattr(self._gate.weight, "SCB", None) is not None
        )
        if self._quantized:
            self._init_quantized(w)
        else:
            self._original = w.clone().cpu()
            self._device = w.device
            self._dtype = w.dtype

    def _resolve_meta(self) -> torch.Tensor:
        """Load gate weights from state_dict when they are on meta device."""
        key = f"model.layers.{self.layer_idx}.block_sparse_moe.gate.weight"
        
        # Try to load from state_dict (may trigger loading from disk)
        try:
            state = self.model.state_dict()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load state_dict for layer {self.layer_idx}. "
                f"This may mean model weights aren't fully downloaded. "
                f"Try running with fewer layers or ensure the model is cached locally. Error: {e}"
            ) from e
        
        if key not in state:
            for k in state:
                if f"layers.{self.layer_idx}" in k and "gate" in k:
                    key = k
                    break
            else:
                raise RuntimeError(f"Gate key {key!r} not in state_dict")
        
        loaded = state[key]
        if loaded.is_meta:
            raise RuntimeError(
                f"Gate for layer {self.layer_idx} still on meta device after state_dict(). "
                f"This likely means:\n"
                f"  1. Model weights aren't fully downloaded from HuggingFace\n"
                f"  2. Insufficient GPU/CPU memory to load this layer\n"
                f"  3. Try: running with quantization (--quantization 8bit), "
                f"using fewer layers, or pre-downloading the model"
            )
        device = next((p.device for p in self.model.parameters() if not p.is_meta), torch.device("cpu"))
        self._gate.weight.data = loaded.clone().to(device, dtype=loaded.dtype)
        return self._gate.weight.data

    def _init_quantized(self, w: torch.Tensor) -> None:
        """Dequantize int8 gate once → float ``nn.Linear`` for the entire run."""
        w_float = _dequantize_int8(self._gate.weight)
        self._original = w_float.cpu().float()
        self._device = w_float.device
        self._dtype = torch.float32
        out_f, in_f = self._original.shape
        compute_dtype = next(self.model.parameters()).dtype
        self._replacement = nn.Linear(in_f, out_f, bias=False,
                                      device=self._device, dtype=compute_dtype)
        self._replacement.weight.data = self._original.to(self._device, dtype=compute_dtype)
        block = self.model.model.layers[self.layer_idx].block_sparse_moe
        self._moe_block = block
        block.gate = self._replacement
        logger.info("Gate quantized (int8) → dequantized float for comparable deltas")

    # -- public API ----------------------------------------------------------

    @property
    def original_weights(self) -> torch.Tensor:
        """Clone of original gate weights [num_experts, dim]."""
        self._materialize()
        return self._original.clone()

    def apply_weights(self, weights: torch.Tensor) -> None:
        """Set new gate weights (must match original shape)."""
        self._materialize()
        if weights.shape != self._original.shape:
            raise ValueError(f"Shape mismatch: {weights.shape} vs {self._original.shape}")
        if self._quantized:
            dtype = self._replacement.weight.dtype
            self._replacement.weight.data = weights.to(self._device, dtype=dtype)
        else:
            self._gate.weight.data = weights.to(self._device, dtype=self._dtype)

    def restore(self) -> None:
        """Restore original weights / gate module."""
        if self._original is None:
            return
        if self._quantized and self._moe_block is not None:
            self._moe_block.gate = self._gate
            self._moe_block = None
        elif not self._quantized:
            self._gate.weight.data = self._original.to(self._device, dtype=self._dtype)

    def __enter__(self) -> RouterManager:
        return self

    def __exit__(self, *exc) -> None:
        self.restore()
