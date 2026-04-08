"""Model and tokenizer loading with optional quantization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if TYPE_CHECKING:
    from .config import ExperimentConfig

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load HuggingFace model + tokenizer (float / 8bit / 4bit, device_map="auto")."""

    def __init__(
        self, config: ExperimentConfig, priority_layers: list[int] | None = None, max_gpu_layers: int = 20
    ) -> None:
        self.config = config
        self.priority_layers = priority_layers or []
        self.max_gpu_layers = max_gpu_layers

    def _create_priority_device_map(self, num_layers: int = 32) -> dict | str:
        """Create device map that prioritizes loading specific layers to GPU.
        
        Strategy:
        1. Put priority layers on GPU first (guaranteed)
        2. Fill remaining GPU slots (up to max_gpu_layers) with other layers sequentially
        3. Rest go to CPU
        
        Args:
            num_layers: Total number of transformer layers (default 32 for Mixtral)
        """
        if not self.priority_layers:
            return "auto"

        device_map = {
            "model.embed_tokens": "cpu",  # Keep small components on CPU
            "model.norm": "cpu",
            "lm_head": "cpu",
        }

        # Create ordered list: priority layers first, then fill sequentially
        priority_set = set(self.priority_layers)
        gpu_layers = list(self.priority_layers)  # Start with priority layers
        
        # Fill remaining slots with other layers (0, 1, 2, ...) up to max_gpu_layers
        for i in range(num_layers):
            if i not in priority_set and len(gpu_layers) < self.max_gpu_layers:
                gpu_layers.append(i)
        
        gpu_layers_set = set(gpu_layers)
        
        # Assign devices
        for i in range(num_layers):
            device_map[f"model.layers.{i}"] = "cuda:0" if i in gpu_layers_set else "cpu"

        logger.info(
            "Priority device map: %d layers on GPU (priority: %s, others: %s), %d on CPU",
            len(gpu_layers),
            sorted(self.priority_layers),
            sorted([l for l in gpu_layers if l not in priority_set]),
            num_layers - len(gpu_layers),
        )
        return device_map

    def load_tokenizer(self) -> AutoTokenizer:
        logger.info("Loading tokenizer from %s", self.config.model_id)
        return AutoTokenizer.from_pretrained(self.config.model_id)

    def load_model(self) -> AutoModelForCausalLM:
        kwargs = self._load_kwargs()
        logger.info(
            "Loading model from %s (%s)",
            self.config.model_id,
            self.config.quantization or "float",
        )
        model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **kwargs)

        # Force materialization of any meta tensors
        logger.info("Materializing model parameters...")
        for name, param in model.named_parameters():
            if param.is_meta:
                logger.warning(
                    "Parameter %s is on meta device - this shouldn't happen with device_map='auto'",
                    name,
                )

        model.eval()
        return model

    def _load_kwargs(self) -> dict:
        device_map = self._create_priority_device_map()
        q = self.config.quantization
        if q == "8bit":
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True
                ),
                "device_map": device_map,
            }
        if q == "4bit":
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_enable_fp32_cpu_offload=True,
                ),
                "device_map": device_map,
            }
        return {"torch_dtype": torch.bfloat16, "device_map": device_map}
