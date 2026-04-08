import json
from dataclasses import dataclass
from typing import Optional

import humanize
from huggingface_hub import (
    get_hf_file_metadata,
    hf_hub_download,
    hf_hub_url,
)
from safetensors import safe_open
from torch import Tensor


@dataclass
class TensorMetadata:
    model_id: str
    tensor_name: str
    hf_filename: str
    local_path: Optional[str] = None

    @property
    def hf_url(self) -> str:
        return hf_hub_url(repo_id=self.model_id, filename=self.hf_filename)

    def size(self, human_readable: bool = False) -> int | str:
        """Get the size of the tensor file.
        
        Args:
            human_readable: If True, return human-readable string (e.g., "1.5 MB")
            
        Returns:
            File size in bytes (int) or human-readable string (str)
        """
        metadata = get_hf_file_metadata(self.hf_url)
        if human_readable:
            return humanize.naturalsize(metadata.size)
        return metadata.size
    
    def download_file(self) -> str:
        """Download the file from Hugging Face Hub with optimizations.
        
        Uses optimizations for faster downloads:
        - Enables hf_transfer if available (via HF_HUB_ENABLE_HF_TRANSFER env var)
        - resume_download=True to resume interrupted downloads
        
        Returns:
            Local path to the downloaded file
        """
        import os
        
        # Enable hf_transfer for faster downloads (if available)
        # This can provide 10-100x speedup for large files
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        
        self.local_path = hf_hub_download(
            repo_id=self.model_id,
            filename=self.hf_filename,
            # Note: resume_download is deprecated, downloads always resume by default
        )
        return self.local_path
    
    def load(self) -> Tensor:
        """Load the tensor from the local file.
        
        Returns:
            PyTorch tensor
            
        Raises:
            ValueError: If file has not been downloaded yet
        """
        if self.local_path is None:
            raise ValueError("Tensor file not downloaded yet. Call `download_file()` first.")
        
        with safe_open(self.local_path, framework="pt") as f:
            tensor = f.get_tensor(self.tensor_name)
        
        return tensor


class BaseMoE:
    """Base class for Mixture of Experts models.
    
    Subclasses should define:
    - model_id: Hugging Face model identifier
    - n_experts: Number of experts in the MoE layer
    - n_layers: Total number of layers in the model
    - expert_tensor_name_template: Format string for expert tensor names
    - router_tensor_name_template: Format string for router tensor names
    """
    
    model_id: str
    n_experts: int
    n_layers: int
    expert_tensor_name_template: str
    router_tensor_name_template: str
    _weight_map: Optional[dict[str, str]] = None


    @property
    def weight_map(self) -> dict[str, str]:
        """Returns the mapping of tensor names to Hugging Face filenames.
        
        Returns:
            Dictionary mapping tensor names to their shard filenames
        """
        if self._weight_map is None:
            import os
            
            # Enable hf_transfer for faster downloads (if available)
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
            
            local_index = hf_hub_download(
                repo_id=self.model_id,
                filename="model.safetensors.index.json",
                # Note: resume_download is deprecated, downloads always resume by default
            )

            with open(local_index, "r") as f:
                index = json.load(f)

            self._weight_map = index["weight_map"]

        return self._weight_map

    def get_experts_metadata(self, layer: int) -> list[TensorMetadata]:
        """Get metadata for all expert weight tensors in a layer.
        
        Args:
            layer: Layer number
            
        Returns:
            List of TensorMetadata objects for all experts
        """
        experts = []

        for i in range(self.n_experts):
            tensor_name = self.expert_tensor_name_template.format(layer=layer, expert=i)
            hf_filename = self.weight_map[tensor_name]

            experts.append(
                TensorMetadata(
                    model_id=self.model_id,
                    tensor_name=tensor_name,
                    hf_filename=hf_filename,
                )
            )
        return experts

    def get_router_metadata(self, layer: int) -> TensorMetadata:
        """Get metadata for the router (gate) weight tensor in a layer.
        
        Args:
            layer: Layer number
            
        Returns:
            TensorMetadata object for the router tensor
        """
        tensor_name = self.router_tensor_name_template.format(layer=layer)
        hf_filename = self.weight_map[tensor_name]

        return TensorMetadata(
            model_id=self.model_id,
            tensor_name=tensor_name,
            hf_filename=hf_filename,
        )

class Mixtral8x7B(BaseMoE):
    model_id = "mistralai/Mixtral-8x7B-v0.1"
    n_experts = 8
    n_layers = 32

    expert_tensor_name_template = "model.layers.{layer}.block_sparse_moe.experts.{expert}.w1.weight"
    router_tensor_name_template = "model.layers.{layer}.block_sparse_moe.gate.weight"

