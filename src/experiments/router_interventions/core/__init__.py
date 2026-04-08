"""Core building blocks for router intervention experiments."""
from .config import ExperimentConfig
from .data import BatchLoader, TextListBatchLoader, WikitextBatchLoader, WikiTitlesBatchLoader
from .evaluation import (
    LossEvaluator,
    TokenDistributionComparator,
    confusion_matrix_top_k,
    expert_routing_matrix,
    expert_confusion_matrix,
    register_routing_capture,
    routing_captures_to_selections,
)
from .inference import ModelLoader
from .router_manager import RouterManager
from .timer import Timer, timer
from .vectors import ExpertVectors, VectorIntervention

__all__ = [
    "ExperimentConfig",
    "BatchLoader",
    "TextListBatchLoader",
    "WikitextBatchLoader",
    "WikiTitlesBatchLoader",
    "LossEvaluator",
    "TokenDistributionComparator",
    "confusion_matrix_top_k",
    "expert_routing_matrix",
    "expert_confusion_matrix",
    "register_routing_capture",
    "routing_captures_to_selections",
    "ModelLoader",
    "RouterManager",
    "ExpertVectors",
    "VectorIntervention",
    "Timer",
    "timer",
]
