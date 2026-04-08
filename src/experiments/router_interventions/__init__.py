"""Router interventions: project out expert SVD directions from router rows.

Usage::

    # 1. Build SVD cache from expert w1
    python -m src.experiments.router_interventions.run_svd_from_expert \\
        --cache-dir svd_cache --layer_idx 0 1 2 ... 31

    # 2. Run project-out experiment
    python -m src.experiments.router_interventions.run_project_out \\
        --svd-dir svd_cache --layer-idx {0..31} --top-k 1,2,4,8,16,32,64
"""
from .core import (
    ExperimentConfig,
    BatchLoader,
    LossEvaluator,
    RouterManager,
    ExpertVectors,
    VectorIntervention,
)
from .viz import (
    load_results,
    load_results_dir,
    plot_ablation_results,
    plot_confusion_heatmap,
    plot_confusion_from_results,
    plot_delta_vs_k,
    plot_delta_vs_k_acl,
    plot_delta_vs_layers,
    plot_delta_vs_layers_acl,
    set_acl_style,
)

__all__ = [
    "ExperimentConfig",
    "BatchLoader",
    "LossEvaluator",
    "RouterManager",
    "ExpertVectors",
    "VectorIntervention",
    "load_results",
    "load_results_dir",
    "plot_ablation_results",
    "plot_confusion_heatmap",
    "plot_confusion_from_results",
    "plot_delta_vs_k",
    "plot_delta_vs_k_acl",
    "plot_delta_vs_layers",
    "plot_delta_vs_layers_acl",
    "set_acl_style",
]
