from pathlib import Path

import torch
from transformers import OlmoeForCausalLM

from gated_router import GatedRouter
from svd_init import default_svd_cache_root, svd_gate_init


def _expert_gate_proj_weights(experts_mod, num_experts: int, intermediate_size: int | None):
    """
    Current HF Olmoe: nn.ModuleList of OlmoeMLP, each with .gate_proj.weight.
    Older layout: stacked gate_up_proj [E, 2*inter, hidden]; gate slice is [:inter].
    """
    try:
        e0 = experts_mod[0]
    except (TypeError, IndexError, AttributeError):
        e0 = None
    if e0 is not None and hasattr(e0, "gate_proj"):
        return [experts_mod[j].gate_proj.weight for j in range(num_experts)]
    if hasattr(experts_mod, "gate_up_proj") and intermediate_size is not None:
        w = experts_mod.gate_up_proj
        return [w[j, :intermediate_size, :].detach() for j in range(num_experts)]
    raise TypeError(
        "Cannot extract expert gate projections for SVD init: "
        f"experts type={type(experts_mod).__name__}"
    )


def replace_routers(model: OlmoeForCausalLM, condition: str, cfg=None) -> OlmoeForCausalLM:
    if condition == "baseline":
        return model

    hidden_size = model.config.hidden_size
    num_experts = model.config.num_experts

    skip          = cfg.router.svd_skip if cfg is not None else 0
    k             = cfg.router.svd_k if cfg is not None else 8
    router_layers = cfg.router.layers    if cfg is not None else "all"

    layer_set = None if router_layers == "all" else set(router_layers)
    n_layers_total = len(model.model.layers)
    if isinstance(router_layers, list) and len(router_layers) == 0:
        raise ValueError("router.layers is [] — no layers would get GatedRouter; use 'all' or indices.")
    n_swap = sum(
        1 for i in range(n_layers_total) if layer_set is None or i in layer_set
    )
    if n_swap == 0:
        raise ValueError(
            f"router.layers={router_layers!r} matched no layer in [0..{n_layers_total - 1}]; "
            "training would stay on the original nn.Linear gate everywhere."
        )
    svd_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    svd_use_cache = getattr(cfg.router, "svd_cache", True) if cfg is not None else True
    raw_cache_dir = getattr(cfg.router, "svd_cache_dir", None) if cfg is not None else None
    svd_cache_dir = raw_cache_dir if raw_cache_dir else None
    model_id = getattr(cfg.model, "model_id", "") if cfg is not None else ""
    svd_gate_alpha = float(getattr(cfg.router, "svd_gate_alpha", 0.01)) if cfg is not None else 0.01
    svd_residual_mean = bool(getattr(cfg.router, "svd_residual_mean", False)) if cfg is not None else False

    if condition == "gated_random":
        print(
            f"[{condition}] replacing gate on {n_swap}/{n_layers_total} layer(s) …",
            flush=True,
        )
    elif condition == "gated_svd":
        cache_root = (
            Path(svd_cache_dir).expanduser() if svd_cache_dir else default_svd_cache_root()
        )
        c_on = svd_use_cache and bool(model_id)
        res_note = "  svd_residual_mean=ON" if svd_residual_mean else ""
        print(
            f"[{condition}] W_gate from SVD  layers={n_swap}/{n_layers_total}  "
            f"device={svd_device}  skip={skip}  k={k}  alpha={svd_gate_alpha}{res_note}  "
            f"cache={'on → ' + str(cache_root) if c_on else 'off'}",
            flush=True,
        )

    inter = getattr(model.config, "intermediate_size", None)

    for layer_idx, layer in enumerate(model.model.layers):
        if layer_set is not None and layer_idx not in layer_set:
            continue
        moe = layer.mlp
        old_router = moe.gate

        new_router = GatedRouter(hidden_size, num_experts)
        dt = old_router.weight.dtype
        dev = old_router.weight.device
        new_router.to(device=dev, dtype=dt)
        with torch.no_grad():
            new_router.W_score.weight.copy_(old_router.weight)

        if condition == "gated_svd":
            gate_proj_weights = _expert_gate_proj_weights(moe.experts, num_experts, inter)
            W_gate_init = svd_gate_alpha * svd_gate_init(
                gate_proj_weights,
                skip=skip,
                k=k,
                svd_device=svd_device,
                use_cache=svd_use_cache,
                cache_dir=svd_cache_dir,
                model_id=model_id,
                layer_idx=layer_idx,
                residual_mean=svd_residual_mean,
            )
            with torch.no_grad():
                new_router.W_gate.weight.copy_(W_gate_init.to(dtype=dt, device=dev))

        moe.gate = new_router

    return model
