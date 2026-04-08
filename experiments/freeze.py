"""
Parameter freezing based on training.trainable config.

Options:
  all               — nothing frozen, train every parameter
  router_only       — freeze everything except moe.gate in each layer
  router_gate_only  — freeze everything except GatedRouter.W_gate (SVD/aux branch);
                      W_score (original router weights) stays frozen. No effect on
                      stock single-Linear gates — use a gated_* condition.
  [0, 1, ...]       — freeze everything except the listed transformer layer indices
                      (each listed layer is fully trained, including its router)
"""

from transformers import OlmoeForCausalLM


def apply_freezing(model: OlmoeForCausalLM, trainable) -> None:
    """
    trainable: the value of cfg.training.trainable
               either "all", "router_only", "router_gate_only", or a list of layer indices
    Modifies model in-place. Prints a summary of trainable param count.
    """
    if trainable == "all":
        return  # nothing to do

    # freeze everything first
    for p in model.parameters():
        p.requires_grad_(False)

    if trainable == "router_only":
        for layer in model.model.layers:
            for p in layer.mlp.gate.parameters():
                p.requires_grad_(True)

    elif trainable == "router_gate_only":
        for layer in model.model.layers:
            gate = layer.mlp.gate
            if hasattr(gate, "W_gate"):
                for p in gate.W_gate.parameters():
                    p.requires_grad_(True)

    elif isinstance(trainable, list):
        layer_set = set(trainable)
        for idx, layer in enumerate(model.model.layers):
            if idx in layer_set:
                for p in layer.parameters():
                    p.requires_grad_(True)

    else:
        raise ValueError(
            f"training.trainable must be 'all', 'router_only', 'router_gate_only', "
            f"or a list of layer indices. Got: {trainable!r}"
        )

    total  = sum(p.numel() for p in model.parameters())
    active = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {active:,} / {total:,}  ({100 * active / total:.1f}%)")

    if trainable == "router_gate_only" and active == 0:
        raise ValueError(
            "training.trainable='router_gate_only' requires GatedRouter gates (W_gate). "
            "Use condition gated_random or gated_svd (and swap at least one MoE layer)."
        )
