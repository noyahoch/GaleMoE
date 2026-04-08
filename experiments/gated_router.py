import torch
import torch.nn as nn


class GatedRouter(nn.Module):
    """
    Replacement for OlmoeSparseMoeBlock.gate (nn.Linear): returns raw router logits only.

    Newer transformers Olmoe applies softmax + top-k inside OlmoeSparseMoeBlock.forward;
    the gate must return a single (batch*seq, num_experts) tensor.
    """

    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.hidden_dim = hidden_size
        self.W_score = nn.Linear(hidden_size, num_experts, bias=False)
        self.W_gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        return self.W_score(hidden_states) + self.W_gate(hidden_states)
