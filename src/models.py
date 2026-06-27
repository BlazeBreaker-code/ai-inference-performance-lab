from __future__ import annotations

import torch
from torch import nn


class SyntheticInferenceModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        current_dim = input_dim

        for _ in range(depth):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def create_model(input_dim: int, hidden_dim: int, output_dim: int, depth: int) -> nn.Module:
    return SyntheticInferenceModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        depth=depth,
    )
