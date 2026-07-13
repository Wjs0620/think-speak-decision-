"""Neural networks used by the numeric MADDPG baseline."""

from __future__ import annotations

import torch
from torch import nn


def mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    """Build a small MLP with ReLU activations."""

    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.ReLU())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Deterministic household policy.

    The actor receives one household's local numeric observation and outputs
    ``[savings_rate, labor_supply]`` in ``[0, 1]``.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ):
        super().__init__()
        self.net = mlp(obs_dim, hidden_dims, action_dim)
        self.output = nn.Sigmoid()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.output(self.net(obs))


class Critic(nn.Module):
    """Centralized critic for one household agent."""

    def __init__(
        self,
        state_dim: int,
        joint_action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        self.net = mlp(state_dim + joint_action_dim, hidden_dims, 1)

    def forward(self, state: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, joint_actions], dim=-1)
        return self.net(x)

