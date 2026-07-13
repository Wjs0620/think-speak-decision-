"""Base interfaces for multi-agent economic environments.

The LAMP code should depend on this small interface instead of a concrete
environment. A future TaxAI wrapper can implement the same reset/step contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol

import numpy as np


Observation = Dict[str, Any]
Info = Dict[str, Any]


@dataclass(frozen=True)
class EconomyStep:
    """Result returned by an environment step."""

    observation: Observation
    rewards: np.ndarray
    done: bool
    info: Info


class BaseMultiAgentEconomyEnv(Protocol):
    """Protocol implemented by economic multi-agent environments."""

    n_agents: int
    global_obs_dim: int
    private_obs_dim: int
    action_dim: int

    def reset(self, seed: int | None = None) -> Observation:
        """Reset the environment and return the initial observation."""

    def step(self, actions: np.ndarray) -> EconomyStep:
        """Advance the environment by one step.

        Args:
            actions: Array with shape ``(n_agents, action_dim)``. For household
                agents, the toy environment uses ``[savings_rate, labor_supply]``.
        """

