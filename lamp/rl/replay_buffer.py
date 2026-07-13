"""Replay buffer for multi-agent off-policy reinforcement learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class TransitionBatch:
    """Mini-batch sampled from :class:`MultiAgentReplayBuffer`."""

    global_obs: np.ndarray
    private_obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_global_obs: np.ndarray
    next_private_obs: np.ndarray
    language_obs: np.ndarray
    next_language_obs: np.ndarray
    dones: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "global_obs": self.global_obs,
            "private_obs": self.private_obs,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_global_obs": self.next_global_obs,
            "next_private_obs": self.next_private_obs,
            "language_obs": self.language_obs,
            "next_language_obs": self.next_language_obs,
            "dones": self.dones,
        }


class MultiAgentReplayBuffer:
    """Fixed-size replay buffer for CTDE-style multi-agent algorithms.

    The buffer stores one transition per environment step. Global observations
    are shared, while private observations, actions, and rewards are stored for
    every household agent.
    """

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        global_obs_dim: int,
        private_obs_dim: int,
        action_dim: int,
        language_dim: int = 0,
        seed: int | None = None,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if n_agents <= 0:
            raise ValueError("n_agents must be positive")
        self.capacity = int(capacity)
        self.n_agents = int(n_agents)
        self.global_obs_dim = int(global_obs_dim)
        self.private_obs_dim = int(private_obs_dim)
        self.action_dim = int(action_dim)
        self.language_dim = int(language_dim)
        if self.language_dim < 0:
            raise ValueError("language_dim must be non-negative")
        self._rng = np.random.default_rng(seed)
        self._position = 0
        self._size = 0

        self._global_obs = np.zeros((capacity, global_obs_dim), dtype=np.float32)
        self._private_obs = np.zeros(
            (capacity, n_agents, private_obs_dim), dtype=np.float32
        )
        self._actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity, n_agents), dtype=np.float32)
        self._next_global_obs = np.zeros((capacity, global_obs_dim), dtype=np.float32)
        self._next_private_obs = np.zeros(
            (capacity, n_agents, private_obs_dim), dtype=np.float32
        )
        self._language_obs = np.zeros(
            (capacity, n_agents, self.language_dim), dtype=np.float32
        )
        self._next_language_obs = np.zeros(
            (capacity, n_agents, self.language_dim), dtype=np.float32
        )
        self._dones = np.zeros((capacity,), dtype=np.float32)

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    def can_sample(self, batch_size: int) -> bool:
        return self._size >= batch_size

    def add(
        self,
        global_obs: np.ndarray,
        private_obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_global_obs: np.ndarray,
        next_private_obs: np.ndarray,
        done: bool,
        language_obs: np.ndarray | None = None,
        next_language_obs: np.ndarray | None = None,
    ) -> None:
        """Add a transition to the buffer."""

        idx = self._position
        self._global_obs[idx] = self._check_shape(
            global_obs, (self.global_obs_dim,), "global_obs"
        )
        self._private_obs[idx] = self._check_shape(
            private_obs, (self.n_agents, self.private_obs_dim), "private_obs"
        )
        self._actions[idx] = self._check_shape(
            actions, (self.n_agents, self.action_dim), "actions"
        )
        self._rewards[idx] = self._check_shape(rewards, (self.n_agents,), "rewards")
        self._next_global_obs[idx] = self._check_shape(
            next_global_obs, (self.global_obs_dim,), "next_global_obs"
        )
        self._next_private_obs[idx] = self._check_shape(
            next_private_obs,
            (self.n_agents, self.private_obs_dim),
            "next_private_obs",
        )
        self._language_obs[idx] = self._checked_language(
            language_obs, "language_obs"
        )
        self._next_language_obs[idx] = self._checked_language(
            next_language_obs, "next_language_obs"
        )
        self._dones[idx] = float(done)

        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_from_observations(
        self,
        observation: Dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observation: Dict[str, np.ndarray],
        done: bool,
        language_obs: np.ndarray | None = None,
        next_language_obs: np.ndarray | None = None,
    ) -> None:
        """Add a transition using environment observation dictionaries."""

        self.add(
            global_obs=observation["global"],
            private_obs=observation["private"],
            actions=actions,
            rewards=rewards,
            next_global_obs=next_observation["global"],
            next_private_obs=next_observation["private"],
            done=done,
            language_obs=language_obs,
            next_language_obs=next_language_obs,
        )

    def sample(self, batch_size: int) -> TransitionBatch:
        """Sample a random mini-batch."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.can_sample(batch_size):
            raise ValueError(
                f"cannot sample batch_size={batch_size}; buffer only has {self._size}"
            )
        indices = self._rng.choice(self._size, size=batch_size, replace=False)
        return TransitionBatch(
            global_obs=self._global_obs[indices].copy(),
            private_obs=self._private_obs[indices].copy(),
            actions=self._actions[indices].copy(),
            rewards=self._rewards[indices].copy(),
            next_global_obs=self._next_global_obs[indices].copy(),
            next_private_obs=self._next_private_obs[indices].copy(),
            language_obs=self._language_obs[indices].copy(),
            next_language_obs=self._next_language_obs[indices].copy(),
            dones=self._dones[indices].copy(),
        )

    def clear(self) -> None:
        """Remove all stored transitions while keeping allocated arrays."""

        self._position = 0
        self._size = 0

    @staticmethod
    def _check_shape(array: np.ndarray, expected_shape: tuple[int, ...], name: str) -> np.ndarray:
        value = np.asarray(array, dtype=np.float32)
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {value.shape}")
        return value

    def _checked_language(self, array: np.ndarray | None, name: str) -> np.ndarray:
        expected_shape = (self.n_agents, self.language_dim)
        if self.language_dim == 0:
            if array is None:
                return np.zeros(expected_shape, dtype=np.float32)
            return self._check_shape(array, expected_shape, name)
        if array is None:
            raise ValueError(f"{name} is required when language_dim > 0")
        return self._check_shape(array, expected_shape, name)
