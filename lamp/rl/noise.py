"""Exploration noise utilities."""

from __future__ import annotations

import numpy as np


class GaussianNoise:
    """Gaussian action noise with optional linear decay."""

    def __init__(
        self,
        action_dim: int,
        sigma: float = 0.15,
        min_sigma: float = 0.03,
        decay: float = 0.9995,
        seed: int | None = None,
    ):
        self.action_dim = action_dim
        self.sigma = sigma
        self.min_sigma = min_sigma
        self.decay = decay
        self._rng = np.random.default_rng(seed)

    def sample(self, shape: tuple[int, ...]) -> np.ndarray:
        noise = self._rng.normal(0.0, self.sigma, size=shape)
        return noise.astype(np.float32)

    def step(self) -> None:
        self.sigma = max(self.min_sigma, self.sigma * self.decay)

