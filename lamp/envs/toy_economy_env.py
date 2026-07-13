"""A small TaxAI-like economy for developing the LAMP pipeline.

This environment is intentionally simple. It is not meant to reproduce TaxAI
results; it provides the observation and action structure needed to test LAMP:

- global numerical observation shared by all households
- private household observation for each agent
- continuous household actions: savings rate and labor supply
- rewards based on consumption utility minus labor disutility
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .base_env import EconomyStep, Observation


@dataclass
class ToyEconomyConfig:
    """Configuration for :class:`ToyEconomyEnv`."""

    n_agents: int = 8
    max_steps: int = 200
    initial_wage: float = 1.0
    initial_asset_mean: float = 1.0
    initial_asset_std: float = 0.15
    initial_efficiency_mean: float = 1.0
    initial_efficiency_std: float = 0.10
    interest_rate: float = 0.01
    depreciation: float = 0.005
    base_tax_rate: float = 0.10
    progressive_tax_rate: float = 0.15
    government_spending_share: float = 0.18
    eta: float = 1.5
    gamma: float = 1.0
    h_max: float = 1.0
    shock_std: float = 0.015
    collapse_welfare_threshold: float = -100.0


class ToyEconomyEnv:
    """Minimal multi-household economy with TaxAI-like signals."""

    global_obs_names = (
        "wage",
        "avg_asset",
        "avg_income",
        "avg_efficiency",
        "gdp",
        "social_welfare",
        "wealth_gini",
    )
    private_obs_names = (
        "asset",
        "efficiency",
        "income",
        "previous_savings_rate",
        "previous_labor_supply",
    )
    action_names = ("savings_rate", "labor_supply")

    def __init__(self, config: ToyEconomyConfig | None = None):
        self.config = config or ToyEconomyConfig()
        self.n_agents = self.config.n_agents
        self.global_obs_dim = len(self.global_obs_names)
        self.private_obs_dim = len(self.private_obs_names)
        self.action_dim = len(self.action_names)
        self._rng = np.random.default_rng()
        self._step = 0
        self._wage = self.config.initial_wage
        self._assets = np.zeros(self.n_agents, dtype=np.float32)
        self._efficiency = np.ones(self.n_agents, dtype=np.float32)
        self._income = np.zeros(self.n_agents, dtype=np.float32)
        self._prev_actions = np.zeros((self.n_agents, self.action_dim), dtype=np.float32)
        self._last_rewards = np.zeros(self.n_agents, dtype=np.float32)

    @property
    def step_count(self) -> int:
        return self._step

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step = 0
        cfg = self.config
        self._wage = cfg.initial_wage
        self._assets = self._rng.normal(
            cfg.initial_asset_mean, cfg.initial_asset_std, size=self.n_agents
        ).astype(np.float32)
        self._assets = np.clip(self._assets, 0.05, None)
        self._efficiency = self._rng.normal(
            cfg.initial_efficiency_mean, cfg.initial_efficiency_std, size=self.n_agents
        ).astype(np.float32)
        self._efficiency = np.clip(self._efficiency, 0.20, None)
        self._income = np.zeros(self.n_agents, dtype=np.float32)
        self._prev_actions = np.zeros((self.n_agents, self.action_dim), dtype=np.float32)
        self._last_rewards = np.zeros(self.n_agents, dtype=np.float32)
        return self._make_observation()

    def step(self, actions: np.ndarray) -> EconomyStep:
        actions = self._validate_actions(actions)
        savings_rate = actions[:, 0]
        labor_supply = actions[:, 1]

        income = self._wage * self._efficiency * labor_supply
        taxes = self._compute_taxes(income)
        disposable_income = np.maximum(income - taxes, 0.0)
        consumption = np.maximum((1.0 - savings_rate) * disposable_income, 1e-6)
        savings = savings_rate * disposable_income

        rewards = self._utility(consumption, labor_supply).astype(np.float32)
        public_spending = self.config.government_spending_share * float(np.sum(income))
        asset_growth = savings + self.config.interest_rate * self._assets
        self._assets = np.maximum(
            (1.0 - self.config.depreciation) * self._assets + asset_growth,
            1e-6,
        ).astype(np.float32)

        self._income = income.astype(np.float32)
        self._prev_actions = actions.astype(np.float32)
        self._last_rewards = rewards
        self._step += 1
        self._update_macro_state(public_spending)

        obs = self._make_observation()
        done = self._is_done(obs)
        info = self._make_info(consumption, taxes, public_spending)
        return EconomyStep(observation=obs, rewards=rewards, done=done, info=info)

    def sample_random_actions(self) -> np.ndarray:
        """Sample valid random household actions."""

        savings = self._rng.uniform(0.0, 1.0, size=(self.n_agents, 1))
        labor = self._rng.uniform(0.0, self.config.h_max, size=(self.n_agents, 1))
        return np.concatenate([savings, labor], axis=1).astype(np.float32)

    def _validate_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        expected_shape = (self.n_agents, self.action_dim)
        if actions.shape != expected_shape:
            raise ValueError(f"actions must have shape {expected_shape}, got {actions.shape}")
        clipped = actions.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0.0, 1.0)
        clipped[:, 1] = np.clip(clipped[:, 1], 0.0, self.config.h_max)
        return clipped

    def _compute_taxes(self, income: np.ndarray) -> np.ndarray:
        avg_income = float(np.mean(income)) + 1e-6
        relative_income = income / avg_income
        tax_rate = self.config.base_tax_rate + self.config.progressive_tax_rate * (
            relative_income / (1.0 + relative_income)
        )
        return np.clip(tax_rate, 0.0, 0.70) * income

    def _utility(self, consumption: np.ndarray, labor_supply: np.ndarray) -> np.ndarray:
        eta = self.config.eta
        gamma = self.config.gamma
        if abs(eta - 1.0) < 1e-8:
            consumption_utility = np.log(consumption)
        else:
            consumption_utility = (np.power(consumption, 1.0 - eta) - 1.0) / (1.0 - eta)
        labor_disutility = np.power(labor_supply, 1.0 + gamma) / (1.0 + gamma)
        return consumption_utility - labor_disutility

    def _update_macro_state(self, public_spending: float) -> None:
        gdp = float(np.sum(self._income))
        avg_efficiency = float(np.mean(self._efficiency))
        productivity_term = 0.01 * (avg_efficiency - 1.0)
        fiscal_term = 0.001 * public_spending / max(self.n_agents, 1)
        shock = float(self._rng.normal(0.0, self.config.shock_std))
        self._wage = max(0.05, self._wage * (1.0 + productivity_term + fiscal_term + shock))

        efficiency_shock = self._rng.normal(0.0, 0.01, size=self.n_agents)
        self._efficiency = np.maximum(
            0.20,
            0.98 * self._efficiency + 0.02 * avg_efficiency + efficiency_shock,
        ).astype(np.float32)

        if gdp <= 1e-8:
            self._wage *= 0.995

    def _make_observation(self) -> Observation:
        social_welfare = float(np.sum(self._last_rewards))
        global_obs = np.array(
            [
                self._wage,
                float(np.mean(self._assets)),
                float(np.mean(self._income)),
                float(np.mean(self._efficiency)),
                float(np.sum(self._income)),
                social_welfare,
                self._gini(self._assets),
            ],
            dtype=np.float32,
        )
        private_obs = np.column_stack(
            [
                self._assets,
                self._efficiency,
                self._income,
                self._prev_actions[:, 0],
                self._prev_actions[:, 1],
            ]
        ).astype(np.float32)
        return {
            "global": global_obs,
            "private": private_obs,
            "step": self._step,
            "global_names": self.global_obs_names,
            "private_names": self.private_obs_names,
        }

    def _make_info(
        self, consumption: np.ndarray, taxes: np.ndarray, public_spending: float
    ) -> Dict[str, Any]:
        return {
            "gdp": float(np.sum(self._income)),
            "social_welfare": float(np.sum(self._last_rewards)),
            "total_consumption": float(np.sum(consumption)),
            "total_labor": float(np.sum(self._prev_actions[:, 1])),
            "total_tax": float(np.sum(taxes)),
            "public_spending": public_spending,
            "wealth_gini": self._gini(self._assets),
            "wage": float(self._wage),
        }

    def _is_done(self, obs: Observation) -> bool:
        if self._step >= self.config.max_steps:
            return True
        social_welfare = float(obs["global"][self.global_obs_names.index("social_welfare")])
        return social_welfare < self.config.collapse_welfare_threshold

    @staticmethod
    def _gini(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return 0.0
        sorted_values = np.sort(np.maximum(values, 0.0))
        total = float(np.sum(sorted_values))
        if total <= 1e-12:
            return 0.0
        n = sorted_values.size
        index = np.arange(1, n + 1)
        return float((2.0 * np.sum(index * sorted_values) / (n * total)) - ((n + 1.0) / n))

