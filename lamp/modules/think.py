"""Think module for LAMP.

Think translates numeric economic observations into language signals:

- long-term news at fixed checkpoints
- short-term news when macro indicators shift sharply
- private household reasoning and coarse economic status
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from lamp.llm import BaseLLM
from lamp.llm.prompts import NEWS_SYSTEM_PROMPT, REASONING_SYSTEM_PROMPT


NEWS_LONG = "long"
NEWS_SHORT = "short"
NEWS_NONE = "none"


@dataclass
class ThinkConfig:
    """Configuration for the Think module."""

    shock_threshold: float = 0.15
    long_term_interval: int = 10
    history_window: int = 10
    shock_indicator_names: tuple[str, ...] = (
        "wealth_gini",
        "social_welfare",
        "gdp",
    )


@dataclass(frozen=True)
class AgentReasoning:
    """Private reasoning generated for one household."""

    agent_id: int
    text: str
    economic_status: int
    economic_status_label: str


@dataclass(frozen=True)
class ThinkOutput:
    """Output produced by one Think step."""

    news_type: str
    news: str
    agent_reasonings: list[AgentReasoning]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def reasoning_texts(self) -> list[str]:
        return [item.text for item in self.agent_reasonings]

    @property
    def economic_statuses(self) -> np.ndarray:
        return np.asarray([item.economic_status for item in self.agent_reasonings], dtype=np.int64)


class ThinkModule:
    """Generate language news and household reasoning from numeric observations."""

    def __init__(self, llm: BaseLLM, config: ThinkConfig | None = None):
        self.llm = llm
        self.config = config or ThinkConfig()
        self._global_history: list[np.ndarray] = []
        self._latest_long_news = ""

    def reset(self) -> None:
        self._global_history.clear()
        self._latest_long_news = ""

    def step(self, observation: Mapping[str, Any]) -> ThinkOutput:
        """Run one Think step from an environment observation dictionary."""

        global_obs = np.asarray(observation["global"], dtype=np.float32)
        private_obs = np.asarray(observation["private"], dtype=np.float32)
        global_names = tuple(observation.get("global_names", ()))
        private_names = tuple(observation.get("private_names", ()))
        step_id = int(observation.get("step", len(self._global_history)))

        news_type, trigger_metadata = self._determine_news_type(
            step_id=step_id,
            global_obs=global_obs,
            global_names=global_names,
        )
        self._global_history.append(global_obs.copy())
        self._trim_history()

        news = self._generate_news(
            news_type=news_type,
            step_id=step_id,
            global_obs=global_obs,
            global_names=global_names,
        )
        if news_type == NEWS_LONG:
            self._latest_long_news = news

        agent_reasonings = [
            self._generate_agent_reasoning(
                agent_id=agent_id,
                news=news,
                global_obs=global_obs,
                private_obs=private_obs[agent_id],
                global_names=global_names,
                private_names=private_names,
            )
            for agent_id in range(private_obs.shape[0])
        ]

        return ThinkOutput(
            news_type=news_type,
            news=news,
            agent_reasonings=agent_reasonings,
            metadata={
                "step": step_id,
                "latest_long_news": self._latest_long_news,
                **trigger_metadata,
            },
        )

    def _determine_news_type(
        self,
        step_id: int,
        global_obs: np.ndarray,
        global_names: Sequence[str],
    ) -> tuple[str, dict[str, Any]]:
        if self.config.long_term_interval > 0 and step_id > 0:
            if step_id % self.config.long_term_interval == 0:
                return NEWS_LONG, {"trigger": "long_term_checkpoint"}

        if self._global_history:
            previous = self._global_history[-1]
            max_change = 0.0
            changed_indicator = ""
            for name in self.config.shock_indicator_names:
                if name not in global_names:
                    continue
                idx = global_names.index(name)
                change = self._relative_change(float(previous[idx]), float(global_obs[idx]))
                if change > max_change:
                    max_change = change
                    changed_indicator = name
            if max_change > self.config.shock_threshold:
                return NEWS_SHORT, {
                    "trigger": "short_term_shock",
                    "changed_indicator": changed_indicator,
                    "max_relative_change": max_change,
                }

        return NEWS_NONE, {"trigger": "none"}

    def _generate_news(
        self,
        news_type: str,
        step_id: int,
        global_obs: np.ndarray,
        global_names: Sequence[str],
    ) -> str:
        if news_type == NEWS_NONE:
            return ""
        current_summary = self._format_named_values(global_names, global_obs)
        history_summary = self._history_summary(global_names)
        prompt = (
            f"news_type: {news_type}\n"
            f"step: {step_id}\n"
            f"current_global: {current_summary}\n"
            f"history: {history_summary}\n"
            f"latest_long_news: {self._latest_long_news}"
        )
        return self.llm.generate(
            prompt,
            system_prompt=NEWS_SYSTEM_PROMPT,
            metadata={"task": "news", "news_type": news_type, "step": step_id},
        ).text

    def _generate_agent_reasoning(
        self,
        agent_id: int,
        news: str,
        global_obs: np.ndarray,
        private_obs: np.ndarray,
        global_names: Sequence[str],
        private_names: Sequence[str],
    ) -> AgentReasoning:
        status, label = self._infer_status(global_obs, private_obs, global_names, private_names)
        prompt = (
            f"agent_id: {agent_id}\n"
            f"news: {news or 'No new public news.'}\n"
            f"global: {self._format_named_values(global_names, global_obs)}\n"
            f"private: {self._format_named_values(private_names, private_obs)}\n"
            f"status_hint: {label}"
        )
        text = self.llm.generate(
            prompt,
            system_prompt=REASONING_SYSTEM_PROMPT,
            metadata={"task": "reasoning", "agent_id": agent_id, "status_hint": label},
        ).text
        return AgentReasoning(
            agent_id=agent_id,
            text=text,
            economic_status=status,
            economic_status_label=label,
        )

    def _infer_status(
        self,
        global_obs: np.ndarray,
        private_obs: np.ndarray,
        global_names: Sequence[str],
        private_names: Sequence[str],
    ) -> tuple[int, str]:
        asset = self._value_by_name(private_obs, private_names, "asset", default=0.0)
        income = self._value_by_name(private_obs, private_names, "income", default=0.0)
        efficiency = self._value_by_name(private_obs, private_names, "efficiency", default=1.0)
        avg_asset = self._value_by_name(global_obs, global_names, "avg_asset", default=1.0)
        avg_income = self._value_by_name(global_obs, global_names, "avg_income", default=1.0)

        asset_score = asset / max(abs(avg_asset), 1e-6)
        income_score = income / max(abs(avg_income), 1e-6)
        score = 0.45 * asset_score + 0.35 * income_score + 0.20 * efficiency
        if score < 0.75:
            return 0, "bad"
        if score > 1.20:
            return 2, "good"
        return 1, "neutral"

    def _history_summary(self, global_names: Sequence[str]) -> str:
        if len(self._global_history) < 2:
            return "insufficient history"
        start = self._global_history[0]
        end = self._global_history[-1]
        parts = []
        for name in self.config.shock_indicator_names:
            if name not in global_names:
                continue
            idx = global_names.index(name)
            parts.append(f"{name}_change={end[idx] - start[idx]:.4f}")
        return ", ".join(parts) if parts else "no tracked indicators"

    def _trim_history(self) -> None:
        window = self.config.history_window
        if window > 0 and len(self._global_history) > window:
            self._global_history = self._global_history[-window:]

    @staticmethod
    def _relative_change(previous: float, current: float) -> float:
        denominator = max(abs(previous), 1e-6)
        return abs(current - previous) / denominator

    @staticmethod
    def _format_named_values(names: Sequence[str], values: np.ndarray) -> str:
        if not names:
            return ", ".join(f"x{i}: {float(value):.4f}" for i, value in enumerate(values))
        return ", ".join(
            f"{name}: {float(value):.4f}" for name, value in zip(names, values)
        )

    @staticmethod
    def _value_by_name(
        values: np.ndarray,
        names: Sequence[str],
        name: str,
        *,
        default: float,
    ) -> float:
        if name not in names:
            return default
        return float(values[names.index(name)])

