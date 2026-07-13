"""Deterministic mock LLM for local LAMP reproduction."""

from __future__ import annotations

import hashlib
import re
from typing import Mapping, Sequence

from .base_llm import ChatMessage, LLMGeneration


class MockLLM:
    """Rule-based deterministic LLM substitute.

    The mock is deliberately simple but task-aware. It lets Think, Speak, and
    Reflection run without network access or API credentials.
    """

    def __init__(self, seed: int = 0):
        self.seed = int(seed)

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> LLMGeneration:
        metadata = dict(metadata or {})
        task = str(metadata.get("task", "")).lower()
        combined = f"{system_prompt or ''}\n{prompt}"

        if task == "news":
            text = self._generate_news(combined)
        elif task == "reasoning":
            text = self._generate_reasoning(combined, metadata)
        elif task == "statement":
            text = self._generate_statement(combined, metadata)
        elif task == "reflection":
            text = self._generate_reflection(combined, metadata)
        else:
            text = self._generic_response(combined, metadata)

        return LLMGeneration(text=text, metadata={"provider": "mock", **metadata})

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> LLMGeneration:
        prompt = "\n".join(f"{message.role}: {message.content}" for message in messages)
        system_prompt = None
        if messages and messages[0].role == "system":
            system_prompt = messages[0].content
        return self.generate(prompt, system_prompt=system_prompt, metadata=metadata)

    def _generate_news(self, prompt: str) -> str:
        gdp = self._last_number_after(prompt, "gdp")
        welfare = self._last_number_after(prompt, "social_welfare")
        gini = self._last_number_after(prompt, "wealth_gini")
        signals = []
        if gdp is not None:
            signals.append("GDP is expanding" if gdp >= 1.0 else "GDP is weak")
        if welfare is not None:
            signals.append("household welfare is positive" if welfare >= 0.0 else "household welfare is under pressure")
        if gini is not None:
            signals.append("wealth inequality is elevated" if gini >= 0.25 else "wealth inequality is contained")
        if not signals:
            signals.append("macro indicators are mixed")
        return "Mock economic news: " + "; ".join(signals) + "."

    def _generate_reasoning(self, prompt: str, metadata: Mapping[str, object]) -> str:
        agent_id = metadata.get("agent_id", "unknown")
        asset = self._last_number_after(prompt, "asset")
        income = self._last_number_after(prompt, "income")
        efficiency = self._last_number_after(prompt, "efficiency")
        status = "neutral"
        if income is not None and income < 0.2:
            status = "bad"
        elif asset is not None and efficiency is not None and asset > 1.0 and efficiency > 1.0:
            status = "good"
        return (
            f"Agent {agent_id} reasoning: economic status is {status}. "
            "Balance current consumption, precautionary savings, and labor effort."
        )

    def _generate_statement(self, prompt: str, metadata: Mapping[str, object]) -> str:
        agent_id = metadata.get("agent_id", "unknown")
        variants = [
            "I will keep savings disciplined while avoiding excessive labor.",
            "I see uncertainty and prefer a cautious consumption plan.",
            "I support stable wages and gradual investment in productivity.",
        ]
        index = self._stable_index(prompt, len(variants))
        return f"Agent {agent_id} statement: {variants[index]}"

    def _generate_reflection(self, prompt: str, metadata: Mapping[str, object]) -> str:
        agent_id = metadata.get("agent_id", "unknown")
        n_agents = int(metadata.get("n_agents", 1))
        trust = [8 + ((self.seed + agent_id_int(agent_id) + i) % 3) for i in range(n_agents)]
        beliefs = ["low" if i % 3 == 0 else "mid" if i % 3 == 1 else "high" for i in range(n_agents)]
        return (
            f"Agent {agent_id} reflection: peers appear heterogeneous. "
            f"wealth_belief={beliefs}; trust={trust}; "
            "self_reflection=maintain resilience and avoid overreaction."
        )

    def _generic_response(self, prompt: str, metadata: Mapping[str, object]) -> str:
        digest = hashlib.blake2b(
            f"{self.seed}:{prompt}".encode("utf-8"), digest_size=4
        ).hexdigest()
        return f"MockLLM response {digest}: concise economic interpretation."

    def _stable_index(self, text: str, modulo: int) -> int:
        digest = hashlib.blake2b(
            f"{self.seed}:{text}".encode("utf-8"), digest_size=4
        ).digest()
        return int.from_bytes(digest, "little") % modulo

    @staticmethod
    def _last_number_after(text: str, key: str) -> float | None:
        pattern = re.compile(rf"{re.escape(key)}\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
        matches = pattern.findall(text)
        if not matches:
            return None
        return float(matches[-1])


def agent_id_int(agent_id: object) -> int:
    try:
        return int(agent_id)
    except (TypeError, ValueError):
        return 0

