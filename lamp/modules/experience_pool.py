"""Language reasoning experience pools for LAMP.

These pools store high-reward reasoning trajectories for later retrieval. They
are separate from the RL replay buffer, which stores numeric transitions for
actor-critic training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .text_encoder import TextEncoder


@dataclass(frozen=True)
class ReasoningExperience:
    """One language reasoning trajectory stored in an experience pool."""

    agent_id: int
    time_step: int
    reasoning: str
    reward: float
    observation_summary: str = ""
    reflection: str = ""
    action: tuple[float, ...] = field(default_factory=tuple)
    news_type: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None

    @property
    def text_for_embedding(self) -> str:
        parts = [
            self.observation_summary,
            self.reasoning,
            self.reflection,
            f"news_type: {self.news_type}",
        ]
        return "\n".join(part for part in parts if part)


@dataclass(frozen=True)
class RetrievedExperience:
    """Experience returned by similarity retrieval."""

    experience: ReasoningExperience
    score: float


class ReasoningExperiencePool:
    """Capacity-limited pool with top-reward insertions and kNN retrieval."""

    def __init__(
        self,
        encoder: TextEncoder,
        capacity: int = 1024,
        name: str = "experience",
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.encoder = encoder
        self.capacity = int(capacity)
        self.name = name
        self._items: list[ReasoningExperience] = []

    def __len__(self) -> int:
        return len(self._items)

    @property
    def items(self) -> tuple[ReasoningExperience, ...]:
        return tuple(self._items)

    def clear(self) -> None:
        self._items.clear()

    def add(self, experience: ReasoningExperience) -> None:
        """Add one experience, computing its embedding if needed."""

        prepared = self._with_embedding(experience)
        self._items.append(prepared)
        self._enforce_capacity()

    def add_many(self, experiences: Iterable[ReasoningExperience]) -> None:
        for experience in experiences:
            self.add(experience)

    def add_top_k(self, experiences: Sequence[ReasoningExperience], k: int) -> list[ReasoningExperience]:
        """Add top-k experiences by reward and return the inserted entries."""

        if k <= 0:
            return []
        selected = sorted(experiences, key=lambda item: item.reward, reverse=True)[:k]
        self.add_many(selected)
        return selected

    def retrieve(
        self,
        query: str | np.ndarray,
        top_k: int = 3,
        min_score: float | None = None,
    ) -> list[RetrievedExperience]:
        """Retrieve experiences using cosine similarity."""

        if top_k <= 0 or not self._items:
            return []
        query_embedding = self._query_embedding(query)
        item_embeddings = np.stack([self._require_embedding(item) for item in self._items])
        scores = item_embeddings @ query_embedding
        order = np.argsort(-scores)
        results: list[RetrievedExperience] = []
        for idx in order[:top_k]:
            score = float(scores[idx])
            if min_score is not None and score < min_score:
                continue
            results.append(RetrievedExperience(self._items[int(idx)], score))
        return results

    def harvest_top_k_by_agent(
        self,
        experiences: Sequence[ReasoningExperience],
        k_per_agent: int,
    ) -> list[ReasoningExperience]:
        """Add top-k reward experiences for each agent."""

        if k_per_agent <= 0:
            return []
        by_agent: dict[int, list[ReasoningExperience]] = {}
        for experience in experiences:
            by_agent.setdefault(experience.agent_id, []).append(experience)

        inserted: list[ReasoningExperience] = []
        for agent_experiences in by_agent.values():
            inserted.extend(self.add_top_k(agent_experiences, k_per_agent))
        return inserted

    def _with_embedding(self, experience: ReasoningExperience) -> ReasoningExperience:
        if experience.embedding is not None:
            embedding = self._normalize(np.asarray(experience.embedding, dtype=np.float32))
        else:
            embedding = self.encoder.encode([experience.text_for_embedding])[0]
            embedding = self._normalize(embedding)
        return ReasoningExperience(
            agent_id=experience.agent_id,
            time_step=experience.time_step,
            reasoning=experience.reasoning,
            reward=float(experience.reward),
            observation_summary=experience.observation_summary,
            reflection=experience.reflection,
            action=tuple(float(value) for value in experience.action),
            news_type=experience.news_type,
            metadata=dict(experience.metadata),
            embedding=embedding,
        )

    def _enforce_capacity(self) -> None:
        if len(self._items) <= self.capacity:
            return
        self._items.sort(key=lambda item: item.reward, reverse=True)
        self._items = self._items[: self.capacity]

    def _query_embedding(self, query: str | np.ndarray) -> np.ndarray:
        if isinstance(query, str):
            embedding = self.encoder.encode([query])[0]
        else:
            embedding = np.asarray(query, dtype=np.float32)
        return self._normalize(embedding)

    @staticmethod
    def _require_embedding(experience: ReasoningExperience) -> np.ndarray:
        if experience.embedding is None:
            raise ValueError("experience has no embedding")
        return experience.embedding

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        embedding = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-8:
            return np.zeros_like(embedding, dtype=np.float32)
        return (embedding / norm).astype(np.float32)


class DualExperiencePool:
    """Short-term and long-term reasoning pools used by LAMP."""

    def __init__(
        self,
        encoder: TextEncoder,
        short_capacity: int = 512,
        long_capacity: int = 2048,
    ):
        self.short = ReasoningExperiencePool(
            encoder=encoder,
            capacity=short_capacity,
            name="short",
        )
        self.long = ReasoningExperiencePool(
            encoder=encoder,
            capacity=long_capacity,
            name="long",
        )

    def reset_short(self) -> None:
        self.short.clear()

    def retrieve_context(
        self,
        query: str | np.ndarray,
        top_k_long: int = 3,
        top_k_short: int = 2,
    ) -> list[RetrievedExperience]:
        """Retrieve from both long and short pools."""

        results = []
        results.extend(self.long.retrieve(query, top_k=top_k_long))
        results.extend(self.short.retrieve(query, top_k=top_k_short))
        results.sort(key=lambda item: item.score, reverse=True)
        deduped: list[RetrievedExperience] = []
        seen: set[tuple[int, int, str]] = set()
        for item in results:
            exp = item.experience
            key = (exp.agent_id, exp.time_step, exp.reasoning)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped


def summarize_observation(
    global_names: Sequence[str],
    global_obs: Sequence[float],
    private_names: Sequence[str],
    private_obs: Sequence[float],
) -> str:
    """Create a compact text summary for experience storage."""

    global_part = ", ".join(
        f"{name}: {float(value):.4f}" for name, value in zip(global_names, global_obs)
    )
    private_part = ", ".join(
        f"{name}: {float(value):.4f}" for name, value in zip(private_names, private_obs)
    )
    return f"global {{{global_part}}}; private {{{private_part}}}"
