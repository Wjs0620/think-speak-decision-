"""Text encoding utilities for language-augmented policies.

The paper freezes a text encoder and trains only a small projection layer. This
file provides a lightweight deterministic encoder for early reproduction work.
It keeps the same interface a real sentence encoder or LLM embedding API can
implement later.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

import numpy as np


class TextEncoder(Protocol):
    """Interface for frozen text encoders."""

    embedding_dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts into an array with shape ``(len(texts), embedding_dim)``."""


class HashTextEncoder:
    """Deterministic bag-of-tokens hash encoder.

    This is not semantically rich, but it is stable, dependency-free, and useful
    for verifying all language-vector plumbing before adding a real encoder.
    """

    token_pattern = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")

    def __init__(self, embedding_dim: int = 32, lowercase: bool = True):
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = int(embedding_dim)
        self.lowercase = lowercase

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = self._tokenize(text)
            if not tokens:
                continue
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, byteorder="little", signed=False)
                index = value % self.embedding_dim
                sign = 1.0 if ((value >> 8) & 1) == 0 else -1.0
                embeddings[row, index] += sign
            norm = np.linalg.norm(embeddings[row])
            if norm > 1e-8:
                embeddings[row] /= norm
        return embeddings

    def _tokenize(self, text: str) -> list[str]:
        if self.lowercase:
            text = text.lower()
        return self.token_pattern.findall(text)


@dataclass
class LanguageFeatureBuilder:
    """Build per-agent language embeddings from agent text fields."""

    encoder: TextEncoder

    @property
    def language_dim(self) -> int:
        return self.encoder.embedding_dim

    def build(self, agent_texts: Sequence[Iterable[str] | str]) -> np.ndarray:
        """Return one embedding per agent.

        Args:
            agent_texts: Either a string per agent or an iterable of text
                fragments per agent. Multiple fragments are joined with newlines.
        """

        merged_texts = [self._merge_texts(texts) for texts in agent_texts]
        return self.encoder.encode(merged_texts).astype(np.float32)

    @staticmethod
    def _merge_texts(texts: Iterable[str] | str) -> str:
        if isinstance(texts, str):
            return texts
        return "\n".join(str(text) for text in texts if str(text).strip())


def build_agent_texts(
    reasonings: Sequence[str] | None = None,
    statements: Sequence[str] | None = None,
    reflections: Sequence[str] | None = None,
) -> list[list[str]]:
    """Combine language fields into per-agent text fragments."""

    fields = [field for field in (reasonings, statements, reflections) if field is not None]
    if not fields:
        return []
    n_agents = len(fields[0])
    for field in fields:
        if len(field) != n_agents:
            raise ValueError("all language fields must have the same number of agents")
    agent_texts: list[list[str]] = []
    for agent_id in range(n_agents):
        fragments = [field[agent_id] for field in fields if field[agent_id]]
        agent_texts.append(fragments)
    return agent_texts

