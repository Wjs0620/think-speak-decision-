"""Base LLM interfaces shared by Think, Speak, and Reflection modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True)
class ChatMessage:
    """One chat-style prompt message."""

    role: str
    content: str


@dataclass
class LLMConfig:
    """Configuration for a remote LLM client.

    Values are intentionally blank by default. Fill them later when replacing
    ``MockLLM`` with a real provider.
    """

    api_key: str = ""
    base_url: str = ""
    model: str = ""
    temperature: float = 0.2
    max_tokens: int = 512
    timeout_seconds: int = 60


@dataclass(frozen=True)
class LLMGeneration:
    """Structured return value from an LLM call."""

    text: str
    metadata: Mapping[str, object]


class LLMNotConfiguredError(RuntimeError):
    """Raised when a remote LLM is called without API settings."""


class BaseLLM(Protocol):
    """Minimal interface used by LAMP language modules."""

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> LLMGeneration:
        """Generate one text response."""

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> LLMGeneration:
        """Generate one response from chat messages."""

