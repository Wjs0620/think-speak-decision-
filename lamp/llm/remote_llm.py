"""OpenAI-compatible remote LLM client.

The API key, base URL, and model are intentionally blank by default. Fill
``LLMConfig`` later to use ChatGPT or another OpenAI-compatible endpoint.
"""

from __future__ import annotations

import json
from typing import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base_llm import ChatMessage, LLMConfig, LLMGeneration, LLMNotConfiguredError


class RemoteLLM:
    """HTTP client for OpenAI-compatible chat completions APIs."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> LLMGeneration:
        messages = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))
        return self.chat(messages, metadata=metadata)

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> LLMGeneration:
        self._ensure_configured()
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        response = self._post_json(self._chat_completions_url(), payload)
        text = self._extract_text(response)
        response_metadata = {
            "provider": "remote",
            "model": response.get("model", self.config.model),
            "usage": response.get("usage", {}),
            "id": response.get("id", ""),
            **dict(metadata or {}),
        }
        return LLMGeneration(text=text, metadata=response_metadata)

    def _ensure_configured(self) -> None:
        if not self.config.api_key or not self.config.base_url or not self.config.model:
            raise LLMNotConfiguredError(
                "RemoteLLM requires non-empty api_key, base_url, and model."
            )

    def _chat_completions_url(self) -> str:
        base_url = self.config.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    def _post_json(self, url: str, payload: Mapping[str, object]) -> Mapping[str, object]:
        data = json.dumps(payload).encode("utf-8")
        request = Request(
            url=url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP error {exc.code}: {error_body}") from exc
        except URLError as exc:
            raise RuntimeError(f"LLM connection error: {exc}") from exc
        return json.loads(raw)

    @staticmethod
    def _extract_text(response: Mapping[str, object]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"LLM response missing choices: {response}")
        first_choice = choices[0]
        if not isinstance(first_choice, Mapping):
            raise RuntimeError(f"LLM response choice has invalid shape: {response}")
        message = first_choice.get("message")
        if isinstance(message, Mapping):
            content = message.get("content")
            if isinstance(content, str):
                return content
        text = first_choice.get("text")
        if isinstance(text, str):
            return text
        raise RuntimeError(f"LLM response missing text content: {response}")
