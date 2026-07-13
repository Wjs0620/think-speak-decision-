"""Smoke test for the mock and placeholder LLM clients."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lamp.llm import LLMConfig, LLMNotConfiguredError, MockLLM, RemoteLLM


def main() -> None:
    llm = MockLLM(seed=7)
    news = llm.generate(
        "gdp: 0.8 social_welfare: -3.2 wealth_gini: 0.31",
        metadata={"task": "news"},
    )
    reasoning = llm.generate(
        "asset: 0.5 income: 0.1 efficiency: 0.9",
        metadata={"task": "reasoning", "agent_id": 2},
    )
    statement = llm.generate(
        reasoning.text,
        metadata={"task": "statement", "agent_id": 2},
    )
    reflection = llm.generate(
        statement.text,
        metadata={"task": "reflection", "agent_id": 2, "n_agents": 4},
    )

    print(news.text)
    print(reasoning.text)
    print(statement.text)
    print(reflection.text)

    try:
        RemoteLLM(LLMConfig()).generate("hello")
    except LLMNotConfiguredError as exc:
        print("remote unconfigured:", str(exc))

    configured = RemoteLLM(
        LLMConfig(
            api_key="",
            base_url="",
            model="",
        )
    )
    print("remote client class:", configured.__class__.__name__)


if __name__ == "__main__":
    main()
