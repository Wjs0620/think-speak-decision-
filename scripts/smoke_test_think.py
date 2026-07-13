"""Smoke test for the LAMP Think module."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lamp.envs import ToyEconomyConfig, ToyEconomyEnv
from lamp.llm import MockLLM
from lamp.modules import ThinkConfig, ThinkModule


def main() -> None:
    env = ToyEconomyEnv(ToyEconomyConfig(n_agents=3, max_steps=12))
    think = ThinkModule(
        llm=MockLLM(seed=13),
        config=ThinkConfig(shock_threshold=0.10, long_term_interval=4),
    )
    obs = env.reset(seed=13)

    for _ in range(6):
        output = think.step(obs)
        print(
            f"step={obs['step']} type={output.news_type} "
            f"statuses={output.economic_statuses.tolist()}"
        )
        if output.news:
            print("news:", output.news)
        print("reasoning[0]:", output.agent_reasonings[0].text)

        actions = env.sample_random_actions()
        step = env.step(actions)
        obs = step.observation


if __name__ == "__main__":
    main()

