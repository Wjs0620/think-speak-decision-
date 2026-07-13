"""Smoke test for LAMP reasoning experience pools."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lamp.envs import ToyEconomyConfig, ToyEconomyEnv
from lamp.llm import MockLLM
from lamp.modules import (
    DualExperiencePool,
    HashTextEncoder,
    ReasoningExperience,
    ThinkConfig,
    ThinkModule,
    summarize_observation,
)


def main() -> None:
    env = ToyEconomyEnv(ToyEconomyConfig(n_agents=3, max_steps=8))
    think = ThinkModule(MockLLM(seed=17), ThinkConfig(shock_threshold=0.10, long_term_interval=4))
    pools = DualExperiencePool(HashTextEncoder(embedding_dim=24), short_capacity=5, long_capacity=8)

    obs = env.reset(seed=17)
    collected = []
    for _ in range(6):
        output = think.step(obs)
        actions = env.sample_random_actions()
        step = env.step(actions)

        for reasoning in output.agent_reasonings:
            summary = summarize_observation(
                obs["global_names"],
                obs["global"],
                obs["private_names"],
                obs["private"][reasoning.agent_id],
            )
            collected.append(
                ReasoningExperience(
                    agent_id=reasoning.agent_id,
                    time_step=int(obs["step"]),
                    reasoning=reasoning.text,
                    reward=float(step.rewards[reasoning.agent_id]),
                    observation_summary=summary,
                    action=tuple(actions[reasoning.agent_id]),
                    news_type=output.news_type,
                )
            )
        obs = step.observation

    short_inserted = pools.short.harvest_top_k_by_agent(collected, k_per_agent=1)
    long_inserted = pools.long.add_top_k(collected, k=4)
    query = collected[-1].text_for_embedding
    retrieved = pools.retrieve_context(query, top_k_long=2, top_k_short=2)

    print("collected:", len(collected))
    print("short inserted:", len(short_inserted), "short size:", len(pools.short))
    print("long inserted:", len(long_inserted), "long size:", len(pools.long))
    for item in retrieved:
        exp = item.experience
        print(
            f"retrieved score={item.score:.4f} "
            f"agent={exp.agent_id} step={exp.time_step} reward={exp.reward:.4f}"
        )


if __name__ == "__main__":
    main()

