"""Smoke test for the LAMP Speak and Reflection modules."""

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
    SpeakConfig,
    SpeakModule,
    ThinkConfig,
    ThinkModule,
    build_agent_texts,
    summarize_observation,
)


def main() -> None:
    env = ToyEconomyEnv(ToyEconomyConfig(n_agents=3, max_steps=8))
    llm = MockLLM(seed=23)
    think = ThinkModule(llm, ThinkConfig(shock_threshold=0.10, long_term_interval=4))
    speak = SpeakModule(llm, SpeakConfig(n_candidates=3, max_context_items=2))
    encoder = HashTextEncoder(embedding_dim=24)
    pools = DualExperiencePool(encoder, short_capacity=8, long_capacity=8)

    obs = env.reset(seed=23)
    collected = []
    for _ in range(3):
        think_output = think.step(obs)
        actions = env.sample_random_actions()
        step = env.step(actions)
        for reasoning in think_output.agent_reasonings:
            collected.append(
                ReasoningExperience(
                    agent_id=reasoning.agent_id,
                    time_step=int(obs["step"]),
                    reasoning=reasoning.text,
                    reward=float(step.rewards[reasoning.agent_id]),
                    observation_summary=summarize_observation(
                        obs["global_names"],
                        obs["global"],
                        obs["private_names"],
                        obs["private"][reasoning.agent_id],
                    ),
                    action=tuple(actions[reasoning.agent_id]),
                    news_type=think_output.news_type,
                )
            )
        obs = step.observation

    pools.short.harvest_top_k_by_agent(collected, k_per_agent=1)
    think_output = think.step(obs)
    contexts = {
        agent_id: pools.retrieve_context(think_output.agent_reasonings[agent_id].text)
        for agent_id in range(env.n_agents)
    }
    speak_output = speak.step(obs, think_output, contexts)
    agent_texts = build_agent_texts(
        reasonings=think_output.reasoning_texts,
        statements=speak_output.selected_statements,
        reflections=speak_output.reflection_texts,
    )

    print("selected statements:")
    for statement in speak_output.selected_statements:
        print(" ", statement)
    print("trust matrix shape:", speak_output.trust_matrix.shape)
    print("wealth beliefs:", speak_output.wealth_beliefs)
    print("agent text fragments:", len(agent_texts), len(agent_texts[0]))


if __name__ == "__main__":
    main()

