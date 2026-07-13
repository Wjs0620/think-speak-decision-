"""Smoke test for text encoding and language-augmented MADDPG inputs."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lamp.envs import ToyEconomyConfig, ToyEconomyEnv
from lamp.modules import HashTextEncoder, LanguageFeatureBuilder, build_agent_texts
from lamp.rl import GaussianNoise, MADDPG, MADDPGConfig, MultiAgentReplayBuffer


def main() -> None:
    env = ToyEconomyEnv(ToyEconomyConfig(n_agents=3, max_steps=8))
    encoder = HashTextEncoder(embedding_dim=16)
    language_builder = LanguageFeatureBuilder(encoder)
    agent = MADDPG(
        n_agents=env.n_agents,
        global_obs_dim=env.global_obs_dim,
        private_obs_dim=env.private_obs_dim,
        action_dim=env.action_dim,
        language_dim=language_builder.language_dim,
        config=MADDPGConfig(batch_size=4),
    )
    buffer = MultiAgentReplayBuffer(
        capacity=64,
        n_agents=env.n_agents,
        global_obs_dim=env.global_obs_dim,
        private_obs_dim=env.private_obs_dim,
        action_dim=env.action_dim,
        language_dim=language_builder.language_dim,
        seed=5,
    )
    noise = GaussianNoise(env.action_dim, seed=5)

    obs = env.reset(seed=5)
    language_obs = make_language_embeddings(language_builder, env, obs)
    last_losses = {}
    done = False
    while not done:
        actions = agent.act(obs["private"], language_obs=language_obs, noise=noise.sample((env.n_agents, env.action_dim)))
        step = env.step(actions)
        next_language_obs = make_language_embeddings(language_builder, env, step.observation)
        buffer.add_from_observations(
            observation=obs,
            actions=actions,
            rewards=step.rewards,
            next_observation=step.observation,
            done=step.done,
            language_obs=language_obs,
            next_language_obs=next_language_obs,
        )
        last_losses = agent.update(buffer) or last_losses
        obs = step.observation
        language_obs = next_language_obs
        done = step.done
        noise.step()

    print("language dim:", language_builder.language_dim)
    print("language obs shape:", language_obs.shape)
    print("buffer size:", len(buffer))
    print("last losses:", last_losses)


def make_language_embeddings(language_builder: LanguageFeatureBuilder, env: ToyEconomyEnv, obs):
    reasonings = []
    for agent_id in range(env.n_agents):
        private = obs["private"][agent_id]
        reasonings.append(
            "agent {agent_id} asset {asset:.3f} efficiency {eff:.3f} "
            "income {income:.3f} wage {wage:.3f} gdp {gdp:.3f}".format(
                agent_id=agent_id,
                asset=private[0],
                eff=private[1],
                income=private[2],
                wage=obs["global"][0],
                gdp=obs["global"][4],
            )
        )
    agent_texts = build_agent_texts(reasonings=reasonings)
    return language_builder.build(agent_texts)


if __name__ == "__main__":
    main()

