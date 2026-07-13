"""Smoke test for the multi-agent replay buffer."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lamp.envs import ToyEconomyConfig, ToyEconomyEnv
from lamp.rl import MultiAgentReplayBuffer


def main() -> None:
    env = ToyEconomyEnv(ToyEconomyConfig(n_agents=4, max_steps=8))
    buffer = MultiAgentReplayBuffer(
        capacity=6,
        n_agents=env.n_agents,
        global_obs_dim=env.global_obs_dim,
        private_obs_dim=env.private_obs_dim,
        action_dim=env.action_dim,
        seed=11,
    )

    obs = env.reset(seed=3)
    for _ in range(8):
        actions = env.sample_random_actions()
        step = env.step(actions)
        buffer.add_from_observations(
            observation=obs,
            actions=actions,
            rewards=step.rewards,
            next_observation=step.observation,
            done=step.done,
        )
        obs = step.observation
        if step.done:
            obs = env.reset()

    batch = buffer.sample(batch_size=3)
    print("buffer size:", len(buffer))
    print("buffer full:", buffer.is_full)
    print("global batch shape:", batch.global_obs.shape)
    print("private batch shape:", batch.private_obs.shape)
    print("actions batch shape:", batch.actions.shape)
    print("rewards batch shape:", batch.rewards.shape)
    print("dones:", batch.dones)


if __name__ == "__main__":
    main()

