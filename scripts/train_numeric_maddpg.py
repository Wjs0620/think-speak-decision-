"""Run a short numeric-only MADDPG training loop on the toy economy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lamp.envs import ToyEconomyConfig, ToyEconomyEnv
from lamp.rl import GaussianNoise, MADDPG, MADDPGConfig, MultiAgentReplayBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--n-agents", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--buffer-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = ToyEconomyEnv(
        ToyEconomyConfig(n_agents=args.n_agents, max_steps=args.max_steps)
    )
    agent = MADDPG(
        n_agents=env.n_agents,
        global_obs_dim=env.global_obs_dim,
        private_obs_dim=env.private_obs_dim,
        action_dim=env.action_dim,
        config=MADDPGConfig(batch_size=args.batch_size),
    )
    replay_buffer = MultiAgentReplayBuffer(
        capacity=args.buffer_size,
        n_agents=env.n_agents,
        global_obs_dim=env.global_obs_dim,
        private_obs_dim=env.private_obs_dim,
        action_dim=env.action_dim,
        seed=args.seed,
    )
    noise = GaussianNoise(env.action_dim, sigma=0.20, seed=args.seed)

    total_steps = 0
    for episode in range(1, args.episodes + 1):
        obs = env.reset(seed=args.seed + episode)
        done = False
        episode_reward = 0.0
        last_losses = {}

        while not done:
            action_noise = noise.sample((env.n_agents, env.action_dim))
            actions = agent.act(obs["private"], noise=action_noise)
            step = env.step(actions)
            replay_buffer.add_from_observations(
                observation=obs,
                actions=actions,
                rewards=step.rewards,
                next_observation=step.observation,
                done=step.done,
            )
            last_losses = agent.update(replay_buffer) or last_losses
            episode_reward += float(step.rewards.mean())
            obs = step.observation
            done = step.done
            total_steps += 1
            noise.step()

        critic_loss = last_losses.get("critic_loss", float("nan"))
        actor_loss = last_losses.get("actor_loss", float("nan"))
        print(
            f"episode={episode} "
            f"steps={env.step_count} "
            f"mean_reward_sum={episode_reward:.3f} "
            f"buffer={len(replay_buffer)} "
            f"critic_loss={critic_loss:.4f} "
            f"actor_loss={actor_loss:.4f}"
        )

    print(f"finished numeric MADDPG smoke training, total_steps={total_steps}")


if __name__ == "__main__":
    main()

