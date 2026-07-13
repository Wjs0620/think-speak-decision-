"""Smoke test for the toy economy environment."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lamp.envs import ToyEconomyConfig, ToyEconomyEnv


def main() -> None:
    env = ToyEconomyEnv(ToyEconomyConfig(n_agents=4, max_steps=5))
    obs = env.reset(seed=7)
    print("initial global:", dict(zip(obs["global_names"], obs["global"].round(4))))
    print("initial private shape:", obs["private"].shape)

    done = False
    while not done:
        actions = env.sample_random_actions()
        step = env.step(actions)
        done = step.done
        print(
            f"step={step.observation['step']} "
            f"reward_mean={step.rewards.mean():.4f} "
            f"gdp={step.info['gdp']:.4f} "
            f"welfare={step.info['social_welfare']:.4f} "
            f"gini={step.info['wealth_gini']:.4f}"
        )


if __name__ == "__main__":
    main()
