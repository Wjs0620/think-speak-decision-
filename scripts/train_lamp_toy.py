"""Run the minimal LAMP pipeline on the toy economy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lamp import LAMPTrainer, LAMPTrainerConfig
from lamp.envs import ToyEconomyConfig, ToyEconomyEnv
from lamp.llm import MockLLM
from lamp.modules import (
    DualExperiencePool,
    HashTextEncoder,
    LanguageFeatureBuilder,
    SpeakConfig,
    SpeakModule,
    ThinkConfig,
    ThinkModule,
)
from lamp.rl import GaussianNoise, MADDPG, MADDPGConfig, MultiAgentReplayBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--buffer-size", type=int, default=2048)
    parser.add_argument("--language-dim", type=int, default=24)
    parser.add_argument("--seed", type=int, default=101)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = ToyEconomyEnv(
        ToyEconomyConfig(n_agents=args.n_agents, max_steps=args.max_steps)
    )
    llm = MockLLM(seed=args.seed)
    encoder = HashTextEncoder(embedding_dim=args.language_dim)
    language_builder = LanguageFeatureBuilder(encoder)
    think = ThinkModule(
        llm,
        ThinkConfig(shock_threshold=0.10, long_term_interval=4),
    )
    speak = SpeakModule(llm, SpeakConfig(n_candidates=3, max_context_items=2))
    pools = DualExperiencePool(encoder, short_capacity=128, long_capacity=512)
    maddpg = MADDPG(
        n_agents=env.n_agents,
        global_obs_dim=env.global_obs_dim,
        private_obs_dim=env.private_obs_dim,
        action_dim=env.action_dim,
        language_dim=language_builder.language_dim,
        config=MADDPGConfig(batch_size=args.batch_size),
    )
    replay_buffer = MultiAgentReplayBuffer(
        capacity=args.buffer_size,
        n_agents=env.n_agents,
        global_obs_dim=env.global_obs_dim,
        private_obs_dim=env.private_obs_dim,
        action_dim=env.action_dim,
        language_dim=language_builder.language_dim,
        seed=args.seed,
    )
    trainer = LAMPTrainer(
        env=env,
        think=think,
        speak=speak,
        language_builder=language_builder,
        experience_pool=pools,
        maddpg=maddpg,
        replay_buffer=replay_buffer,
        noise=GaussianNoise(env.action_dim, sigma=0.20, seed=args.seed),
        config=LAMPTrainerConfig(
            episodes=args.episodes,
            seed=args.seed,
            speak_news_types=("long", "short"),
        ),
    )

    for stat in trainer.train():
        losses = dict(stat.last_losses)
        critic_loss = losses.get("critic_loss", float("nan"))
        actor_loss = losses.get("actor_loss", float("nan"))
        print(
            f"episode={stat.episode} steps={stat.steps} "
            f"mean_reward_sum={stat.mean_reward_sum:.3f} "
            f"replay={stat.replay_size} short_pool={stat.short_pool_size} "
            f"long_pool={stat.long_pool_size} "
            f"critic_loss={critic_loss:.4f} actor_loss={actor_loss:.4f}"
        )


if __name__ == "__main__":
    main()

