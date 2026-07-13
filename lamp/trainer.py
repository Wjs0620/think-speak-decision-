"""Minimal LAMP training loop that connects Think, Speak, and Decide."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from lamp.envs import BaseMultiAgentEconomyEnv
from lamp.modules import (
    DualExperiencePool,
    LanguageFeatureBuilder,
    ReasoningExperience,
    SpeakModule,
    SpeakOutput,
    ThinkModule,
    ThinkOutput,
    build_agent_texts,
    summarize_observation,
)
from lamp.rl import GaussianNoise, MADDPG, MultiAgentReplayBuffer


@dataclass
class LAMPTrainerConfig:
    """Configuration for the minimal LAMP training loop."""

    episodes: int = 5
    replay_updates_per_step: int = 1
    short_top_k_per_agent: int = 1
    long_top_k: int = 8
    retrieve_top_k_long: int = 3
    retrieve_top_k_short: int = 2
    speak_news_types: tuple[str, ...] = ("long", "short")
    seed: int = 42


@dataclass(frozen=True)
class LanguageStep:
    """Language-side outputs for one environment observation."""

    think_output: ThinkOutput
    speak_output: SpeakOutput | None
    language_obs: np.ndarray
    selected_statements: list[str]
    reflection_texts: list[str]


@dataclass(frozen=True)
class EpisodeStats:
    """Summary from one LAMP episode."""

    episode: int
    steps: int
    mean_reward_sum: float
    replay_size: int
    short_pool_size: int
    long_pool_size: int
    last_losses: Mapping[str, float] = field(default_factory=dict)


class LAMPTrainer:
    """Connect LAMP language modules with MADDPG training."""

    def __init__(
        self,
        env: BaseMultiAgentEconomyEnv,
        think: ThinkModule,
        speak: SpeakModule,
        language_builder: LanguageFeatureBuilder,
        experience_pool: DualExperiencePool,
        maddpg: MADDPG,
        replay_buffer: MultiAgentReplayBuffer,
        noise: GaussianNoise | None = None,
        config: LAMPTrainerConfig | None = None,
    ):
        self.env = env
        self.think = think
        self.speak = speak
        self.language_builder = language_builder
        self.experience_pool = experience_pool
        self.maddpg = maddpg
        self.replay_buffer = replay_buffer
        self.noise = noise
        self.config = config or LAMPTrainerConfig()

    def train(self) -> list[EpisodeStats]:
        stats = []
        for episode in range(1, self.config.episodes + 1):
            stats.append(self.run_episode(episode))
        return stats

    def run_episode(self, episode: int) -> EpisodeStats:
        obs = self.env.reset(seed=self.config.seed + episode)
        self.think.reset()
        self.experience_pool.reset_short()

        language = self._language_step(obs)
        done = False
        mean_reward_sum = 0.0
        last_losses: Mapping[str, float] = {}
        episode_experiences: list[ReasoningExperience] = []

        while not done:
            action_noise = None
            if self.noise is not None:
                action_noise = self.noise.sample((self.env.n_agents, self.env.action_dim))
            actions = self.maddpg.act(
                obs["private"],
                language_obs=language.language_obs,
                noise=action_noise,
            )
            step = self.env.step(actions)
            next_language = self._language_step(step.observation)

            self.replay_buffer.add_from_observations(
                observation=obs,
                actions=actions,
                rewards=step.rewards,
                next_observation=step.observation,
                done=step.done,
                language_obs=language.language_obs,
                next_language_obs=next_language.language_obs,
            )
            for _ in range(self.config.replay_updates_per_step):
                last_losses = self.maddpg.update(self.replay_buffer) or last_losses

            step_experiences = self._build_experiences(
                observation=obs,
                language_step=language,
                actions=actions,
                rewards=step.rewards,
            )
            episode_experiences.extend(step_experiences)
            self.experience_pool.short.harvest_top_k_by_agent(
                step_experiences,
                k_per_agent=self.config.short_top_k_per_agent,
            )
            if language.think_output.news_type == "long":
                self.experience_pool.long.add_top_k(
                    episode_experiences,
                    k=self.config.long_top_k,
                )

            mean_reward_sum += float(np.mean(step.rewards))
            if self.noise is not None:
                self.noise.step()
            obs = step.observation
            language = next_language
            done = step.done

        return EpisodeStats(
            episode=episode,
            steps=int(obs["step"]),
            mean_reward_sum=mean_reward_sum,
            replay_size=len(self.replay_buffer),
            short_pool_size=len(self.experience_pool.short),
            long_pool_size=len(self.experience_pool.long),
            last_losses=last_losses,
        )

    def _language_step(self, observation: Mapping[str, Any]) -> LanguageStep:
        think_output = self.think.step(observation)
        contexts = {
            reasoning.agent_id: self.experience_pool.retrieve_context(
                reasoning.text,
                top_k_long=self.config.retrieve_top_k_long,
                top_k_short=self.config.retrieve_top_k_short,
            )
            for reasoning in think_output.agent_reasonings
        }

        speak_output = None
        selected_statements = [""] * self.env.n_agents
        reflection_texts = [""] * self.env.n_agents
        if think_output.news_type in self.config.speak_news_types:
            speak_output = self.speak.step(observation, think_output, contexts)
            selected_statements = speak_output.selected_statements
            reflection_texts = speak_output.reflection_texts

        agent_texts = build_agent_texts(
            reasonings=think_output.reasoning_texts,
            statements=selected_statements,
            reflections=reflection_texts,
        )
        language_obs = self.language_builder.build(agent_texts)
        return LanguageStep(
            think_output=think_output,
            speak_output=speak_output,
            language_obs=language_obs,
            selected_statements=selected_statements,
            reflection_texts=reflection_texts,
        )

    def _build_experiences(
        self,
        observation: Mapping[str, Any],
        language_step: LanguageStep,
        actions: np.ndarray,
        rewards: np.ndarray,
    ) -> list[ReasoningExperience]:
        experiences = []
        for reasoning in language_step.think_output.agent_reasonings:
            agent_id = reasoning.agent_id
            reflection = language_step.reflection_texts[agent_id]
            experiences.append(
                ReasoningExperience(
                    agent_id=agent_id,
                    time_step=int(observation["step"]),
                    reasoning=reasoning.text,
                    reward=float(rewards[agent_id]),
                    observation_summary=summarize_observation(
                        observation.get("global_names", ()),
                        observation["global"],
                        observation.get("private_names", ()),
                        observation["private"][agent_id],
                    ),
                    reflection=reflection,
                    action=tuple(float(value) for value in actions[agent_id]),
                    news_type=language_step.think_output.news_type,
                    metadata={
                        "selected_statement": language_step.selected_statements[agent_id],
                        "economic_status": reasoning.economic_status,
                        "economic_status_label": reasoning.economic_status_label,
                    },
                )
            )
        return experiences

