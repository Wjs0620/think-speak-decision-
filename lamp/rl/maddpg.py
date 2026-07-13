"""MADDPG implementation with optional language-augmented observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .networks import Actor, Critic
from .replay_buffer import MultiAgentReplayBuffer


@dataclass
class MADDPGConfig:
    """Hyperparameters for the MADDPG trainer."""

    gamma: float = 0.95
    tau: float = 0.01
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    batch_size: int = 64
    hidden_dims: tuple[int, ...] = (128, 128)
    critic_hidden_dims: tuple[int, ...] = (256, 256)
    grad_clip_norm: float | None = 1.0
    device: str = "cpu"


class MADDPG:
    """Multi-agent DDPG with centralized critics and decentralized actors."""

    def __init__(
        self,
        n_agents: int,
        global_obs_dim: int,
        private_obs_dim: int,
        action_dim: int,
        language_dim: int = 0,
        config: MADDPGConfig | None = None,
    ):
        self.n_agents = n_agents
        self.global_obs_dim = global_obs_dim
        self.private_obs_dim = private_obs_dim
        self.action_dim = action_dim
        self.language_dim = int(language_dim)
        if self.language_dim < 0:
            raise ValueError("language_dim must be non-negative")
        self.config = config or MADDPGConfig()
        self.device = torch.device(self.config.device)
        self.actor_obs_dim = private_obs_dim + self.language_dim
        self.state_dim = global_obs_dim + n_agents * self.actor_obs_dim
        self.joint_action_dim = n_agents * action_dim

        self.actors = nn.ModuleList(
            [
                Actor(self.actor_obs_dim, action_dim, self.config.hidden_dims)
                for _ in range(n_agents)
            ]
        ).to(self.device)
        self.target_actors = nn.ModuleList(
            [
                Actor(self.actor_obs_dim, action_dim, self.config.hidden_dims)
                for _ in range(n_agents)
            ]
        ).to(self.device)
        self.critics = nn.ModuleList(
            [
                Critic(self.state_dim, self.joint_action_dim, self.config.critic_hidden_dims)
                for _ in range(n_agents)
            ]
        ).to(self.device)
        self.target_critics = nn.ModuleList(
            [
                Critic(self.state_dim, self.joint_action_dim, self.config.critic_hidden_dims)
                for _ in range(n_agents)
            ]
        ).to(self.device)

        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=self.config.actor_lr)
            for actor in self.actors
        ]
        self.critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=self.config.critic_lr)
            for critic in self.critics
        ]

        self._hard_update_targets()

    @torch.no_grad()
    def act(
        self,
        private_obs: np.ndarray,
        language_obs: np.ndarray | None = None,
        noise: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return one continuous action per household."""

        private_obs = np.asarray(private_obs, dtype=np.float32)
        if private_obs.shape != (self.n_agents, self.private_obs_dim):
            raise ValueError(
                "private_obs must have shape "
                f"{(self.n_agents, self.private_obs_dim)}, got {private_obs.shape}"
            )
        actor_obs = self._combine_numpy_actor_obs(private_obs, language_obs)
        obs_tensor = torch.as_tensor(actor_obs, dtype=torch.float32, device=self.device)
        actions = []
        for i, actor in enumerate(self.actors):
            action = actor(obs_tensor[i : i + 1]).cpu().numpy()[0]
            actions.append(action)
        actions_array = np.stack(actions).astype(np.float32)
        if noise is not None:
            actions_array = actions_array + noise
        return np.clip(actions_array, 0.0, 1.0).astype(np.float32)

    def update(self, replay_buffer: MultiAgentReplayBuffer) -> Dict[str, float]:
        """Run one MADDPG gradient update from replay buffer samples."""

        batch_size = self.config.batch_size
        if not replay_buffer.can_sample(batch_size):
            return {}
        batch = replay_buffer.sample(batch_size)
        tensors = self._batch_to_tensors(batch.as_dict())

        state = self._make_state(
            tensors["global_obs"],
            tensors["private_obs"],
            tensors["language_obs"],
        )
        next_state = self._make_state(
            tensors["next_global_obs"],
            tensors["next_private_obs"],
            tensors["next_language_obs"],
        )
        actions = tensors["actions"].reshape(batch_size, -1)

        with torch.no_grad():
            target_next_actions = self._target_actions(
                tensors["next_private_obs"], tensors["next_language_obs"]
            )
            done_mask = tensors["dones"].unsqueeze(-1)

        critic_losses: List[float] = []
        actor_losses: List[float] = []

        for agent_id in range(self.n_agents):
            reward_i = tensors["rewards"][:, agent_id : agent_id + 1]
            with torch.no_grad():
                target_q = self.target_critics[agent_id](next_state, target_next_actions)
                y = reward_i + self.config.gamma * (1.0 - done_mask) * target_q

            critic_q = self.critics[agent_id](state, actions)
            critic_loss = F.mse_loss(critic_q, y)
            self.critic_optimizers[agent_id].zero_grad()
            critic_loss.backward()
            self._clip_gradients(self.critics[agent_id])
            self.critic_optimizers[agent_id].step()
            critic_losses.append(float(critic_loss.detach().cpu()))

            current_actions = self._current_actions(
                tensors["private_obs"], tensors["language_obs"]
            )
            actor_loss = -self.critics[agent_id](state, current_actions).mean()
            self.actor_optimizers[agent_id].zero_grad()
            actor_loss.backward()
            self._clip_gradients(self.actors[agent_id])
            self.actor_optimizers[agent_id].step()
            actor_losses.append(float(actor_loss.detach().cpu()))

        self._soft_update_targets()
        return {
            "critic_loss": float(np.mean(critic_losses)),
            "actor_loss": float(np.mean(actor_losses)),
        }

    def _batch_to_tensors(self, batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in batch.items()
        }

    def _make_state(
        self,
        global_obs: torch.Tensor,
        private_obs: torch.Tensor,
        language_obs: torch.Tensor,
    ) -> torch.Tensor:
        actor_obs = self._combine_tensor_actor_obs(private_obs, language_obs)
        actor_obs_flat = actor_obs.reshape(actor_obs.shape[0], -1)
        return torch.cat([global_obs, actor_obs_flat], dim=-1)

    def _target_actions(
        self, next_private_obs: torch.Tensor, next_language_obs: torch.Tensor
    ) -> torch.Tensor:
        actor_obs = self._combine_tensor_actor_obs(next_private_obs, next_language_obs)
        actions = [
            self.target_actors[i](actor_obs[:, i, :])
            for i in range(self.n_agents)
        ]
        return torch.cat(actions, dim=-1)

    def _current_actions(
        self, private_obs: torch.Tensor, language_obs: torch.Tensor
    ) -> torch.Tensor:
        actor_obs = self._combine_tensor_actor_obs(private_obs, language_obs)
        actions = []
        for i in range(self.n_agents):
            action_i = self.actors[i](actor_obs[:, i, :])
            actions.append(action_i)
        return torch.cat(actions, dim=-1)

    def _combine_numpy_actor_obs(
        self, private_obs: np.ndarray, language_obs: np.ndarray | None
    ) -> np.ndarray:
        if self.language_dim == 0:
            if language_obs is not None and language_obs.shape[-1] != 0:
                raise ValueError("language_obs was provided but language_dim is 0")
            return private_obs
        if language_obs is None:
            raise ValueError("language_obs is required when language_dim > 0")
        language_obs = np.asarray(language_obs, dtype=np.float32)
        expected_shape = (self.n_agents, self.language_dim)
        if language_obs.shape != expected_shape:
            raise ValueError(
                f"language_obs must have shape {expected_shape}, got {language_obs.shape}"
            )
        return np.concatenate([private_obs, language_obs], axis=-1)

    def _combine_tensor_actor_obs(
        self, private_obs: torch.Tensor, language_obs: torch.Tensor
    ) -> torch.Tensor:
        if self.language_dim == 0:
            return private_obs
        if language_obs.shape[-1] != self.language_dim:
            raise ValueError(
                f"language_obs last dim must be {self.language_dim}, got {language_obs.shape[-1]}"
            )
        return torch.cat([private_obs, language_obs], dim=-1)

    def _clip_gradients(self, module: nn.Module) -> None:
        if self.config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(module.parameters(), self.config.grad_clip_norm)

    def _hard_update_targets(self) -> None:
        for target, source in zip(self.target_actors, self.actors):
            target.load_state_dict(source.state_dict())
        for target, source in zip(self.target_critics, self.critics):
            target.load_state_dict(source.state_dict())

    @torch.no_grad()
    def _soft_update_targets(self) -> None:
        tau = self.config.tau
        for target, source in zip(self.target_actors, self.actors):
            self._soft_update_module(target, source, tau)
        for target, source in zip(self.target_critics, self.critics):
            self._soft_update_module(target, source, tau)

    @staticmethod
    def _soft_update_module(target: nn.Module, source: nn.Module, tau: float) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * source_param.data)
