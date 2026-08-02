"""D4RL dataset loading and replay buffer."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gym
import numpy as np
import torch
from torch import Tensor

# Register D4RL envs with gym (must import before gym.make).
try:
    import d4rl  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "d4rl is required. Activate offrl_backup or install D4RL."
    ) from exc


@dataclass
class DatasetStats:
    state_mean: np.ndarray
    state_std: np.ndarray
    max_action: float


def _return_reward_range(dataset: Dict[str, np.ndarray], max_episode_steps: int):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    if ep_len > 0:
        returns.append(ep_ret)
    return (min(returns), max(returns)) if returns else (0.0, 1.0)


def modify_reward(dataset: Dict[str, np.ndarray], env_name: str, max_episode_steps: int = 1000):
    name = env_name.lower()
    if any(s in name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = _return_reward_range(dataset, max_episode_steps)
        scale = max(max_ret - min_ret, 1e-8)
        dataset["rewards"] = dataset["rewards"] / scale * max_episode_steps
    elif "antmaze" in name:
        dataset["rewards"] = dataset["rewards"] - 1.0


def compute_mean_std(states: np.ndarray, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((states - mean) / std).astype(np.float32)


def make_env(env_name: str, seed: int = 0):
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    # AntMaze prints "Target Goal: ..." on make/reset; silence that noise.
    if "antmaze" in env_name.lower():
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            env = gym.make(env_name)
    else:
        env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def load_d4rl_dataset(
    env_name: str,
    normalize: bool = True,
    normalize_reward: bool = False,
    device: str = "cpu",
) -> Tuple[Dict[str, np.ndarray], DatasetStats, gym.Env]:
    env = make_env(env_name)
    dataset = env.get_dataset()
    observations = np.asarray(dataset["observations"], dtype=np.float32)
    actions = np.asarray(dataset["actions"], dtype=np.float32)
    rewards = np.asarray(dataset["rewards"], dtype=np.float32)
    terminals = np.asarray(dataset["terminals"], dtype=np.float32)
    timeouts = np.asarray(dataset.get("timeouts", np.zeros_like(terminals)), dtype=np.float32)
    dones = np.clip(terminals + timeouts, 0.0, 1.0)

    if "next_observations" in dataset:
        next_observations = np.asarray(dataset["next_observations"], dtype=np.float32)
    else:
        # AntMaze (and some maze) datasets omit next_observations.
        next_observations = np.concatenate([observations[1:], observations[-1:]], axis=0)

    data = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": dones,
        "next_observations": next_observations,
    }
    if normalize_reward:
        modify_reward(data, env_name, getattr(env, "_max_episode_steps", 1000))

    if normalize:
        mean, std = compute_mean_std(data["observations"])
        data["observations"] = normalize_states(data["observations"], mean, std)
        data["next_observations"] = normalize_states(data["next_observations"], mean, std)
    else:
        mean = np.zeros(data["observations"].shape[1], dtype=np.float32)
        std = np.ones(data["observations"].shape[1], dtype=np.float32)

    max_action = float(np.abs(env.action_space.high).max())
    stats = DatasetStats(state_mean=mean, state_std=std, max_action=max_action)
    return data, stats, env


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, buffer_size: int, device: str):
        self.device = torch.device(device)
        self.max_size = int(buffer_size)
        self.ptr = 0
        self.size = 0
        self.states = torch.zeros((self.max_size, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.max_size, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((self.max_size, 1), dtype=torch.float32, device=self.device)
        self.next_states = torch.zeros((self.max_size, state_dim), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.max_size, 1), dtype=torch.float32, device=self.device)

    def load_d4rl(self, data: Dict[str, np.ndarray]):
        n = data["observations"].shape[0]
        if n > self.max_size:
            raise ValueError(f"dataset size {n} > buffer_size {self.max_size}")
        self.states[:n] = torch.from_numpy(data["observations"])
        self.actions[:n] = torch.from_numpy(data["actions"])
        self.rewards[:n] = torch.from_numpy(data["rewards"]).unsqueeze(-1)
        self.next_states[:n] = torch.from_numpy(data["next_observations"])
        self.dones[:n] = torch.from_numpy(data["terminals"]).unsqueeze(-1)
        self.ptr = n % self.max_size
        self.size = n

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )


class NormalizeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, mean: np.ndarray, std: np.ndarray):
        super().__init__(env)
        self.mean = mean
        self.std = std

    def observation(self, observation):
        return (observation - self.mean) / self.std
