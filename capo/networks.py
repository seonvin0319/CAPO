"""Actor / critic / value networks for CAPO on D4RL."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor


def _mlp(sizes: Sequence[int], activation=nn.ReLU) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Deterministic tanh-squashed actor (works with TD3+BC / IQL-det / CQL-det)."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0, hidden: int = 256):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.max_action = float(max_action)
        self.hidden = int(hidden)
        self.net = _mlp([state_dim, hidden, hidden, action_dim])

    def forward(self, state: Tensor) -> Tensor:
        return self.max_action * torch.tanh(self.net(state))

    def act(self, states: Tensor) -> Tensor:
        return self.forward(states)

    def copy(self) -> "Actor":
        device = next(self.parameters()).device
        cloned = Actor(self.state_dim, self.action_dim, self.max_action, self.hidden).to(device)
        cloned.load_state_dict(self.state_dict())
        return cloned


class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256, n_hidden: int = 2):
        super().__init__()
        sizes = [state_dim + action_dim] + [hidden] * n_hidden + [1]
        self.net = _mlp(sizes)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256, n_hidden: int = 2):
        super().__init__()
        sizes = [state_dim] + [hidden] * n_hidden + [1]
        self.net = _mlp(sizes)

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state).squeeze(-1)


class CriticEnsemble(nn.Module):
    """Independent Q ensemble for pairwise improvement uncertainty."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_critics: int = 4,
        hidden: int = 256,
        n_hidden: int = 2,
    ):
        super().__init__()
        self.n_critics = int(n_critics)
        self.qs = nn.ModuleList(
            [QNet(state_dim, action_dim, hidden=hidden, n_hidden=n_hidden) for _ in range(self.n_critics)]
        )

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        """Stacked Q values: [M, B] or [M, B, K] if actions have an extra dim."""
        if action.dim() == state.dim() + 1:
            # actions: [B, K, A] → evaluate each sample
            b, k, a = action.shape
            s = state.unsqueeze(1).expand(-1, k, -1).reshape(b * k, -1)
            act = action.reshape(b * k, a)
            q = torch.stack([qnet(s, act) for qnet in self.qs], dim=0)  # [M, B*K]
            return q.view(self.n_critics, b, k)
        return torch.stack([q(state, action) for q in self.qs], dim=0)

    def q_mean(self, state: Tensor, action: Tensor) -> Tensor:
        return self.forward(state, action).mean(dim=0)

    def q_min(self, state: Tensor, action: Tensor) -> Tensor:
        return self.forward(state, action).min(dim=0).values

    def as_adapters(self) -> list:
        return [_CriticAdapter(q) for q in self.qs]


class _CriticAdapter:
    def __init__(self, qnet: QNet):
        self.qnet = qnet

    def q(self, states: Tensor, actions: Tensor) -> Tensor:
        return self.qnet(states, actions)
