"""Proximal W2 actor refinement for deterministic policies.

Solves approximately:
    π' = argmax_π  E[Q(s, π(s))] - (1/(2τ)) ||π(s) - π_center(s)||²
"""
from __future__ import annotations

import copy
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .networks import Actor, CriticEnsemble, _CriticAdapter


class ProximalW2Refiner:
    """Few-step proximal refinement around a frozen center policy."""

    def __init__(
        self,
        lr: float = 3e-4,
        n_steps: int = 1,
        max_grad_norm: float = 10.0,
        use_min_q: bool = False,
    ):
        self.lr = lr
        self.n_steps = int(n_steps)
        self.max_grad_norm = max_grad_norm
        self.use_min_q = use_min_q

    def refine(
        self,
        policy_center: Actor,
        critics: Sequence[_CriticAdapter] | CriticEnsemble,
        tau: float,
        states: Tensor,
    ) -> Actor:
        tau = max(float(tau), 1e-8)
        w2_weight = 1.0 / (2.0 * tau)

        candidate = policy_center.copy()
        candidate.train()
        for p in candidate.parameters():
            p.requires_grad_(True)

        with torch.no_grad():
            center_actions = policy_center.act(states).detach()

        opt = torch.optim.Adam(candidate.parameters(), lr=self.lr)
        ensemble = critics if isinstance(critics, CriticEnsemble) else None

        # Freeze critic params so proximal grads only update the candidate actor.
        frozen = []
        if ensemble is not None:
            for p in ensemble.parameters():
                frozen.append((p, p.requires_grad))
                p.requires_grad_(False)
        else:
            for c in critics:
                net = getattr(c, "qnet", None)
                if net is None:
                    continue
                for p in net.parameters():
                    frozen.append((p, p.requires_grad))
                    p.requires_grad_(False)

        try:
            for _ in range(self.n_steps):
                actions = candidate.act(states)
                if ensemble is not None:
                    q_stack = ensemble(states, actions)
                    q_val = q_stack.min(dim=0).values if self.use_min_q else q_stack.mean(dim=0)
                else:
                    q_vals = torch.stack([c.q(states, actions) for c in critics], dim=0)
                    q_val = q_vals.min(dim=0).values if self.use_min_q else q_vals.mean(dim=0)

                prox = ((actions - center_actions) ** 2).sum(dim=-1)
                loss = (-q_val + w2_weight * prox).mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(candidate.parameters(), self.max_grad_norm)
                opt.step()
        finally:
            for p, req in frozen:
                p.requires_grad_(req)

        candidate.eval()
        return candidate

    def refine_inplace_soft(
        self,
        actor: Actor,
        center_actions: Tensor,
        states: Tensor,
        critics: CriticEnsemble,
        tau: float,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Single gradient step used inside the online training loop."""
        tau = max(float(tau), 1e-8)
        w2_weight = 1.0 / (2.0 * tau)
        actions = actor.act(states)
        q_stack = critics(states, actions)
        q_val = q_stack.min(dim=0).values if self.use_min_q else q_stack.mean(dim=0)
        prox = ((actions - center_actions.detach()) ** 2).sum(dim=-1)
        loss = (-q_val + w2_weight * prox).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
        optimizer.step()
        return float(loss.detach().item())
