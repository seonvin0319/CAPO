"""Flax actor / critic / value networks for CAPO on D4RL."""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


class Actor(nn.Module):
    """Deterministic tanh-squashed actor (TD3+BC / IQL-det / CQL-det)."""

    action_dim: int
    max_action: float = 1.0
    hidden: int = 256

    @nn.compact
    def __call__(self, state: jnp.ndarray, return_dist: bool = False):
        x = nn.Dense(self.hidden)(state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        mean = self.max_action * jnp.tanh(x)
        if return_dist:
            # Dirac: std=0 so Wasserstein collapses to action MSE on means.
            return mean, jnp.zeros_like(mean)
        return mean


class GaussianActor(nn.Module):
    """Diagonal Gaussian actor with tanh-squashed mean (action-space std).

    ``act`` / default forward returns the mean. ``return_dist=True`` returns
    ``(mean, std)`` for closed-form W2 between diagonal Gaussians:
    W2^2 = ||μ−μ'||^2 + ||σ−σ'||^2.
    """

    action_dim: int
    max_action: float = 1.0
    hidden: int = 256
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, state: jnp.ndarray, return_dist: bool = False):
        x = nn.Dense(self.hidden)(state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden)(x)
        x = nn.relu(x)
        mean = self.max_action * jnp.tanh(nn.Dense(self.action_dim)(x))
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        std = jnp.exp(log_std)
        if return_dist:
            return mean, std
        return mean


class QNet(nn.Module):
    hidden: int = 256
    n_hidden: int = 2

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([state, action], axis=-1)
        for _ in range(self.n_hidden):
            x = nn.Dense(self.hidden)(x)
            x = nn.relu(x)
        return nn.Dense(1)(x).squeeze(-1)


class ValueFunction(nn.Module):
    hidden: int = 256
    n_hidden: int = 2

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        x = state
        for _ in range(self.n_hidden):
            x = nn.Dense(self.hidden)(x)
            x = nn.relu(x)
        return nn.Dense(1)(x).squeeze(-1)


class CriticEnsemble(nn.Module):
    """Independent Q ensemble; outputs [M, B] or [M, B, K]."""

    n_critics: int = 4
    hidden: int = 256
    n_hidden: int = 2

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        Ensemble = nn.vmap(
            QNet,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.n_critics,
            in_axes=(None, None),
            out_axes=0,
        )
        # actions: [B, K, A] → flatten for shared apply, then reshape
        if action.ndim == state.ndim + 1:
            b, k, a = action.shape
            s = jnp.broadcast_to(state[:, None, :], (b, k, state.shape[-1])).reshape(b * k, -1)
            act = action.reshape(b * k, a)
            q = Ensemble(hidden=self.hidden, n_hidden=self.n_hidden)(s, act)  # [M, B*K]
            return q.reshape(self.n_critics, b, k)
        return Ensemble(hidden=self.hidden, n_hidden=self.n_hidden)(state, action)


def q_mean(q_stack: jnp.ndarray) -> jnp.ndarray:
    return q_stack.mean(axis=0)


def q_min(q_stack: jnp.ndarray) -> jnp.ndarray:
    return q_stack.min(axis=0)


def slice_ensemble_params(params: Any, start: int, end: int) -> Any:
    """Slice critic ensemble params along the vmap axis (critics)."""
    return jax.tree_util.tree_map(lambda x: x[start:end], params)


class ActorPolicy:
    """Policy protocol adapter: params + apply_fn (used by CAPO core)."""

    def __init__(
        self,
        params: Any,
        apply_fn: Callable[[Any, jnp.ndarray], jnp.ndarray],
        dist_fn: Optional[
            Callable[[Any, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]
        ] = None,
    ):
        self.params = params
        self.apply_fn = apply_fn
        self.dist_fn = dist_fn

    def act(self, states: jnp.ndarray) -> jnp.ndarray:
        return self.apply_fn(self.params, states)

    def dist_params(self, states: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.dist_fn is not None:
            return self.dist_fn(self.params, states)
        mean = self.act(states)
        return mean, jnp.zeros_like(mean)

    def copy(self) -> "ActorPolicy":
        return ActorPolicy(
            params=jax.tree_util.tree_map(lambda x: x, self.params),
            apply_fn=self.apply_fn,
            dist_fn=self.dist_fn,
        )


class CriticAdapter:
    """Single-critic adapter for certificate scoring."""

    def __init__(
        self,
        params: Any,
        apply_fn: Callable[[Any, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        index: Optional[int] = None,
    ):
        self.params = params
        self.apply_fn = apply_fn
        self.index = index

    def q(self, states: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        q_stack = self.apply_fn(self.params, states, actions)
        if self.index is None:
            if q_stack.ndim >= 2 and q_stack.shape[0] == 1:
                return q_stack[0]
            return q_stack
        return q_stack[self.index]


class CriticEnsembleAdapter:
    """Vectorized critic adapter used by certificates.

    Keeping the selected critics together avoids evaluating the complete
    ensemble once per critic when the certificate needs all members anyway.
    """

    def __init__(
        self,
        params: Any,
        apply_fn: Callable[[Any, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        start: int = 0,
        end: Optional[int] = None,
    ):
        self.params = params
        self.apply_fn = apply_fn
        self.start = int(start)
        self.end = end

    def q_all(self, states: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        return self.apply_fn(self.params, states, actions)[self.start : self.end]

    def q(self, states: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        return self.q_all(states, actions)


def make_critic_adapters(
    critic_params: Any,
    critic_apply: Callable,
    n_critics: int,
) -> Tuple[CriticAdapter, ...]:
    return tuple(
        CriticAdapter(critic_params, critic_apply, index=i) for i in range(n_critics)
    )
