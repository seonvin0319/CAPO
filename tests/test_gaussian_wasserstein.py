"""Tests for Gaussian actor + Wasserstein distance path."""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capo_jax.core import (  # noqa: E402
    CAPOConfig,
    candidate_certificate,
    policy_action_mse,
    policy_wasserstein_sq,
)
from capo_jax.networks import Actor, ActorPolicy, GaussianActor  # noqa: E402
from capo_jax.refiner import ProximalW2Refiner  # noqa: E402
from capo_jax.trainer import TrainConfig  # noqa: E402


class ScaledCritic:
    def __init__(self, scale: float):
        self.scale = float(scale)

    def q(self, states, actions):
        del states
        return self.scale * actions[..., 0]


def test_auto_distance_metric_resolves():
    cfg = TrainConfig(actor_type="gaussian", distance_metric="auto")
    assert cfg.distance_metric == "wasserstein"
    cfg2 = TrainConfig(actor_type="deterministic", distance_metric="auto")
    assert cfg2.distance_metric == "amse"


def test_wasserstein_matches_amse_for_dirac():
    key = jax.random.PRNGKey(0)
    actor = Actor(action_dim=2, max_action=1.0)
    s = jnp.zeros((8, 3), dtype=jnp.float32)
    p0 = actor.init(key, s)["params"]
    p1 = jax.tree_util.tree_map(lambda x: x + 0.01, p0)

    def apply(params, states):
        return actor.apply({"params": params}, states)

    pol0 = ActorPolicy(p0, apply)
    pol1 = ActorPolicy(p1, apply)
    amse = policy_action_mse(pol0, pol1, s)
    w2 = policy_wasserstein_sq(pol0, pol1, s)
    assert abs(amse - w2) < 1e-6


def test_wasserstein_includes_std():
    class DistPolicy:
        def __init__(self, mean, std):
            self._mean = np.asarray(mean, dtype=np.float64)
            self._std = np.asarray(std, dtype=np.float64)

        def act(self, states):
            return np.broadcast_to(self._mean, (states.shape[0], self._mean.shape[-1]))

        def dist_params(self, states):
            b = states.shape[0]
            return (
                np.broadcast_to(self._mean, (b, self._mean.shape[-1])),
                np.broadcast_to(self._std, (b, self._std.shape[-1])),
            )

        def copy(self):
            return DistPolicy(self._mean, self._std)

    states = np.zeros((4, 2), dtype=np.float64)
    p0 = DistPolicy([0.0, 0.0], [0.1, 0.1])
    p1 = DistPolicy([0.0, 0.0], [0.5, 0.5])
    # means equal → AMSE=0, W2^2 = 2*(0.4^2)=0.32
    assert policy_action_mse(p0, p1, states) < 1e-12
    w2 = policy_wasserstein_sq(p0, p1, states)
    assert abs(w2 - 0.32) < 1e-6

    cfg = CAPOConfig(distance_metric="wasserstein", max_action_mse=None, data_penalty_coef=0.0)
    stats = candidate_certificate(
        [ScaledCritic(1.0)], p0, p1, states, tau=0.01, cfg=cfg, q_scale=1.0
    )
    assert abs(stats.action_mse - 0.32) < 1e-6


def test_gaussian_actor_refine_runs():
    key = jax.random.PRNGKey(1)
    actor = GaussianActor(action_dim=2, max_action=1.0)
    from capo_jax.networks import CriticEnsemble

    critic = CriticEnsemble(n_critics=2, hidden=32)
    s = jnp.zeros((16, 3), dtype=jnp.float32)
    a = jnp.zeros((16, 2), dtype=jnp.float32)
    k1, k2 = jax.random.split(key)
    actor_params = actor.init(k1, s)["params"]
    critic_params = critic.init(k2, s, a)["params"]

    def actor_apply(params, states):
        return actor.apply({"params": params}, states)

    def actor_dist_apply(params, states):
        return actor.apply({"params": params}, states, return_dist=True)

    def critic_apply(params, states, actions):
        return critic.apply({"params": params}, states, actions)

    refiner = ProximalW2Refiner(
        lr=3e-4,
        n_steps=2,
        actor_apply=actor_apply,
        critic_apply=critic_apply,
        actor_dist_apply=actor_dist_apply,
        use_wasserstein=True,
    )
    center = ActorPolicy(actor_params, actor_apply, dist_fn=actor_dist_apply)
    out = refiner.refine(
        center, critics=[], tau=0.01, states=s, gen_critic_params=critic_params
    )
    mean, std = out.dist_params(s)
    assert mean.shape == (16, 2)
    assert std.shape == (16, 2)
    assert float(std.min()) > 0.0


if __name__ == "__main__":
    test_auto_distance_metric_resolves()
    test_wasserstein_matches_amse_for_dirac()
    test_wasserstein_includes_std()
    test_gaussian_actor_refine_runs()
    print("ok")
