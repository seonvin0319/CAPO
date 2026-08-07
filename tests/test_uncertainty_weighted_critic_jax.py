"""Focused checks for CAPO JAX uncertainty-weighted TD3 critic loss."""
from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _loss(q_pred, target, q_next, *, enabled, kappa=1.0):
    # Keep this import local: the JAX-port suite intentionally checks that
    # importing networks/core does not transitively import the trainer.
    from capo_jax.trainer import _td3_critic_td_loss

    return _td3_critic_td_loss(
        q_pred,
        target,
        q_next,
        use_uncertainty_weighted_critic=enabled,
        kappa=kappa,
        eps=1e-6,
        min_weight=0.0,
    )


def test_disabled_and_zero_kappa_match_original_td_loss_exactly():
    q_pred = jnp.array([[1.0, 3.0], [2.0, 5.0]], dtype=jnp.float32)
    target = jnp.array([1.5, 4.0], dtype=jnp.float32)
    q_next = jnp.array([[1.0, 1.0], [3.0, 5.0]], dtype=jnp.float32)
    original = jnp.mean((q_pred - target[None, :]) ** 2)

    disabled, _ = _loss(q_pred, target, q_next, enabled=False)
    zero_kappa, zero_stats = _loss(q_pred, target, q_next, enabled=True, kappa=0.0)

    np.testing.assert_array_equal(disabled, original)
    np.testing.assert_array_equal(zero_kappa, original)
    np.testing.assert_array_equal(zero_stats["critic/uncertainty_weight_mean"], 1.0)
    original_grad = jax.grad(lambda q: jnp.mean((q - target[None, :]) ** 2))(q_pred)
    disabled_grad = jax.grad(lambda q: _loss(q, target, q_next, enabled=False)[0])(q_pred)
    zero_kappa_grad = jax.grad(
        lambda q: _loss(q, target, q_next, enabled=True, kappa=0.0)[0]
    )(q_pred)
    np.testing.assert_array_equal(disabled_grad, original_grad)
    np.testing.assert_array_equal(zero_kappa_grad, original_grad)


def test_zero_and_increasing_uncertainty_produce_expected_weights():
    q_pred = jnp.zeros((2, 2), dtype=jnp.float32)
    target = jnp.zeros((2,), dtype=jnp.float32)
    no_disagreement = jnp.ones((2, 2), dtype=jnp.float32)
    increasing_disagreement = jnp.array([[1.0, 1.0], [1.0, 3.0]], dtype=jnp.float32)

    _, zero_stats = _loss(q_pred, target, no_disagreement, enabled=True)
    _, uncertain_stats = _loss(q_pred, target, increasing_disagreement, enabled=True)

    np.testing.assert_array_equal(zero_stats["critic/uncertainty_weight_min"], 1.0)
    assert float(uncertain_stats["critic/uncertainty_weight_mean"]) < 1.0
    assert float(uncertain_stats["critic/uncertainty_weight_min"]) < 1.0


def test_weight_path_is_detached_and_jittable_vmappable():
    q_pred = jnp.array([[1.0, 3.0], [2.0, 5.0]], dtype=jnp.float32)
    target = jnp.array([1.5, 4.0], dtype=jnp.float32)
    q_next = jnp.array([[1.0, 1.0], [3.0, 5.0]], dtype=jnp.float32)

    def loss_from_target_ensemble(next_ensemble):
        return _loss(q_pred, target, next_ensemble, enabled=True)[0]

    np.testing.assert_array_equal(
        jax.grad(loss_from_target_ensemble)(q_next), jnp.zeros_like(q_next)
    )

    compiled = jax.jit(lambda qp, t, qn: _loss(qp, t, qn, enabled=True)[0])
    assert jnp.isfinite(compiled(q_pred, target, q_next))
    vmapped = jax.vmap(compiled)(
        jnp.stack([q_pred, q_pred]),
        jnp.stack([target, target]),
        jnp.stack([q_next, q_next]),
    )
    assert vmapped.shape == (2,)


def test_new_config_defaults_preserve_checkpoint_compatible_shape():
    from capo_jax.trainer import TrainConfig

    cfg = TrainConfig()
    assert cfg.use_uncertainty_weighted_critic is False
    assert cfg.critic_uncertainty_kappa == 1.0
    assert cfg.critic_uncertainty_eps == 1e-6
    assert cfg.critic_uncertainty_min_weight == 0.0
