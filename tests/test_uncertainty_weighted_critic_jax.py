"""Focused checks for CAPO JAX uncertainty-weighted TD3 critic loss."""
from __future__ import annotations

from pathlib import Path
import sys
import types

# The focused loss tests do not need D4RL environments. Avoid importing
# mujoco_py (and compiling its system GL bindings) as a trainer import side effect.
sys.modules.setdefault("d4rl", types.ModuleType("d4rl"))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _loss(
    q_pred,
    target,
    q_next,
    *,
    enabled,
    bootstrap_mask=None,
    kappa=1.0,
    normalization="none",
    eps=1e-6,
):
    # Keep this import local: the JAX-port suite intentionally checks that
    # importing networks/core does not transitively import the trainer.
    from capo_jax.trainer import _td3_critic_td_loss

    if bootstrap_mask is None:
        bootstrap_mask = jnp.ones_like(target)
    return _td3_critic_td_loss(
        q_pred,
        target,
        q_next,
        bootstrap_mask,
        use_uncertainty_weighted_critic=enabled,
        kappa=kappa,
        eps=eps,
        min_weight=0.0,
        weight_normalization=normalization,
    )


@pytest.mark.parametrize("normalization", ["none", "batch_mean"])
def test_disabled_and_zero_kappa_match_original_td_loss_and_gradient_exactly(
    normalization,
):
    q_pred = jnp.array([[1.0, 3.0], [2.0, 5.0]], dtype=jnp.float32)
    target = jnp.array([1.5, 4.0], dtype=jnp.float32)
    q_next = jnp.array([[1.0, 1.0], [3.0, 5.0]], dtype=jnp.float32)
    original = jnp.mean((q_pred - target[None, :]) ** 2)

    disabled, _ = _loss(
        q_pred, target, q_next, enabled=False, normalization=normalization
    )
    zero_kappa, zero_stats = _loss(
        q_pred,
        target,
        q_next,
        enabled=True,
        kappa=0.0,
        normalization=normalization,
    )

    np.testing.assert_array_equal(disabled, original)
    np.testing.assert_array_equal(zero_kappa, original)
    np.testing.assert_array_equal(zero_stats["critic/uncertainty_weight_mean"], 1.0)
    np.testing.assert_array_equal(zero_stats["critic/uncertainty_weight_min"], 1.0)
    np.testing.assert_array_equal(zero_stats["critic/uncertainty_weight_max"], 1.0)
    original_grad = jax.grad(lambda q: jnp.mean((q - target[None, :]) ** 2))(q_pred)
    disabled_grad = jax.grad(
        lambda q: _loss(
            q, target, q_next, enabled=False, normalization=normalization
        )[0]
    )(q_pred)
    zero_kappa_grad = jax.grad(
        lambda q: _loss(
            q,
            target,
            q_next,
            enabled=True,
            kappa=0.0,
            normalization=normalization,
        )[0]
    )(q_pred)
    np.testing.assert_array_equal(disabled_grad, original_grad)
    np.testing.assert_array_equal(zero_kappa_grad, original_grad)


def test_zero_and_increasing_uncertainty_produce_expected_raw_weights():
    q_pred = jnp.zeros((2, 2), dtype=jnp.float32)
    target = jnp.zeros((2,), dtype=jnp.float32)
    no_disagreement = jnp.ones((2, 2), dtype=jnp.float32)
    increasing_disagreement = jnp.array([[1.0, 1.0], [1.0, 3.0]], dtype=jnp.float32)

    _, zero_stats = _loss(q_pred, target, no_disagreement, enabled=True)
    _, uncertain_stats = _loss(q_pred, target, increasing_disagreement, enabled=True)

    np.testing.assert_array_equal(
        zero_stats["critic/raw_uncertainty_weight_min"], 1.0
    )
    assert float(uncertain_stats["critic/raw_uncertainty_weight_mean"]) < 1.0
    assert float(uncertain_stats["critic/raw_uncertainty_weight_min"]) < 1.0


def test_terminal_transition_ignores_even_large_target_disagreement():
    q_pred = jnp.array([[2.0], [4.0]], dtype=jnp.float32)
    target = jnp.array([1.0], dtype=jnp.float32)
    q_next = jnp.array([[-1_000.0], [1_000.0]], dtype=jnp.float32)

    loss, stats = _loss(
        q_pred,
        target,
        q_next,
        enabled=True,
        bootstrap_mask=jnp.array([0.0], dtype=jnp.float32),
        normalization="batch_mean",
    )
    original = jnp.mean((q_pred - target[None, :]) ** 2)

    assert float(stats["critic/uncertainty_mean"]) > 0.0
    np.testing.assert_array_equal(stats["critic/effective_uncertainty_mean"], 0.0)
    np.testing.assert_array_equal(stats["critic/raw_uncertainty_weight_mean"], 1.0)
    np.testing.assert_array_equal(stats["critic/uncertainty_weight_mean"], 1.0)
    np.testing.assert_array_equal(loss, original)


def test_none_normalization_reproduces_original_uwc_formula():
    q_pred = jnp.array([[1.0, 3.0], [2.0, 5.0]], dtype=jnp.float32)
    target = jnp.array([1.5, 4.0], dtype=jnp.float32)
    q_next = jnp.array([[1.0, 1.0], [3.0, 5.0]], dtype=jnp.float32)
    uncertainty = jnp.std(q_next, axis=0)
    scale = jnp.mean(jnp.abs(q_next), axis=0) + 1e-6
    expected_weight = 1.0 / (1.0 + uncertainty / scale)
    expected_loss = jnp.mean(expected_weight[None, :] * (q_pred - target[None, :]) ** 2)

    loss, stats = _loss(
        q_pred, target, q_next, enabled=True, normalization="none"
    )

    np.testing.assert_allclose(loss, expected_loss, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        stats["critic/uncertainty_weight_mean"], expected_weight.mean(),
        rtol=0.0, atol=0.0,
    )


def test_batch_mean_normalization_preserves_nonterminal_mean_weight():
    q_pred = jnp.ones((2, 3), dtype=jnp.float32)
    target = jnp.zeros((3,), dtype=jnp.float32)
    q_next = jnp.array([[1.0, 1.0, -4.0], [1.0, 3.0, 8.0]], dtype=jnp.float32)

    loss, stats = _loss(
        q_pred, target, q_next, enabled=True, normalization="batch_mean"
    )

    np.testing.assert_allclose(loss, 1.0, rtol=0.0, atol=3e-6)
    np.testing.assert_allclose(
        stats["critic/uncertainty_weight_mean"], 1.0, rtol=0.0, atol=3e-6
    )
    assert float(stats["critic/uncertainty_weight_max"]) > 1.0


def test_mixed_batch_keeps_terminal_weight_one_and_nonterminal_mean_one():
    # Unit squared TD error on non-terminals and zero error on the terminal lets
    # the weighted loss directly expose the non-terminal final-weight mean.
    q_pred = jnp.array([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=jnp.float32)
    target = jnp.zeros((3,), dtype=jnp.float32)
    q_next = jnp.array([[1.0, -1_000.0, -4.0], [1.0, 1_000.0, 8.0]], dtype=jnp.float32)
    bootstrap_mask = jnp.array([1.0, 0.0, 1.0], dtype=jnp.float32)

    loss, stats = _loss(
        q_pred,
        target,
        q_next,
        enabled=True,
        bootstrap_mask=bootstrap_mask,
        normalization="batch_mean",
    )

    # loss = sum(two non-terminal weights) / (2 critics * 3 samples) * 2 critics
    np.testing.assert_allclose(loss, 2.0 / 3.0, rtol=0.0, atol=3e-6)
    np.testing.assert_allclose(
        stats["critic/bootstrap_fraction"], 2.0 / 3.0, rtol=0.0, atol=1e-7
    )
    assert float(stats["critic/uncertainty_weight_max"]) > 1.0

    terminal_error_only = jnp.array(
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=jnp.float32
    )
    terminal_loss, _ = _loss(
        terminal_error_only,
        target,
        q_next,
        enabled=True,
        bootstrap_mask=bootstrap_mask,
        normalization="batch_mean",
    )
    np.testing.assert_array_equal(
        terminal_loss, jnp.asarray(1.0 / 3.0, dtype=jnp.float32)
    )


def test_weight_path_is_detached_and_jittable_vmappable():
    q_pred = jnp.array([[1.0, 3.0], [2.0, 5.0]], dtype=jnp.float32)
    target = jnp.array([1.5, 4.0], dtype=jnp.float32)
    q_next = jnp.array([[1.0, 1.0], [3.0, 5.0]], dtype=jnp.float32)
    bootstrap_mask = jnp.array([1.0, 0.0], dtype=jnp.float32)

    def loss_from_target_ensemble(next_ensemble):
        return _loss(
            q_pred,
            target,
            next_ensemble,
            enabled=True,
            bootstrap_mask=bootstrap_mask,
            normalization="batch_mean",
        )[0]

    np.testing.assert_array_equal(
        jax.grad(loss_from_target_ensemble)(q_next), jnp.zeros_like(q_next)
    )

    compiled = jax.jit(
        lambda qp, t, qn, bm: _loss(
            qp,
            t,
            qn,
            enabled=True,
            bootstrap_mask=bm,
            normalization="batch_mean",
        )[0]
    )
    assert jnp.isfinite(compiled(q_pred, target, q_next, bootstrap_mask))
    vmapped = jax.vmap(compiled)(
        jnp.stack([q_pred, q_pred]),
        jnp.stack([target, target]),
        jnp.stack([q_next, q_next]),
        jnp.stack([bootstrap_mask, bootstrap_mask]),
    )
    assert vmapped.shape == (2,)


def test_two_critic_uncertainty_uses_the_entire_target_ensemble():
    q_pred = jnp.zeros((2, 1), dtype=jnp.float32)
    target = jnp.zeros((1,), dtype=jnp.float32)
    q_next = jnp.array([[-3.0], [5.0]], dtype=jnp.float32)

    _, stats = _loss(q_pred, target, q_next, enabled=True)

    np.testing.assert_allclose(stats["critic/uncertainty_mean"], 4.0)
    assert float(stats["critic/raw_uncertainty_weight_mean"]) < 1.0


def test_new_config_defaults_and_validation():
    from capo_jax.trainer import TrainConfig

    cfg = TrainConfig()
    assert cfg.use_uncertainty_weighted_critic is False
    assert cfg.critic_uncertainty_kappa == 1.0
    assert cfg.critic_uncertainty_eps == 1e-6
    assert cfg.critic_uncertainty_min_weight == 0.0
    assert cfg.critic_uncertainty_weight_normalization == "none"
    with pytest.raises(ValueError, match="critic_uncertainty_weight_normalization"):
        TrainConfig(critic_uncertainty_weight_normalization="invalid")
