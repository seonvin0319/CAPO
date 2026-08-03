"""Focused correctness tests for the JAX CAPO port."""
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
    calibrated_adaptive_mpi,
    estimate_q_scale,
    pairwise_gain_ensemble,
)
from capo_jax.networks import (  # noqa: E402
    Actor,
    ActorPolicy,
    CriticEnsemble,
    slice_ensemble_params,
)
from capo_jax.refiner import ProximalW2Refiner  # noqa: E402


class ConstantPolicy:
    def __init__(self, bias: float):
        self.bias = float(bias)

    def act(self, states):
        return jnp.full((states.shape[0], 1), self.bias, dtype=jnp.float32)

    def copy(self):
        return ConstantPolicy(self.bias)


class ScaledCritic:
    def __init__(self, scale: float):
        self.scale = float(scale)

    def q(self, states, actions):
        del states
        return self.scale * actions[..., 0]


class VectorCritic:
    def __init__(self, scales):
        self.scales = jnp.asarray(scales, dtype=jnp.float32)
        self.calls = 0

    def q_all(self, states, actions):
        del states
        self.calls += 1
        return self.scales[:, None] * actions[None, :, 0]


class ScriptedRefiner:
    def __init__(self):
        self.calls = []

    def refine(
        self,
        policy_center,
        critics,
        tau,
        states,
        *,
        gen_critic_params=None,
    ):
        del critics, states, gen_critic_params
        self.calls.append(float(tau))
        return ConstantPolicy(policy_center.bias + float(tau))


def test_import_is_lightweight():
    assert "capo_jax.trainer" not in sys.modules


def test_critic_ensemble_shapes_and_param_slicing():
    key = jax.random.PRNGKey(0)
    states = jnp.zeros((8, 3), dtype=jnp.float32)
    actions = jnp.zeros((8, 2), dtype=jnp.float32)
    sampled_actions = jnp.zeros((8, 5, 2), dtype=jnp.float32)

    full = CriticEnsemble(n_critics=4, hidden=16, n_hidden=2)
    params = full.init(key, states, actions)["params"]
    assert full.apply({"params": params}, states, actions).shape == (4, 8)
    assert full.apply({"params": params}, states, sampled_actions).shape == (4, 8, 5)

    subset = CriticEnsemble(n_critics=2, hidden=16, n_hidden=2)
    subset_params = slice_ensemble_params(params, 0, 2)
    expected = full.apply({"params": params}, states, actions)[:2]
    actual = subset.apply({"params": subset_params}, states, actions)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_vectorized_certificate_matches_generic_path():
    states = jnp.zeros((16, 3), dtype=jnp.float32)
    actions = jnp.linspace(-0.5, 0.5, 16, dtype=jnp.float32)[:, None]
    current = ConstantPolicy(0.1)
    candidate = ConstantPolicy(0.3)
    generic = (ScaledCritic(1.0), ScaledCritic(2.0))
    vector = VectorCritic([1.0, 2.0])

    expected = pairwise_gain_ensemble(
        generic, current, candidate, states, q_scale=2.0
    )
    actual = pairwise_gain_ensemble(
        (vector,), current, candidate, states, q_scale=2.0
    )
    np.testing.assert_allclose(actual[2], expected[2], rtol=1e-6, atol=1e-7)
    assert abs(actual[0] - expected[0]) < 1e-7
    assert abs(actual[1] - expected[1]) < 1e-7
    assert vector.calls == 2

    generic_scale = estimate_q_scale(generic, states, actions=actions)
    vector_scale = estimate_q_scale((vector,), states, actions=actions)
    assert abs(generic_scale - vector_scale) < 1e-7


def test_refiner_moves_actor_toward_higher_q():
    key = jax.random.PRNGKey(1)
    states = jnp.ones((32, 3), dtype=jnp.float32)
    actor = Actor(action_dim=1, hidden=16)
    params = actor.init(key, states)["params"]
    actor_apply = lambda p, s: actor.apply({"params": p}, s)

    def critic_apply(params, s, actions):
        del params, s
        return jnp.stack([actions[:, 0], 2.0 * actions[:, 0]], axis=0)

    center = ActorPolicy(params, actor_apply)
    refiner = ProximalW2Refiner(
        lr=1e-3,
        n_steps=2,
        actor_apply=actor_apply,
        critic_apply=critic_apply,
    )
    candidate = refiner.refine(
        center,
        critics=(),
        tau=0.01,
        states=states,
        gen_critic_params={},
    )
    compiled = refiner._compiled_refine
    refiner.refine(
        center,
        critics=(),
        tau=0.02,
        states=states,
        gen_critic_params={},
    )
    assert refiner._compiled_refine is compiled
    before = np.asarray(center.act(states)).mean()
    after = np.asarray(candidate.act(states)).mean()
    assert np.isfinite(after)
    assert after > before


def test_jax_pilot_adaptive_controller():
    states = jnp.zeros((16, 3), dtype=jnp.float32)
    refiner = ScriptedRefiner()
    cfg = CAPOConfig(
        n_max=1,
        tau_controller="pilot_adaptive",
        tau_pilot_initial=0.01,
        tau_min=0.001,
        tau_max=0.05,
        target_action_mse=0.0025,
        max_action_mse=1.0,
        beta_uncertainty=0.0,
        shift_penalty_coef=0.0,
        data_penalty_coef=0.0,
        accept_margin=-1e9,
    )
    controller = [
        {
            "previous_selected_tau": None,
            "tau": 0.01,
            "action_mse": 0.0025,
        }
    ]
    result = calibrated_adaptive_mpi(
        base_policy=ConstantPolicy(0.0),
        refiner=refiner,
        states=states,
        cfg=cfg,
        critics=(ScaledCritic(1.0),),
        tau_controller_state=controller,
    )
    assert refiner.calls[0] == 0.01
    assert len(refiner.calls) <= 2
    assert result.selected_n == 1
    assert controller[0]["previous_selected_tau"] is not None


if __name__ == "__main__":
    tests = [
        test_import_is_lightweight,
        test_critic_ensemble_shapes_and_param_slicing,
        test_vectorized_certificate_matches_generic_path,
        test_refiner_moves_actor_toward_higher_q,
        test_jax_pilot_adaptive_controller,
    ]
    for test in tests:
        test()
        print(f"OK {test.__name__}")
    print("ALL JAX TESTS PASSED")
