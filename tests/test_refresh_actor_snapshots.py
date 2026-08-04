from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from capo_jax.networks import Actor
from capo_jax.refresh_snapshots import (
    actor_params_for_role, load_refresh_actor_bundle, save_refresh_actor_bundle,
)


@dataclass
class FakeConfig:
    run_id: str = "snapshot_test"
    env: str = "hopper-medium-v2"
    seed: int = 0
    hidden: int = 16
    normalize: bool = True


def test_refresh_snapshot_roundtrip_deduplicates_and_reproduces_actions(tmp_path):
    actor = Actor(action_dim=2, hidden=16)
    states = jnp.arange(15, dtype=jnp.float32).reshape(5, 3) / 10
    params = actor.init(jax.random.PRNGKey(7), states)["params"]
    trainer = SimpleNamespace(
        cfg=FakeConfig(), state_dim=3, action_dim=2, max_action=1.0,
        stats=SimpleNamespace(
            state_mean=np.zeros(3, dtype=np.float32),
            state_std=np.ones(3, dtype=np.float32),
        ),
    )
    key_before = np.asarray(jax.random.PRNGKey(123))
    path = save_refresh_actor_bundle(
        run_dir=tmp_path, trainer=trainer, refresh_step=100,
        roles={
            "student_before_refresh": params,
            "active_teacher_after_gate": params,
            "newly_generated_challenger": None,
        },
        refresh_row={
            "N_star": 0, "selected_tau_per_ladder_step": [],
            "gate_action": "keep_old", "previous_teacher_state": "active",
            "next_teacher_state": "active",
        },
    )
    bundle = load_refresh_actor_bundle(path)
    assert len(bundle["actors"]) == 1
    assert bundle["roles"]["student_before_refresh"] == bundle["roles"]["active_teacher_after_gate"]
    restored = actor_params_for_role(bundle, "student_before_refresh")
    expected = actor.apply({"params": params}, states)
    actual = actor.apply({"params": restored}, states)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    np.testing.assert_array_equal(np.asarray(jax.random.PRNGKey(123)), key_before)
    assert (path.parent / "metadata.json").exists()
