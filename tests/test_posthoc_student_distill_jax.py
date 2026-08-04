"""Correctness tests for persistent JAX student distillation."""
from __future__ import annotations

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from capo_jax.posthoc_student_distill import (
    discover_checkpoints,
    distill_loss_pure_bc,
    load_student_params,
    parameter_l2_distance,
)
def test_discover_jax_checkpoints_sorted_and_complete(tmp_path: Path):
    for step in (200_000, 100_000, 150_000, 695_000):
        (tmp_path / f"checkpoint_{step}.pkl").write_bytes(b"x")
    checkpoints = discover_checkpoints(
        tmp_path, "checkpoint_*.pkl", 100_000, 200_000, 50_000
    )
    assert [step for step, _ in checkpoints] == [100_000, 150_000, 200_000]
    with pytest.raises(FileNotFoundError, match="missing"):
        discover_checkpoints(
            tmp_path, "checkpoint_*.pkl", 100_000, 250_000, 50_000
        )


def test_pure_bc_stops_student_gradient():
    pred = jnp.asarray([[0.0, 1.0], [2.0, 3.0]])
    target = jnp.asarray([[1.0, 1.0], [0.0, 1.0]])
    pred_grad, target_grad = jax.grad(
        lambda p, t: distill_loss_pure_bc(p, t), argnums=(0, 1)
    )(pred, target)
    np.testing.assert_allclose(
        distill_loss_pure_bc(pred, target), jnp.mean((pred - target) ** 2)
    )
    assert np.any(np.asarray(pred_grad) != 0.0)
    np.testing.assert_array_equal(target_grad, jnp.zeros_like(target))


def test_jax_checkpoint_load_and_tree_distance(tmp_path: Path):
    params = {"Dense_0": {"kernel": np.ones((3, 2), dtype=np.float32)}}
    path = tmp_path / "checkpoint_100000.pkl"
    with open(path, "wb") as stream:
        pickle.dump({"actor": params, "step": 100_000}, stream)
    loaded = load_student_params(path, jax.devices("cpu")[0])
    assert parameter_l2_distance(loaded, loaded) == 0.0
    np.testing.assert_array_equal(
        loaded["Dense_0"]["kernel"], params["Dense_0"]["kernel"]
    )
