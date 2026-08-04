from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from capo_jax.experiment_manifests import baseline_manifest, broad_manifest, legacy_manifest
from capo_jax.gate import decide_replacement_gate
from capo_jax.method_comparison import assert_compatible
from capo_jax.td3_objectives import td3bc_legacy_actor_components


def test_nstar_zero_modes_differ_exactly():
    common = dict(
        teacher_state="active", has_new=False,
        student_to_new=float("nan"), student_to_existing=-1.0,
        existing_to_new=float("nan"), margin=0.0, stale_action="disable",
        nstar_zero=True,
    )
    legacy = decide_replacement_gate(**common, nstar_zero_action="legacy_hold")
    current = decide_replacement_gate(**common, nstar_zero_action="revalidate_current")
    assert legacy.gate_action == "keep_old" and legacy.next_teacher_state == "active"
    assert current.gate_action == "remain_inactive" and current.next_teacher_state == "disabled"


def test_legacy_stale_replaces_new():
    decision = decide_replacement_gate(
        teacher_state="active", has_new=True,
        student_to_new=0.1, student_to_existing=-0.1, existing_to_new=-0.1,
        margin=0.0, stale_action="replace_new",
    )
    assert decision.gate_action == "replace_new"
    assert decision.activate_new


def test_revised_and_reference_manifests():
    broad = broad_manifest()
    baseline = baseline_manifest()
    legacy = legacy_manifest()
    assert len(broad) == 324 and len({row["run_id"] for row in broad}) == 324
    assert sum(
        1 for row in broad if row["config"]["stale_incumbent_action"] == "replace_new"
    ) == 108
    assert all(row["config"]["paired_eval_episodes"] == 0 for row in broad)
    assert all(row["config"]["save_refresh_actors"] is True for row in broad)
    assert all(row["config"]["nstar_zero_action"] == "revalidate_current" for row in broad)
    assert len(baseline) == len(legacy) == 9
    assert all(row["config"]["td3_actor_objective"] == "td3bc_legacy" for row in baseline)
    assert all(row["config"]["stale_incumbent_action"] == "replace_new" for row in legacy)
    all_ids = [row["run_id"] for row in broad + baseline + legacy]
    assert len(all_ids) == len(set(all_ids))


def test_original_td3bc_objective_uses_mean_four_q_and_unit_bc():
    q = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    pi = jnp.asarray([[0.0], [1.0]])
    actions = jnp.asarray([[1.0], [1.0]])
    out = td3bc_legacy_actor_components(q, pi, actions, alpha=2.5, eps=0.0)
    expected_q = jnp.asarray([4.0, 5.0])
    assert jnp.all(out["q_values"] == expected_q)
    assert math.isclose(float(out["q_scale"]), 4.5)
    assert math.isclose(float(out["data_bc_loss"]), 0.5)
    assert math.isclose(float(out["total_actor_loss"]), -2.0, rel_tol=1e-6)


def test_analysis_refuses_incompatible_baseline_metadata():
    capo = broad_manifest()[0]["config"]
    baseline = baseline_manifest()[0]["config"]
    assert_compatible(capo, baseline)
    incompatible = {**baseline, "normalize_reward": False}
    with pytest.raises(ValueError, match="normalize_reward"):
        assert_compatible(capo, incompatible)
