"""Deterministic safety and manifest tests for the CAPO stability sweep."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from capo_jax.gate import decide_replacement_gate, teacher_bc_components
from capo_jax.stability_analysis import compute_curve_metrics
from capo_jax.stability_sweep import (
    ENVIRONMENTS,
    FACTORS,
    deterministic_rng_seed,
    generate_manifest,
    sweep_configurations,
)
from scripts.run_capo_stability_sweep import run_status


def test_exact_factorial_and_seed0_manifest():
    configs = sweep_configurations()
    rows = generate_manifest(seeds=(0,))
    assert len(configs) == 36
    assert len(ENVIRONMENTS) == 9
    assert set(row["environment"] for row in rows) == set(ENVIRONMENTS)
    assert len(rows) == 324
    assert len({row["run_id"] for row in rows}) == 324
    assert len({row["config_id"] for row in rows}) == 36
    assert tuple(FACTORS["lambda_T"]) == (0.0, 0.5, 1.0)
    assert tuple(FACTORS["stale_incumbent_action"]) == (
        "disable", "quarantine", "replace_new",
    )


def test_run_ids_and_rng_streams_are_stable_and_distinct():
    rows_a = generate_manifest(seeds=(0,))
    rows_b = generate_manifest(seeds=(0,))
    assert [row["run_id"] for row in rows_a] == [row["run_id"] for row in rows_b]
    assert [row["rng_seed"] for row in rows_a] == [row["rng_seed"] for row in rows_b]
    assert len({row["rng_seed"] for row in rows_a}) == len(rows_a)
    first = rows_a[0]
    assert first["rng_seed"] == deterministic_rng_seed(
        first["environment"], first["seed"], first["config_id"]
    )


def test_lambda_zero_and_quarantine_never_apply_teacher_bc():
    student = jnp.asarray([[0.0], [1.0]])
    teacher = jnp.asarray([[1.0], [0.0]])
    q_student = jnp.zeros((2, 2))
    q_teacher = jnp.ones((2, 2))
    zero_lambda = teacher_bc_components(
        student, teacher, q_student, q_teacher,
        teacher_active=1.0, lambda_t=0.0, beta_uncertainty=0.75,
        mode="uniform",
    )
    quarantined = teacher_bc_components(
        student, teacher, q_student, q_teacher,
        teacher_active=0.0, lambda_t=1.0, beta_uncertainty=0.75,
        mode="uniform",
    )
    assert float(zero_lambda["teacher_bc_loss_weighted"]) == 0.0
    assert float(quarantined["teacher_bc_loss_weighted"]) == 0.0


def test_replacement_margin_is_strict_and_consistent():
    at_margin = decide_replacement_gate(
        teacher_state="active", has_new=True,
        student_to_new=0.001, student_to_existing=0.001,
        existing_to_new=0.001, margin=0.001, stale_action="disable",
    )
    above_margin = decide_replacement_gate(
        teacher_state="active", has_new=True,
        student_to_new=0.0011, student_to_existing=0.0011,
        existing_to_new=0.0011, margin=0.001, stale_action="disable",
    )
    assert not at_margin.valid_new and not at_margin.valid_old
    assert not at_margin.new_beats_old
    assert above_margin.valid_new and above_margin.valid_old
    assert above_margin.new_beats_old
    # The pure replacement function has no ladder-acceptance parameter.
    assert "accept_margin" not in inspect.signature(decide_replacement_gate).parameters


def test_stale_disable_and_quarantine_semantics():
    common = dict(
        teacher_state="active", has_new=True, student_to_new=0.2,
        student_to_existing=-0.1, existing_to_new=0.3, margin=0.0,
    )
    disabled = decide_replacement_gate(**common, stale_action="disable")
    quarantined = decide_replacement_gate(**common, stale_action="quarantine")
    replaced = decide_replacement_gate(**common, stale_action="replace_new")
    kept = decide_replacement_gate(**common, stale_action="keep_old")
    assert disabled.gate_action == "stale_disable"
    assert disabled.next_teacher_state == "disabled"
    assert not disabled.activate_new and not disabled.preserve_quarantined
    assert quarantined.gate_action == "stale_quarantine"
    assert quarantined.next_teacher_state == "quarantined"
    assert not quarantined.activate_new and quarantined.preserve_quarantined
    assert replaced.gate_action == "replace_new"
    assert replaced.activate_new and replaced.stale_event
    assert kept.gate_action == "stale_keep_old"
    assert kept.next_teacher_state == "active"
    assert kept.activate_existing and not kept.activate_new
    assert kept.stale_event


def test_quarantine_uses_current_certificates_and_cannot_accumulate():
    signature = inspect.signature(decide_replacement_gate)
    assert "historical_certificate" not in signature.parameters
    inactive = decide_replacement_gate(
        teacher_state="quarantined", has_new=True,
        student_to_new=0.2, student_to_existing=-0.1,
        existing_to_new=-0.1, margin=0.0, stale_action="quarantine",
    )
    reactivated = decide_replacement_gate(
        teacher_state="quarantined", has_new=True,
        student_to_new=0.2, student_to_existing=0.1,
        existing_to_new=-0.1, margin=0.0, stale_action="quarantine",
    )
    replaced = decide_replacement_gate(
        teacher_state="quarantined", has_new=True,
        student_to_new=0.2, student_to_existing=0.1,
        existing_to_new=0.1, margin=0.0, stale_action="quarantine",
    )
    assert inactive.gate_action == "remain_inactive"
    assert inactive.preserve_quarantined
    assert reactivated.gate_action == "reactivate_quarantined"
    assert replaced.gate_action == "replace_quarantined_with_new"
    # State has exactly one existing slot; no transition returns an archive.
    assert not hasattr(inactive, "archive")


def test_explicit_non_stale_truth_table():
    def decide(old, new, beats):
        return decide_replacement_gate(
            teacher_state="active", has_new=True,
            student_to_new=1.0 if new else -1.0,
            student_to_existing=1.0 if old else -1.0,
            existing_to_new=1.0 if beats else -1.0,
            margin=0.0, stale_action="disable",
        ).gate_action

    assert decide(True, True, True) == "replace_new"
    assert decide(True, True, False) == "keep_old"
    assert decide(True, False, False) == "keep_old"
    assert decide(False, True, False) == "stale_disable"
    assert decide(False, False, False) == "remain_inactive"


def test_statewise_mask_detached_and_empty_mask_zero():
    student = jnp.asarray([[0.0], [1.0]])
    teacher = jnp.asarray([[1.0], [0.0]])
    q_student = jnp.ones((2, 2))
    q_teacher = jnp.zeros((2, 2))

    def weighted(q_s, q_t):
        return teacher_bc_components(
            student, teacher, q_s, q_t,
            teacher_active=1.0, lambda_t=1.0, beta_uncertainty=0.75,
            mode="statewise_lcb_mask",
        )["teacher_bc_loss_weighted"]

    grad_student, grad_teacher = jax.grad(weighted, argnums=(0, 1))(
        q_student, q_teacher
    )
    np.testing.assert_array_equal(grad_student, jnp.zeros_like(q_student))
    np.testing.assert_array_equal(grad_teacher, jnp.zeros_like(q_teacher))
    assert float(weighted(q_student, q_teacher)) == 0.0


def test_resume_status_skips_complete(tmp_path: Path):
    row = {
        "output_dir": str(tmp_path),
        "config": {"max_timesteps": 100},
    }
    assert run_status(row) == "pending"
    (tmp_path / "latest.pkl").write_bytes(b"checkpoint")
    assert run_status(row) == "incomplete"
    (tmp_path / "summary.json").write_text(
        json.dumps({"status": "complete", "final_eval": {"step": 100}})
    )
    assert run_status(row) == "complete"


def test_synthetic_late_mean_and_drawdown():
    rows = [
        {"step": 100_000, "student_d4rl_score": 10.0},
        {"step": 700_000, "student_d4rl_score": 80.0},
        {"step": 750_000, "student_d4rl_score": 50.0},
        {"step": 800_000, "student_d4rl_score": 60.0},
        {"step": 1_000_000, "student_d4rl_score": 40.0},
    ]
    metrics = compute_curve_metrics(rows)
    assert metrics["late_mean_700k_1M"] == 57.5
    assert metrics["max_peak_to_later_drawdown"] == 40.0
    assert metrics["largest_consecutive_eval_drop"] == 30.0
    assert metrics["largest_drop_within_50k"] == 30.0
