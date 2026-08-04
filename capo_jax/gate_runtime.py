"""Trainer-facing application of the pure CAPO replacement gate."""
from __future__ import annotations

import json
from typing import Any, Dict

import jax.numpy as jnp

from .core import dataset_action_mse
from .gate import GateDecision, decide_replacement_gate
from .refresh_snapshots import save_refresh_actor_bundle


def _teacher_state(state) -> str:
    if float(state.has_teacher) > 0.5:
        return "active"
    if float(state.has_quarantined) > 0.5:
        return "quarantined"
    return "disabled"


def apply_teacher_replace_gate(
    trainer,
    result,
    states,
    data_actions,
    cert_critics,
    info: Dict[str, Any],
    paired_eval_fn,
) -> Dict[str, Any]:
    """Evaluate current actors, decide, mutate trainer state, and log."""
    cfg = trainer.cfg
    margin = float(cfg.replace_cert_margin)
    has_new = bool(result.selected_n > 0 and result.accepted)
    nstar_zero = int(result.selected_n) == 0
    previous_state = _teacher_state(trainer.state)
    has_existing = previous_state != "disabled"
    student = trainer._policy(trainer.state.actor_params)
    pi_new = result.final_policy if has_new else None
    if previous_state == "active":
        existing_params = trainer.state.teacher_params
        existing_n = int(trainer.state.teacher_n)
        existing_tau = float(trainer.state.teacher_tau)
    elif previous_state == "quarantined":
        existing_params = trainer.state.quarantined_params
        existing_n = int(trainer.state.quarantined_n)
        existing_tau = float(trainer.state.quarantined_tau)
    else:
        existing_params = None
        existing_n = 0
        existing_tau = float("nan")
    pi_existing = trainer._policy(existing_params) if has_existing else None
    q_scale = float(trainer.state.q_scale)

    c_sn = float("nan")
    c_se = float("nan")
    c_en = float("nan")
    if has_new:
        c_sn = trainer._pairwise_lcb_cert(
            student, pi_new, states, data_actions, cert_critics, q_scale
        )
    legacy_hold_without_revalidation = (
        nstar_zero
        and previous_state == "active"
        and cfg.nstar_zero_action == "legacy_hold"
    )
    if has_existing and not legacy_hold_without_revalidation:
        c_se = trainer._pairwise_lcb_cert(
            student, pi_existing, states, data_actions, cert_critics, q_scale
        )
    if has_existing and has_new:
        c_en = trainer._pairwise_lcb_cert(
            pi_existing, pi_new, states, data_actions, cert_critics, q_scale
        )

    if cfg.use_replace_gate:
        decision = decide_replacement_gate(
            teacher_state=previous_state,
            has_new=has_new,
            student_to_new=c_sn,
            student_to_existing=c_se,
            existing_to_new=c_en,
            margin=margin,
            stale_action=cfg.stale_incumbent_action,
            nstar_zero=nstar_zero,
            nstar_zero_action=cfg.nstar_zero_action,
        )
    else:
        # Compatibility path; the stability sweep always enables the gate.
        if has_new:
            decision = GateDecision(
                "replace_new", previous_state, "active", True, False, False,
                activate_new=True,
            )
        elif previous_state == "active" and cfg.teacher_hold:
            decision = GateDecision(
                "keep_old", previous_state, "active", False, True, False,
                activate_existing=True,
            )
        else:
            decision = GateDecision(
                "remain_inactive", previous_state, "disabled", False, False, False
            )

    new_tau = (
        float(result.selected_tau[-1])
        if has_new and result.selected_tau
        else float("nan")
    )
    if decision.activate_new:
        trainer.state = trainer.state.replace(
            teacher_params=pi_new.params,
            has_teacher=jnp.asarray(1.0),
            teacher_n=jnp.asarray(int(result.selected_n)),
            teacher_tau=jnp.asarray(new_tau),
            has_quarantined=jnp.asarray(0.0),
            quarantined_n=jnp.asarray(0),
            quarantined_tau=jnp.asarray(float("nan")),
        )
    elif decision.activate_existing:
        if previous_state == "quarantined":
            trainer.state = trainer.state.replace(
                teacher_params=trainer.state.quarantined_params,
                has_teacher=jnp.asarray(1.0),
                teacher_n=jnp.asarray(existing_n),
                teacher_tau=jnp.asarray(existing_tau),
                has_quarantined=jnp.asarray(0.0),
                quarantined_n=jnp.asarray(0),
                quarantined_tau=jnp.asarray(float("nan")),
            )
        else:
            trainer.state = trainer.state.replace(has_teacher=jnp.asarray(1.0))
    elif decision.preserve_quarantined:
        if previous_state == "active":
            trainer.state = trainer.state.replace(
                quarantined_params=trainer.state.teacher_params,
                has_quarantined=jnp.asarray(1.0),
                quarantined_n=jnp.asarray(existing_n),
                quarantined_tau=jnp.asarray(existing_tau),
                has_teacher=jnp.asarray(0.0),
                teacher_n=jnp.asarray(0),
                teacher_tau=jnp.asarray(float("nan")),
            )
        else:
            trainer.state = trainer.state.replace(has_teacher=jnp.asarray(0.0))
    else:
        trainer.state = trainer.state.replace(
            has_teacher=jnp.asarray(0.0),
            teacher_n=jnp.asarray(0),
            teacher_tau=jnp.asarray(float("nan")),
            has_quarantined=jnp.asarray(0.0),
            quarantined_n=jnp.asarray(0),
            quarantined_tau=jnp.asarray(float("nan")),
        )

    next_state = _teacher_state(trainer.state)
    counts = trainer.gate_counts
    if decision.gate_action in ("replace_new", "replace_quarantined_with_new"):
        counts["replace_count"] += 1
    if previous_state == "active" and next_state != "active":
        counts["disable_count"] += 1
    if decision.gate_action == "stale_quarantine":
        counts["quarantine_count"] += 1
    if decision.gate_action == "reactivate_quarantined":
        counts["reactivation_count"] += 1
    if decision.stale_event:
        counts["stale_count"] += 1

    selected_movements = [float(value) for value in result.movements]
    refresh_row: Dict[str, Any] = {
        "refresh_step": int(trainer.state.total_it),
        "N_star": int(result.selected_n),
        "challenger_n": int(result.selected_n) if has_new else 0,
        "incumbent_n": existing_n,
        "selected_tau_per_ladder_step": list(result.selected_tau),
        "selected_cert_per_ladder_step": list(result.certificates),
        "selected_movement_per_ladder_step": selected_movements,
        "ladder_accepted": has_new,
        "stop_cert": info.get("capo_stop_cert"),
        "C_student_to_new": None if c_sn != c_sn else c_sn,
        "C_student_to_old": None if c_se != c_se else c_se,
        "C_old_to_new": None if c_en != c_en else c_en,
        # Backward-compatible aliases.
        "student_to_new_cert": None if c_sn != c_sn else c_sn,
        "student_to_old_cert": None if c_se != c_se else c_se,
        "old_to_new_replace_cert": None if c_en != c_en else c_en,
        "valid_new": decision.valid_new,
        "valid_old": decision.valid_old,
        "new_beats_old": decision.new_beats_old,
        "gate_action": decision.gate_action,
        "replacement_decision": decision.gate_action,
        "previous_teacher_state": previous_state,
        "next_teacher_state": next_state,
        "teacher_state": next_state,
        "q_scale": q_scale,
        "use_replace_gate": bool(cfg.use_replace_gate),
        "replace_cert_margin": margin,
        "stale_incumbent_action": cfg.stale_incumbent_action,
        "nstar_zero_action": cfg.nstar_zero_action,
        "nstar_zero_revalidation_performed": bool(
            nstar_zero and has_existing and not legacy_hold_without_revalidation
        ),
        "nstar_zero_incumbent_raw_cert": (
            c_se if nstar_zero and has_existing and c_se == c_se else None
        ),
        "nstar_zero_incumbent_valid": (
            decision.valid_old if nstar_zero and has_existing
            and not legacy_hold_without_revalidation else None
        ),
        "nstar_zero_gate_action": decision.gate_action if nstar_zero else None,
        "replace_count": counts["replace_count"],
        "disable_count": counts["disable_count"],
        "quarantine_count": counts["quarantine_count"],
        "reactivation_count": counts["reactivation_count"],
        "stale_count": counts["stale_count"],
        "backend": "jax",
    }
    if has_new:
        refresh_row["dataset_amse"] = float(
            dataset_action_mse(pi_new, states, data_actions)
        )

    if cfg.save_refresh_actors:
        snapshot_path = save_refresh_actor_bundle(
            run_dir=trainer.run_dir,
            trainer=trainer,
            refresh_step=int(trainer.state.total_it),
            roles={
                "student_before_refresh": trainer.state.actor_params,
                "active_incumbent_before_gate": (
                    existing_params if previous_state == "active" else None
                ),
                "newly_generated_challenger": (
                    pi_new.params if has_new else None
                ),
                "quarantined_incumbent_before_gate": (
                    existing_params if previous_state == "quarantined" else None
                ),
                "active_teacher_after_gate": (
                    trainer.state.teacher_params if next_state == "active" else None
                ),
            },
            refresh_row=refresh_row,
        )
        refresh_row["refresh_actor_bundle"] = str(snapshot_path)

    # Diagnostic rollouts happen strictly after the offline gate decision.
    if cfg.paired_eval_episodes > 0:
        seeds = [cfg.paired_eval_seed0 + i for i in range(cfg.paired_eval_episodes)]
        if has_new:
            paired_new = paired_eval_fn(
                trainer.eval_env,
                trainer.actor_apply,
                trainer.state.actor_params,
                trainer.actor_apply,
                pi_new.params,
                seeds,
            )
            refresh_row["paired_delta_d4rl_new"] = paired_new.get(
                "paired_delta_d4rl"
            )
            refresh_row["new_teacher_d4rl"] = paired_new.get("teacher_d4rl_score")
        if has_existing:
            paired_existing = paired_eval_fn(
                trainer.eval_env,
                trainer.actor_apply,
                trainer.state.actor_params,
                trainer.actor_apply,
                existing_params,
                seeds,
            )
            refresh_row["paired_delta_d4rl_old"] = paired_existing.get(
                "paired_delta_d4rl"
            )
            refresh_row["old_teacher_d4rl"] = paired_existing.get(
                "teacher_d4rl_score"
            )

    info.update(
        {
            "student_to_new_cert": c_sn,
            "student_to_old_cert": c_se,
            "old_to_new_replace_cert": c_en,
            "valid_new": float(decision.valid_new),
            "valid_old": float(decision.valid_old),
            "new_beats_old": float(decision.new_beats_old),
            "replacement_decision_code": {
                "replace_new": 2.0,
                "keep_old": 1.0,
                "stale_disable": -1.0,
                "stale_quarantine": -2.0,
                "reactivate_quarantined": 3.0,
                "replace_quarantined_with_new": 4.0,
                "remain_inactive": 0.0,
            }[decision.gate_action],
            "replace_count": float(counts["replace_count"]),
            "disable_count": float(counts["disable_count"]),
            "quarantine_count": float(counts["quarantine_count"]),
            "reactivation_count": float(counts["reactivation_count"]),
            "stale_count": float(counts["stale_count"]),
        }
    )
    trainer.last_gate_action = decision.gate_action
    with open(trainer.run_dir / "capo_refresh.jsonl", "a") as stream:
        stream.write(json.dumps(refresh_row) + "\n")
    print(
        f"  gate action={decision.gate_action} {previous_state}→{next_state} "
        f"valid_new={decision.valid_new} valid_old={decision.valid_old} "
        f"new_beats_old={decision.new_beats_old} "
        f"C_S→N={c_sn:+.5f} C_S→O={c_se:+.5f} C_O→N={c_en:+.5f}",
        flush=True,
    )
    return refresh_row
