"""Pure replacement-gate state machine and teacher-BC loss helpers.

This module deliberately has no Gym/D4RL dependency so all safety semantics
can be tested deterministically without constructing a training environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

TeacherState = Literal["active", "quarantined", "disabled"]
StaleAction = Literal["disable", "quarantine", "replace_new", "keep_old"]


@dataclass(frozen=True)
class GateDecision:
    gate_action: str
    previous_teacher_state: TeacherState
    next_teacher_state: TeacherState
    valid_new: bool
    valid_old: bool
    new_beats_old: bool
    activate_new: bool = False
    activate_existing: bool = False
    preserve_quarantined: bool = False
    stale_event: bool = False


def certificate_passes(value: float, margin: float) -> bool:
    """Strict replacement check. NaN/inf certificates are always invalid."""
    return bool(jnp.isfinite(value)) and float(value) > float(margin)


def decide_replacement_gate(
    *,
    teacher_state: TeacherState,
    has_new: bool,
    student_to_new: float,
    student_to_existing: float,
    existing_to_new: float,
    margin: float,
    stale_action: StaleAction,
    nstar_zero: bool = False,
    nstar_zero_action: str = "revalidate_current",
) -> GateDecision:
    """Apply the current-refresh gate without any historical certificate.

    ``existing`` means the active incumbent in ``active`` state and the single
    retained incumbent in ``quarantined`` state. All three certificates must
    have been computed using the current critics, state batch, and q scale.
    """
    if teacher_state not in ("active", "quarantined", "disabled"):
        raise ValueError(f"invalid teacher_state: {teacher_state!r}")
    if stale_action not in ("disable", "quarantine", "replace_new", "keep_old"):
        raise ValueError(f"invalid stale_action: {stale_action!r}")
    if nstar_zero_action not in ("legacy_hold", "revalidate_current"):
        raise ValueError(f"invalid nstar_zero_action: {nstar_zero_action!r}")

    valid_new = bool(has_new) and certificate_passes(student_to_new, margin)
    valid_old = teacher_state != "disabled" and certificate_passes(
        student_to_existing, margin
    )
    new_beats_old = (
        bool(has_new)
        and teacher_state != "disabled"
        and certificate_passes(existing_to_new, margin)
    )

    common = dict(
        previous_teacher_state=teacher_state,
        valid_new=valid_new,
        valid_old=valid_old,
        new_beats_old=new_beats_old,
    )

    if nstar_zero and teacher_state == "active" and nstar_zero_action == "legacy_hold":
        return GateDecision(
            gate_action="keep_old",
            next_teacher_state="active",
            activate_existing=True,
            **common,
        )

    if teacher_state == "disabled":
        if valid_new:
            return GateDecision(
                gate_action="replace_new",
                next_teacher_state="active",
                activate_new=True,
                **common,
            )
        return GateDecision(
            gate_action="remain_inactive",
            next_teacher_state="disabled",
            **common,
        )

    if teacher_state == "quarantined":
        if valid_new and new_beats_old:
            return GateDecision(
                gate_action="replace_quarantined_with_new",
                next_teacher_state="active",
                activate_new=True,
                **common,
            )
        if valid_old:
            return GateDecision(
                gate_action="reactivate_quarantined",
                next_teacher_state="active",
                activate_existing=True,
                **common,
            )
        return GateDecision(
            gate_action="remain_inactive",
            next_teacher_state="quarantined",
            preserve_quarantined=True,
            **common,
        )

    # Active incumbent: explicit five-case replacement truth table.
    if valid_old and valid_new:
        if new_beats_old:
            return GateDecision(
                gate_action="replace_new",
                next_teacher_state="active",
                activate_new=True,
                **common,
            )
        return GateDecision(
            gate_action="keep_old",
            next_teacher_state="active",
            activate_existing=True,
            **common,
        )
    if valid_old:
        return GateDecision(
            gate_action="keep_old",
            next_teacher_state="active",
            activate_existing=True,
            **common,
        )
    if valid_new:
        if stale_action == "replace_new":
            return GateDecision(
                gate_action="replace_new",
                next_teacher_state="active",
                activate_new=True,
                stale_event=True,
                **common,
            )
        if stale_action == "quarantine":
            return GateDecision(
                gate_action="stale_quarantine",
                next_teacher_state="quarantined",
                preserve_quarantined=True,
                stale_event=True,
                **common,
            )
        if stale_action == "keep_old":
            # Stale challenger: keep incumbent teacher active (torch ablation parity).
            return GateDecision(
                gate_action="stale_keep_old",
                next_teacher_state="active",
                activate_existing=True,
                stale_event=True,
                **common,
            )
        return GateDecision(
            gate_action="stale_disable",
            next_teacher_state="disabled",
            stale_event=True,
            **common,
        )
    return GateDecision(
        gate_action="remain_inactive",
        next_teacher_state="disabled",
        **common,
    )


def teacher_bc_components(
    student_actions,
    teacher_actions,
    q_student,
    q_teacher,
    *,
    teacher_active,
    lambda_t,
    beta_uncertainty: float,
    mode: str,
):
    """Return uniform/masked BC diagnostics and the weighted teacher term."""
    if mode not in ("uniform", "statewise_lcb_mask"):
        raise ValueError(f"unknown teacher_bc_mode: {mode!r}")
    per_state_mse = jnp.mean((student_actions - teacher_actions) ** 2, axis=-1)
    uniform = jnp.mean(per_state_mse)

    delta = q_teacher - q_student
    state_lcb = jnp.mean(delta, axis=0) - beta_uncertainty * jnp.std(
        delta, axis=0
    )
    mask = jax.lax.stop_gradient((state_lcb > 0).astype(per_state_mse.dtype))
    selected = jnp.sum(mask)
    masked = jnp.sum(mask * per_state_mse) / jnp.maximum(selected, 1.0)
    selected_loss = uniform if mode == "uniform" else masked
    active = jnp.asarray(teacher_active, dtype=per_state_mse.dtype)
    coefficient = jnp.asarray(lambda_t, dtype=per_state_mse.dtype)
    weighted = active * coefficient * selected_loss
    return {
        "teacher_bc_uniform": uniform,
        "teacher_bc_masked": masked,
        "teacher_bc_loss_unweighted": selected_loss,
        "teacher_bc_loss_weighted": weighted,
        "teacher_mask_fraction": jnp.mean(mask),
        "teacher_lcb_mean": jnp.mean(state_lcb),
        "teacher_lcb_std": jnp.std(state_lcb),
    }
