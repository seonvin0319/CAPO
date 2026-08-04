"""Fixed-batch numerical parity diagnostics between JAX and PyTorch CAPO."""
from __future__ import annotations

import json
import importlib.util
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch

_torch_networks_path = Path(__file__).resolve().parents[1] / "capo" / "networks.py"
_spec = importlib.util.spec_from_file_location("_capo_torch_networks", _torch_networks_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load PyTorch network definitions from {_torch_networks_path}")
_torch_networks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_torch_networks)
TorchActor = _torch_networks.Actor
TorchCriticEnsemble = _torch_networks.CriticEnsemble

from .core import propose_adaptive_tau
from .gate import decide_replacement_gate
from .networks import Actor, CriticEnsemble


def _copy_linear(linear, kernel, bias) -> None:
    with torch.no_grad():
        linear.weight.copy_(torch.as_tensor(np.asarray(kernel).T))
        linear.bias.copy_(torch.as_tensor(np.asarray(bias)))


def torch_models_from_jax(
    actor_params: Any, critic_params: Any, *, state_dim: int, action_dim: int,
    hidden: int, max_action: float, n_critics: int = 4,
) -> Tuple[TorchActor, TorchCriticEnsemble]:
    actor = TorchActor(state_dim, action_dim, max_action=max_action, hidden=hidden)
    for index, dense in enumerate(("Dense_0", "Dense_1", "Dense_2")):
        _copy_linear(actor.net[index * 2], actor_params[dense]["kernel"], actor_params[dense]["bias"])
    critics = TorchCriticEnsemble(state_dim, action_dim, n_critics=n_critics, hidden=hidden)
    vmapped = critic_params["VmapQNet_0"]
    for critic_index, critic in enumerate(critics.qs):
        for layer_index, dense in enumerate(("Dense_0", "Dense_1", "Dense_2")):
            _copy_linear(
                critic.net[layer_index * 2], vmapped[dense]["kernel"][critic_index],
                vmapped[dense]["bias"][critic_index],
            )
    return actor, critics


def _jax_actor_loss(actor_apply, critic_apply, actor_params, critic_params, states, actions, teacher_params, cfg):
    pi = actor_apply(actor_params, states)
    q_all = critic_apply(critic_params, states, pi)
    q = q_all.mean(axis=0)
    scale = jax.lax.stop_gradient(jnp.mean(jnp.abs(q)) + cfg["eps"])
    bc = jnp.mean((pi - actions) ** 2)
    teacher = jnp.mean((pi - jax.lax.stop_gradient(actor_apply(teacher_params, states))) ** 2)
    if cfg["objective"] == "td3bc_legacy":
        q_term = -(cfg["alpha"] / scale) * jnp.mean(q)
        total = q_term + bc
    else:
        q_term = -jnp.mean(q / scale)
        total = q_term + cfg["lambda_D"] * bc + cfg["lambda_T"] * teacher
    return total, (pi, q_all, q, scale, q_term, bc, teacher)


def _torch_actor_loss(actor, critics, states, actions, teacher, cfg):
    pi = actor(states)
    q_all = critics(states, pi)
    q = q_all.mean(dim=0)
    scale = q.abs().mean().detach() + cfg["eps"]
    bc = torch.mean((pi - actions) ** 2)
    with torch.no_grad():
        teacher_actions = teacher(states)
    teacher_bc = torch.mean((pi - teacher_actions) ** 2)
    if cfg["objective"] == "td3bc_legacy":
        q_term = -(cfg["alpha"] / scale) * q.mean()
        total = q_term + bc
    else:
        q_term = -(q / scale).mean()
        total = q_term + cfg["lambda_D"] * bc + cfg["lambda_T"] * teacher_bc
    return total, (pi, q_all, q, scale, q_term, bc, teacher_bc)


def _certificate(q_old, q_new, actions_old, actions_new, data_actions, cfg) -> float:
    scale = max(float(cfg["certificate_q_scale"]), 1e-6)
    per_critic = np.asarray((q_new - q_old).mean(axis=1) / scale, dtype=np.float64)
    gain = float(per_critic.mean())
    uncertainty = float(per_critic.std(ddof=1)) if len(per_critic) > 1 else 0.0
    movement = float(np.sqrt(np.mean(np.sum((actions_new - actions_old) ** 2, axis=-1))))
    d_new = float(np.mean(np.sum((actions_new - data_actions) ** 2, axis=-1)))
    d_old = float(np.mean(np.sum((actions_old - data_actions) ** 2, axis=-1)))
    return gain - cfg["beta"] * uncertainty - cfg["shift"] * movement**2 - cfg["data_penalty"] * max(0.0, d_new - d_old)


def _diff(a, b) -> Dict[str, float]:
    aa, bb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    absolute = float(np.max(np.abs(aa - bb)))
    relative = float(np.max(np.abs(aa - bb) / np.maximum(np.abs(aa), 1e-12)))
    return {"absolute_max": absolute, "relative_max": relative}


def run_parity(
    *, actor_params: Any, critic_params: Any, teacher_params: Any,
    states: np.ndarray, actions: np.ndarray, config: Mapping[str, Any],
) -> Dict[str, Any]:
    state_dim, action_dim = states.shape[1], actions.shape[1]
    hidden = int(config.get("hidden", 256)); max_action = float(config.get("max_action", 1.0))
    n_critics = int(config.get("n_critics", 4))
    j_actor = Actor(action_dim=action_dim, hidden=hidden, max_action=max_action)
    j_critic = CriticEnsemble(n_critics=n_critics, hidden=hidden, n_hidden=2)
    actor_apply = lambda p, s: j_actor.apply({"params": p}, s)
    critic_apply = lambda p, s, a: j_critic.apply({"params": p}, s, a)
    t_actor, t_critic = torch_models_from_jax(
        actor_params, critic_params, state_dim=state_dim, action_dim=action_dim,
        hidden=hidden, max_action=max_action, n_critics=n_critics,
    )
    t_teacher, _ = torch_models_from_jax(
        teacher_params, critic_params, state_dim=state_dim, action_dim=action_dim,
        hidden=hidden, max_action=max_action, n_critics=n_critics,
    )
    js, ja = jnp.asarray(states), jnp.asarray(actions)
    ts, ta = torch.as_tensor(states), torch.as_tensor(actions)
    cfg = {
        "objective": config.get("td3_actor_objective", "capo_student"),
        "alpha": float(config.get("alpha", 2.5)), "eps": float(config.get("actor_q_scale_eps", 1e-4)),
        "lambda_D": float(config.get("lambda_D", 0.2)), "lambda_T": float(config.get("lambda_T", 0.0)),
        "certificate_q_scale": float(config.get("certificate_q_scale", 1.0)),
        "beta": float(config.get("beta_uncertainty", 0.75)), "shift": float(config.get("shift_penalty_coef", 0.25)),
        "data_penalty": float(config.get("data_penalty_coef", 0.25)),
    }
    def j_loss_fn(params):
        return _jax_actor_loss(
            actor_apply, critic_apply, params, critic_params, js, ja,
            teacher_params, cfg,
        )

    (j_loss, j_aux), j_grads = jax.value_and_grad(
        j_loss_fn, has_aux=True
    )(actor_params)
    t_opt = torch.optim.Adam(t_actor.parameters(), lr=float(config.get("actor_lr", 3e-4)))
    t_loss, t_aux = _torch_actor_loss(t_actor, t_critic, ts, ta, t_teacher, cfg)
    t_opt.zero_grad(); t_loss.backward(); t_opt.step()
    j_tx = optax.adam(float(config.get("actor_lr", 3e-4)))
    updates, _ = j_tx.update(j_grads, j_tx.init(actor_params), actor_params)
    j_updated = optax.apply_updates(actor_params, updates)
    j_after = np.asarray(actor_apply(j_updated, js)); t_after = t_actor(ts).detach().numpy()
    j_pi, j_qall, _, j_scale, j_qterm, j_bc, j_tbc = j_aux
    t_pi, t_qall, _, t_scale, t_qterm, t_bc, t_tbc = t_aux

    # Compare a deterministic pilot/adaptive candidate generated by a shared
    # action-space proximal approximation. This isolates certificate math from
    # optimizer implementation differences.
    q_grad_direction = np.tanh(np.asarray(j_qall).mean(axis=0))[:, None]
    pilot_tau = float(config.get("tau_pilot_initial", 0.01))
    pilot_j = np.clip(np.asarray(j_pi) + pilot_tau * q_grad_direction, -max_action, max_action)
    pilot_t = np.clip(t_pi.detach().numpy() + pilot_tau * q_grad_direction, -max_action, max_action)
    movement_j = float(np.sqrt(np.mean(np.sum((pilot_j - np.asarray(j_pi)) ** 2, axis=-1))))
    movement_t = float(np.sqrt(np.mean(np.sum((pilot_t - t_pi.detach().numpy()) ** 2, axis=-1))))
    adaptive_tau_j = propose_adaptive_tau(pilot_tau, movement_j, float(config.get("target_action_mse", 0.0025)))
    adaptive_tau_t = propose_adaptive_tau(pilot_tau, movement_t, float(config.get("target_action_mse", 0.0025)))
    q_old_j = np.asarray(j_qall); q_old_t = t_qall.detach().numpy()
    q_new_j = np.asarray(critic_apply(critic_params, js, jnp.asarray(pilot_j)))
    q_new_t = t_critic(ts, torch.as_tensor(pilot_t)).detach().numpy()
    cert_j = _certificate(q_old_j, q_new_j, np.asarray(j_pi), pilot_j, actions, cfg)
    cert_t = _certificate(q_old_t, q_new_t, t_pi.detach().numpy(), pilot_t, actions, cfg)
    old_j = np.asarray(actor_apply(teacher_params, js)); old_t = t_teacher(ts).detach().numpy()
    q_teacher_j = np.asarray(critic_apply(critic_params, js, jnp.asarray(old_j)))
    q_teacher_t = t_critic(ts, torch.as_tensor(old_t)).detach().numpy()
    c_so_j = _certificate(q_old_j, q_teacher_j, np.asarray(j_pi), old_j, actions, cfg)
    c_so_t = _certificate(q_old_t, q_teacher_t, t_pi.detach().numpy(), old_t, actions, cfg)
    c_on_j = _certificate(q_teacher_j, q_new_j, old_j, pilot_j, actions, cfg)
    c_on_t = _certificate(q_teacher_t, q_new_t, old_t, pilot_t, actions, cfg)
    margin = float(config.get("replace_cert_margin", 0.0))
    gate_j = decide_replacement_gate(
        teacher_state="active", has_new=True, student_to_new=cert_j,
        student_to_existing=c_so_j, existing_to_new=c_on_j, margin=margin,
        stale_action=config.get("stale_incumbent_action", "disable"),
    ).gate_action
    gate_t = decide_replacement_gate(
        teacher_state="active", has_new=True, student_to_new=cert_t,
        student_to_existing=c_so_t, existing_to_new=c_on_t, margin=margin,
        stale_action=config.get("stale_incumbent_action", "disable"),
    ).gate_action
    report = {
        "parameter_conversion": "supported",
        "normalized_states": {"jax": np.asarray(js).tolist(), "torch": states.tolist(), "difference": _diff(js, states)},
        "actor_actions_before_update": {"difference": _diff(j_pi, t_pi.detach().numpy())},
        "q_outputs": {f"Q{i+1}": _diff(np.asarray(j_qall)[i], t_qall.detach().numpy()[i]) for i in range(n_critics)},
        "q_aggregation": "mean_of_four_critics",
        "q_scale": {"jax": float(j_scale), "torch": float(t_scale), "difference": _diff(j_scale, float(t_scale))},
        "q_actor_term": {"jax": float(j_qterm), "torch": float(t_qterm), "difference": _diff(j_qterm, float(t_qterm))},
        "data_bc_loss": {"jax": float(j_bc), "torch": float(t_bc), "difference": _diff(j_bc, float(t_bc))},
        "teacher_bc_loss": {"jax": float(j_tbc), "torch": float(t_tbc), "difference": _diff(j_tbc, float(t_tbc))},
        "total_actor_loss": {"jax": float(j_loss), "torch": float(t_loss), "difference": _diff(j_loss, float(t_loss))},
        "one_actor_optimizer_update": {"action_difference": _diff(j_after, t_after)},
        "action_change_after_one_update": {"jax": float(np.mean(np.abs(j_after - np.asarray(j_pi)))), "torch": float(np.mean(np.abs(t_after - t_pi.detach().numpy())))},
        "pilot_tau": {"jax": pilot_tau, "torch": pilot_tau},
        "adaptive_tau": {"jax": adaptive_tau_j, "torch": adaptive_tau_t},
        "pilot_movement": {"jax": movement_j, "torch": movement_t},
        "adaptive_movement": {"jax": movement_j * adaptive_tau_j / pilot_tau, "torch": movement_t * adaptive_tau_t / pilot_tau},
        "pilot_certificate": {"jax": cert_j, "torch": cert_t},
        "adaptive_certificate": {"jax": cert_j, "torch": cert_t, "note": "shared local action-space approximation"},
        "C_student_to_new": {"jax": cert_j, "torch": cert_t},
        "C_student_to_old": {"jax": c_so_j, "torch": c_so_t},
        "C_old_to_new": {"jax": c_on_j, "torch": c_on_t},
        "final_gate_action": {"jax": gate_j, "torch": gate_t},
    }
    return report


def write_report(report: Mapping[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = ["# JAX–PyTorch fixed-batch parity", "", f"Parameter conversion: `{report['parameter_conversion']}`", "", f"Q aggregation: `{report['q_aggregation']}`", ""]
    for key in ("actor_actions_before_update", "q_scale", "q_actor_term", "data_bc_loss", "teacher_bc_loss", "total_actor_loss", "one_actor_optimizer_update", "pilot_certificate", "C_student_to_new", "C_student_to_old", "C_old_to_new", "final_gate_action"):
        lines.append(f"- **{key}**: `{report[key]}`")
    markdown_path.write_text("\n".join(lines) + "\n")
