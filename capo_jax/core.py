"""Certificate-driven capped adaptive MPI core (JAX).

Same design constraints as capo.core:
- Candidate generation critics and certification critics may be split.
- Certificates use scale-normalized ΔQ and optional dataset-distance penalty.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import jax.numpy as jnp
import numpy as np


class Policy(Protocol):
    def act(self, states: Any) -> Any: ...

    def copy(self) -> "Policy": ...


class Critic(Protocol):
    def q(self, states: Any, actions: Any) -> Any: ...


class ProxRefiner(Protocol):
    def refine(
        self,
        policy_center: Policy,
        critics: Sequence[Critic],
        tau: float,
        states: Any,
        *,
        gen_critic_params: Any = None,
    ) -> Policy: ...


@dataclass(frozen=True)
class CAPOConfig:
    n_max: int = 2
    beta_uncertainty: float = 2.0
    shift_penalty_coef: float = 1.0
    data_penalty_coef: float = 1.0
    actor_opt_error: float = 0.0
    accept_margin: float = 0.0
    tau_candidates: Tuple[float, ...] = (1e-3, 3e-3, 5e-3, 1e-2)
    tau_max: float = 1e-2
    tau_min: float = 1e-3
    max_action_mse: Optional[float] = 0.01
    normalize_delta_q: bool = True
    q_scale_eps: float = 1e-6
    reference_eps: float = 1e-6
    tau_controller: str = "pilot_adaptive"
    target_action_mse: float = 0.0025
    initial_tau: float = 0.01
    tau_pilot_initial: float = 0.01
    tau_duplicate_log_tolerance: float = 1e-6


@dataclass
class CandidateStats:
    tau: float
    estimated_gain: float
    uncertainty: float
    shift_penalty: float
    data_penalty: float
    actor_error: float
    certificate: float
    movement: float
    action_mse: float
    q_scale: float


@dataclass
class StepRecord:
    step: int
    selected_tau: Optional[float]
    accepted: bool
    selected_certificate: float
    ladder_value: float
    candidates: List[CandidateStats]
    diagnostics: Dict[str, float] = field(default_factory=dict)


@dataclass
class CAPOResult:
    actor: Policy
    policies: List[Policy]
    records: List[StepRecord]
    final_policy: Policy
    selected_n: int
    selected_tau: List[float] = field(default_factory=list)
    certificates: List[float] = field(default_factory=list)
    movements: List[float] = field(default_factory=list)
    accepted: bool = False


def _as_float(x) -> float:
    if hasattr(x, "mean"):
        return float(np.asarray(x).astype(np.float64).mean())
    return float(x)


def _filter_taus(cfg: CAPOConfig) -> Tuple[float, ...]:
    taus = tuple(float(t) for t in cfg.tau_candidates if float(t) <= float(cfg.tau_max) + 1e-12)
    if not taus:
        taus = (min(cfg.tau_candidates),)
    return taus


def clip_tau(tau: float, tau_min: float, tau_max: float) -> float:
    return float(min(max(float(tau), float(tau_min)), float(tau_max)))


def propose_adaptive_tau(
    tau0: float,
    movement0: float,
    delta_target: float,
    eps: float = 1e-8,
) -> float:
    return float(tau0) * math.sqrt(float(delta_target) / max(float(movement0), eps))


def taus_are_duplicates(tau0: float, tau1: float, log_tol: float) -> bool:
    t0 = max(float(tau0), 1e-12)
    t1 = max(float(tau1), 1e-12)
    return abs(math.log(t1 / t0)) < float(log_tol)


def estimate_q_scale(
    critics: Sequence[Critic],
    states,
    actions=None,
    policy: Optional[Policy] = None,
    eps: float = 1e-6,
) -> float:
    if actions is None:
        if policy is None:
            raise ValueError("estimate_q_scale requires actions or policy")
        actions = policy.act(states)
    vals = []
    if len(critics) == 1 and hasattr(critics[0], "q_all"):
        vals.append(np.asarray(critics[0].q_all(states, actions)).astype(np.float64).reshape(-1))
    else:
        for critic in critics:
            q = critic.q(states, actions)
            vals.append(np.asarray(q).astype(np.float64).reshape(-1))
    q_all = np.concatenate(vals, axis=0)
    scale = float(q_all.std(ddof=0))  # population std (PyTorch unbiased=False)
    if scale < eps:
        scale = float(np.abs(q_all).mean())
    return max(scale, eps)


def pairwise_gain_ensemble(
    critics: Sequence[Critic],
    current_policy: Policy,
    candidate_policy: Policy,
    states,
    q_scale: float = 1.0,
    normalize: bool = True,
    eps: float = 1e-6,
) -> Tuple[float, float, np.ndarray]:
    a_new = candidate_policy.act(states)
    a_old = current_policy.act(states)
    scale = max(float(q_scale), eps) if normalize else 1.0
    if len(critics) == 1 and hasattr(critics[0], "q_all"):
        q_new = np.asarray(critics[0].q_all(states, a_new), dtype=np.float64)
        q_old = np.asarray(critics[0].q_all(states, a_old), dtype=np.float64)
        delta = (q_new - q_old) / scale
        if delta.ndim <= 1:
            per_critic = np.asarray([delta.mean()], dtype=np.float64)
        else:
            per_critic = delta.reshape(delta.shape[0], -1).mean(axis=1)
    else:
        deltas = []
        for critic in critics:
            q_new = critic.q(states, a_new)
            q_old = critic.q(states, a_old)
            deltas.append(_as_float((q_new - q_old) / scale))
        per_critic = np.asarray(deltas, dtype=np.float64)
    mean_delta = float(np.mean(per_critic))
    std_delta = float(np.std(per_critic, ddof=1)) if len(per_critic) > 1 else 0.0
    return mean_delta, std_delta, per_critic


def policy_movement(current_policy: Policy, candidate_policy: Policy, states) -> float:
    a0 = np.asarray(current_policy.act(states))
    a1 = np.asarray(candidate_policy.act(states))
    return float(np.sqrt(((a1 - a0) ** 2).sum(axis=-1).mean()))


def policy_action_mse(current_policy: Policy, candidate_policy: Policy, states) -> float:
    a0 = np.asarray(current_policy.act(states))
    a1 = np.asarray(candidate_policy.act(states))
    return float(((a1 - a0) ** 2).sum(axis=-1).mean())


def dataset_action_mse(policy: Policy, states, data_actions) -> float:
    a = np.asarray(policy.act(states))
    da = np.asarray(data_actions)
    return float(((a - da) ** 2).sum(axis=-1).mean())


def candidate_certificate(
    cert_critics: Sequence[Critic],
    current_policy: Policy,
    candidate_policy: Policy,
    states,
    tau: float,
    cfg: CAPOConfig,
    q_scale: float = 1.0,
    data_actions=None,
) -> CandidateStats:
    mean_gain, uncertainty, _ = pairwise_gain_ensemble(
        critics=cert_critics,
        current_policy=current_policy,
        candidate_policy=candidate_policy,
        states=states,
        q_scale=q_scale,
        normalize=cfg.normalize_delta_q,
        eps=cfg.q_scale_eps,
    )
    move = policy_movement(current_policy, candidate_policy, states)
    action_mse = policy_action_mse(current_policy, candidate_policy, states)
    shift = cfg.shift_penalty_coef * (move**2)

    data_penalty = 0.0
    if data_actions is not None and cfg.data_penalty_coef > 0:
        d_new = dataset_action_mse(candidate_policy, states, data_actions)
        d_old = dataset_action_mse(current_policy, states, data_actions)
        data_penalty = cfg.data_penalty_coef * max(0.0, d_new - d_old)

    cert = mean_gain - cfg.beta_uncertainty * uncertainty - shift - data_penalty - cfg.actor_opt_error
    if cfg.max_action_mse is not None and action_mse > cfg.max_action_mse:
        cert = min(cert, -abs(cert) - 1.0)

    return CandidateStats(
        tau=tau,
        estimated_gain=mean_gain,
        uncertainty=uncertainty,
        shift_penalty=shift,
        data_penalty=data_penalty,
        actor_error=cfg.actor_opt_error,
        certificate=cert,
        movement=move,
        action_mse=action_mse,
        q_scale=float(q_scale),
    )


def update_reference_ladder(ladder_value: float, accepted_certificate: float) -> float:
    return ladder_value + max(0.0, accepted_certificate)


def propose_tau(
    prev_tau: float,
    prev_action_mse: float,
    target_action_mse: float,
    tau_grid: Sequence[float],
) -> float:
    grid = [float(t) for t in tau_grid]
    if not grid:
        raise ValueError("empty tau_grid")
    raw = float(prev_tau) * (
        float(target_action_mse) / max(float(prev_action_mse), 1e-8)
    ) ** 0.5
    raw = max(raw, 1e-12)
    return min(grid, key=lambda t: abs(math.log(t) - math.log(raw)))


def _ensure_controller_state(
    tau_controller_state: Optional[List[dict]],
    n_max: int,
    cfg: CAPOConfig,
) -> List[dict]:
    if tau_controller_state is None:
        tau_controller_state = []
    while len(tau_controller_state) < n_max:
        tau_controller_state.append(
            {
                "previous_selected_tau": None,
                "tau": float(cfg.tau_pilot_initial),
                "action_mse": float(cfg.target_action_mse),
            }
        )
    return tau_controller_state


def _refine_and_certify(
    *,
    refiner: ProxRefiner,
    gen_critics: Sequence[Critic],
    cert_critics: Sequence[Critic],
    center: Policy,
    states,
    tau: float,
    cfg: CAPOConfig,
    q_scale: float,
    data_actions,
    gen_critic_params=None,
) -> Tuple[Policy, CandidateStats]:
    candidate = refiner.refine(
        policy_center=center,
        critics=gen_critics,
        tau=float(tau),
        states=states,
        gen_critic_params=gen_critic_params,
    )
    stats = candidate_certificate(
        cert_critics=cert_critics,
        current_policy=center,
        candidate_policy=candidate,
        states=states,
        tau=float(tau),
        cfg=cfg,
        q_scale=float(q_scale),
        data_actions=data_actions,
    )
    return candidate, stats


def _pilot_adaptive_step(
    *,
    n: int,
    current: Policy,
    refiner: ProxRefiner,
    gen_critics: Sequence[Critic],
    cert_critics: Sequence[Critic],
    states,
    cfg: CAPOConfig,
    q_scale: float,
    data_actions,
    controller_n: dict,
    gen_critic_params=None,
) -> Tuple[Optional[Policy], CandidateStats, List[CandidateStats], Dict[str, float], bool]:
    eps = float(cfg.reference_eps)
    prev = controller_n.get("previous_selected_tau")
    tau0 = float(cfg.tau_pilot_initial) if prev is None else float(prev)
    tau0 = clip_tau(tau0, cfg.tau_min, cfg.tau_max)

    cand0, stats0 = _refine_and_certify(
        refiner=refiner,
        gen_critics=gen_critics,
        cert_critics=cert_critics,
        center=current,
        states=states,
        tau=tau0,
        cfg=cfg,
        q_scale=q_scale,
        data_actions=data_actions,
        gen_critic_params=gen_critic_params,
    )
    movement0 = float(stats0.action_mse)
    c_hat = movement0 / max(tau0**2, eps)
    tau1_raw = propose_adaptive_tau(tau0, movement0, cfg.target_action_mse, eps=eps)
    tau1 = clip_tau(tau1_raw, cfg.tau_min, cfg.tau_max)

    candidates: List[Tuple[Policy, CandidateStats]] = [(cand0, stats0)]
    stats1: Optional[CandidateStats] = None
    duplicate = taus_are_duplicates(tau0, tau1, cfg.tau_duplicate_log_tolerance)
    if not duplicate:
        cand1, stats1 = _refine_and_certify(
            refiner=refiner,
            gen_critics=gen_critics,
            cert_critics=cert_critics,
            center=current,
            states=states,
            tau=tau1,
            cfg=cfg,
            q_scale=q_scale,
            data_actions=data_actions,
            gen_critic_params=gen_critic_params,
        )
        candidates.append((cand1, stats1))

    feasible: List[Tuple[Policy, CandidateStats]] = []
    for pol, st in candidates:
        if cfg.max_action_mse is None or float(st.action_mse) <= float(cfg.max_action_mse):
            feasible.append((pol, st))

    diag: Dict[str, float] = {
        "ladder_index": float(n),
        "pilot_tau": float(tau0),
        "pilot_cert": float(stats0.certificate),
        "pilot_movement": float(movement0),
        "adaptive_tau": float(tau1),
        "adaptive_cert": float("nan") if stats1 is None else float(stats1.certificate),
        "adaptive_movement": float("nan") if stats1 is None else float(stats1.action_mse),
        "c_hat": float(c_hat),
        "tau_candidate_count": float(len(candidates)),
        "tau1_raw": float(tau1_raw),
        "tau_duplicate": float(duplicate),
    }
    if (
        stats1 is not None
        and movement0 > 0
        and float(stats1.action_mse) > 0
        and abs(math.log(max(tau1, 1e-12) / max(tau0, 1e-12))) > 0
    ):
        diag["empirical_scaling_exponent"] = (
            math.log(float(stats1.action_mse)) - math.log(movement0)
        ) / (math.log(max(tau1, 1e-12)) - math.log(max(tau0, 1e-12)))
        diag["movement_target_ratio"] = float(stats1.action_mse) / max(
            float(cfg.target_action_mse), eps
        )
    else:
        diag["empirical_scaling_exponent"] = float("nan")
        diag["movement_target_ratio"] = float("nan")

    if not feasible:
        diag["selected_tau"] = float("nan")
        diag["selected_cert"] = float(stats0.certificate)
        diag["selected_movement"] = float(movement0)
        return None, stats0, [s for _, s in candidates], diag, False

    best_policy, best_stats = max(feasible, key=lambda x: x[1].certificate)
    accepted = bool(best_stats.certificate > cfg.accept_margin)
    diag["selected_tau"] = float(best_stats.tau)
    diag["selected_cert"] = float(best_stats.certificate)
    diag["selected_movement"] = float(best_stats.action_mse)

    if accepted:
        controller_n["previous_selected_tau"] = float(best_stats.tau)

    return (
        best_policy if accepted else None,
        best_stats,
        [s for _, s in candidates],
        diag,
        accepted,
    )


def calibrated_adaptive_mpi(
    base_policy: Policy,
    refiner: ProxRefiner,
    states,
    cfg: CAPOConfig,
    *,
    gen_critics: Optional[Sequence[Critic]] = None,
    cert_critics: Optional[Sequence[Critic]] = None,
    critics: Optional[Sequence[Critic]] = None,
    data_actions=None,
    q_scale: Optional[float] = None,
    initial_ladder_value: float = 0.0,
    tau_controller_state: Optional[List[dict]] = None,
    gen_critic_params=None,
) -> CAPOResult:
    if gen_critics is None:
        gen_critics = critics
    if gen_critics is None:
        raise ValueError("gen_critics (or critics) required")
    if cert_critics is None:
        cert_critics = critics if critics is not None else gen_critics

    controller = str(getattr(cfg, "tau_controller", "pilot_adaptive")).lower()
    taus = _filter_taus(cfg)
    assert cfg.n_max >= 0
    assert len(gen_critics) > 0
    assert len(cert_critics) > 0
    if controller in ("full_grid", "movement_warm_start"):
        assert len(taus) > 0

    if q_scale is None:
        q_scale = estimate_q_scale(
            cert_critics,
            states,
            actions=data_actions,
            policy=base_policy,
            eps=cfg.q_scale_eps,
        )

    policies: List[Policy] = [base_policy.copy()]
    records: List[StepRecord] = []
    selected_tau: List[float] = []
    certificates: List[float] = []
    movements: List[float] = []
    ladder_value = float(initial_ladder_value)
    current = base_policy.copy()
    any_accepted = False
    warm = controller == "movement_warm_start"
    pilot = controller == "pilot_adaptive"

    tau_controller_state = _ensure_controller_state(tau_controller_state, cfg.n_max, cfg)

    for n in range(cfg.n_max):
        if pilot:
            best_policy, best_stats, cand_stats, diag, accepted = _pilot_adaptive_step(
                n=n,
                current=current,
                refiner=refiner,
                gen_critics=gen_critics,
                cert_critics=cert_critics,
                states=states,
                cfg=cfg,
                q_scale=float(q_scale),
                data_actions=data_actions,
                controller_n=tau_controller_state[n],
                gen_critic_params=gen_critic_params,
            )
            candidates_for_record = cand_stats
        elif warm:
            candidates = []
            tau = propose_tau(
                prev_tau=float(tau_controller_state[n]["tau"]),
                prev_action_mse=float(tau_controller_state[n]["action_mse"]),
                target_action_mse=float(cfg.target_action_mse),
                tau_grid=taus,
            )
            candidate, stats = _refine_and_certify(
                refiner=refiner,
                gen_critics=gen_critics,
                cert_critics=cert_critics,
                center=current,
                states=states,
                tau=float(tau),
                cfg=cfg,
                q_scale=float(q_scale),
                data_actions=data_actions,
                gen_critic_params=gen_critic_params,
            )
            candidates.append((candidate, stats))
            tau_controller_state[n]["tau"] = float(tau)
            tau_controller_state[n]["action_mse"] = float(stats.action_mse)
            best_policy, best_stats = candidate, stats
            move_ok = True
            if cfg.max_action_mse is not None and stats.action_mse > float(cfg.max_action_mse):
                move_ok = False
            accepted = bool(best_stats.certificate > cfg.accept_margin) and move_ok
            candidates_for_record = [s for _, s in candidates]
            diag = {}
            if not accepted:
                best_policy = None
        else:
            candidates = []
            for tau in taus:
                candidate, stats = _refine_and_certify(
                    refiner=refiner,
                    gen_critics=gen_critics,
                    cert_critics=cert_critics,
                    center=current,
                    states=states,
                    tau=float(tau),
                    cfg=cfg,
                    q_scale=float(q_scale),
                    data_actions=data_actions,
                    gen_critic_params=gen_critic_params,
                )
                candidates.append((candidate, stats))

            best_idx = int(np.argmax([s.certificate for _, s in candidates]))
            best_policy, best_stats = candidates[best_idx]
            accepted = bool(best_stats.certificate > cfg.accept_margin)
            candidates_for_record = [s for _, s in candidates]
            diag = {}
            if not accepted:
                best_policy = None

        if not accepted:
            records.append(
                StepRecord(
                    step=n,
                    selected_tau=None,
                    accepted=False,
                    selected_certificate=best_stats.certificate,
                    ladder_value=ladder_value,
                    candidates=candidates_for_record,
                    diagnostics=diag,
                )
            )
            break

        any_accepted = True
        assert best_policy is not None
        current = best_policy.copy()
        ladder_value = update_reference_ladder(ladder_value, best_stats.certificate)
        policies.append(current.copy())
        selected_tau.append(float(best_stats.tau))
        certificates.append(float(best_stats.certificate))
        movements.append(float(best_stats.movement))
        records.append(
            StepRecord(
                step=n,
                selected_tau=best_stats.tau,
                accepted=True,
                selected_certificate=best_stats.certificate,
                ladder_value=ladder_value,
                candidates=candidates_for_record,
                diagnostics=diag,
            )
        )

    final = policies[-1]
    return CAPOResult(
        actor=final,
        policies=policies,
        records=records,
        final_policy=final,
        selected_n=len(policies) - 1,
        selected_tau=selected_tau,
        certificates=certificates,
        movements=movements,
        accepted=any_accepted,
    )
