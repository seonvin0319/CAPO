"""Tabular CAMPI validation with exact returns and certificate diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class TabularMDP:
    P: np.ndarray  # [S, A, S]
    R: np.ndarray  # [S, A]
    gamma: float
    init_dist: np.ndarray  # [S]


@dataclass
class TabularData:
    counts: np.ndarray  # [S, A]
    rewards_sum: np.ndarray  # [S, A]
    transitions: np.ndarray  # [S, A, S]


@dataclass
class TabularCAMPIConfig:
    n_max: int = 3
    taus: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.5)
    beta: float = 2.0
    min_count: int = 1
    accept_margin: float = 0.0
    max_iter_eval: int = 10_000
    tol: float = 1e-10


def policy_eval(mdp: TabularMDP, pi: np.ndarray, tol: float = 1e-10, max_iter: int = 10000) -> np.ndarray:
    S = mdp.R.shape[0]
    V = np.zeros(S)
    for _ in range(max_iter):
        Q = mdp.R + mdp.gamma * np.einsum("sat,t->sa", mdp.P, V)
        V_new = np.einsum("sa,sa->s", pi, Q)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new
    return V


def q_from_v(mdp: TabularMDP, V: np.ndarray) -> np.ndarray:
    return mdp.R + mdp.gamma * np.einsum("sat,t->sa", mdp.P, V)


def empirical_mdp(data: TabularData, gamma: float, init_dist: np.ndarray) -> TabularMDP:
    counts = np.maximum(data.counts, 1.0)
    R_hat = data.rewards_sum / counts
    P_hat = data.transitions / counts[:, :, None]
    P_hat = P_hat / np.maximum(P_hat.sum(axis=-1, keepdims=True), 1e-12)
    return TabularMDP(P=P_hat, R=R_hat, gamma=gamma, init_dist=init_dist)


def bernstein_bonus(data: TabularData, gamma: float, delta: float = 0.05) -> np.ndarray:
    S, A = data.counts.shape
    C = max(S * A / delta, 2.0)
    return np.sqrt(np.log(C) / np.maximum(data.counts, 1.0)) / (1.0 - gamma)


def kl_prox_update(pi_old: np.ndarray, q_lcb: np.ndarray, tau: float, support_mask: np.ndarray) -> np.ndarray:
    logits = np.log(np.maximum(pi_old, 1e-12)) + tau * q_lcb
    logits = np.where(support_mask, logits, -1e9)
    logits -= np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs


def expected_return(mdp: TabularMDP, pi: np.ndarray) -> float:
    V = policy_eval(mdp, pi)
    return float(mdp.init_dist @ V)


def campi_tabular(
    true_mdp: TabularMDP,
    data: TabularData,
    pi0: np.ndarray,
    cfg: TabularCAMPIConfig,
) -> Dict[str, object]:
    emp = empirical_mdp(data, true_mdp.gamma, true_mdp.init_dist)
    bonus = bernstein_bonus(data, true_mdp.gamma)
    support_mask = data.counts >= cfg.min_count
    state_weight = np.maximum(data.counts.sum(axis=1), 0.0)
    state_weight = state_weight / max(float(state_weight.sum()), 1.0)

    policies: List[np.ndarray] = [pi0.copy()]
    records: List[Dict[str, float]] = []
    pi = pi0.copy()
    true_J = [expected_return(true_mdp, pi)]
    ladder = 0.0

    for n in range(cfg.n_max):
        V_hat = policy_eval(emp, pi, cfg.tol, cfg.max_iter_eval)
        Q_hat = q_from_v(emp, V_hat)
        Q_lcb = Q_hat - cfg.beta * bonus

        candidates = []
        for tau in cfg.taus:
            pi_cand = kl_prox_update(pi, Q_lcb, tau, support_mask)
            per_state = np.sum((pi_cand - pi) * Q_lcb, axis=1)
            move = float(np.sum(state_weight * 0.5 * np.sum((pi_cand - pi) ** 2, axis=1)))
            gain = float(np.sum(state_weight * per_state) - move)
            true_gain = expected_return(true_mdp, pi_cand) - expected_return(true_mdp, pi)
            candidates.append((tau, pi_cand, gain, true_gain))

        tau, pi_best, cert, true_gain = max(candidates, key=lambda x: x[2])
        accepted = cert > cfg.accept_margin
        records.append(
            {
                "n": float(n),
                "tau": float(tau),
                "certificate": float(cert),
                "true_gain": float(true_gain),
                "accepted": float(accepted),
            }
        )
        if not accepted:
            break

        pi = pi_best
        ladder += max(cert, 0.0)
        policies.append(pi.copy())
        true_J.append(expected_return(true_mdp, pi))

    return {"policies": policies, "records": records, "true_J": true_J, "ladder": ladder}


def make_coverage_mdp(S: int = 5, A: int = 3, gamma: float = 0.9, seed: int = 0) -> TabularMDP:
    """Dense random MDP where offline coverage is easy to obtain."""
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(S), size=(S, A))
    R = rng.normal(loc=0.0, scale=1.0, size=(S, A))
    # Make action 0 clearly best in every state for a clean improvement story.
    R[:, 0] = np.max(R, axis=1) + 1.0
    init = np.ones(S) / S
    return TabularMDP(P=P, R=R, gamma=gamma, init_dist=init)


def collect_offline_data(
    mdp: TabularMDP,
    n_transitions: int = 8000,
    behavior_random: float = 0.5,
    seed: int = 0,
) -> Tuple[TabularData, np.ndarray]:
    rng = np.random.default_rng(seed)
    S, A = mdp.R.shape
    counts = np.zeros((S, A))
    rewards_sum = np.zeros((S, A))
    transitions = np.zeros((S, A, S))

    # Near-uniform behavior for high coverage (certificate calibration setting).
    behavior = np.ones((S, A)) / A
    behavior = (1.0 - behavior_random) * behavior + behavior_random * np.eye(A)[0]
    behavior /= behavior.sum(axis=1, keepdims=True)

    s = int(rng.choice(S, p=mdp.init_dist))
    for _ in range(n_transitions):
        a = int(rng.choice(A, p=behavior[s]))
        r = mdp.R[s, a] + 0.05 * rng.normal()
        s2 = int(rng.choice(S, p=mdp.P[s, a]))
        counts[s, a] += 1
        rewards_sum[s, a] += r
        transitions[s, a, s2] += 1
        s = s2

    data = TabularData(counts=counts, rewards_sum=rewards_sum, transitions=transitions)
    return data, behavior


def run_tabular_demo(seed: int = 0, n_transitions: int = 8000) -> Dict[str, object]:
    mdp = make_coverage_mdp(seed=seed)
    data, behavior = collect_offline_data(mdp, n_transitions=n_transitions, seed=seed)
    # Intentionally suboptimal start: uniform policy.
    S, A = mdp.R.shape
    pi0 = np.ones((S, A)) / A
    cfg = TabularCAMPIConfig(beta=1.0, min_count=1)
    out = campi_tabular(mdp, data, pi0, cfg)
    print("Tabular CAMPI demo (dense coverage MDP)")
    print(f"  true_J sequence: {[round(x, 4) for x in out['true_J']]}")
    print(f"  selected_n: {len(out['true_J']) - 1}")
    print(f"  ladder: {out['ladder']:.4f}")
    for rec in out["records"]:
        print(
            f"  n={int(rec['n'])} tau={rec['tau']:.3f} "
            f"cert={rec['certificate']:.4f} true_gain={rec['true_gain']:.4f} "
            f"accept={bool(rec['accepted'])}"
        )
    return out
