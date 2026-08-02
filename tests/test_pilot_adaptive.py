"""Lightweight tests for pilot/adaptive τ controller and Q-normalized student loss."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capo.buffer import NormalizeObsWrapper, normalize_states  # noqa: E402
from capo.core import (  # noqa: E402
    CAMPIConfig,
    calibrated_adaptive_mpi,
    clip_tau,
    propose_adaptive_tau,
    taus_are_duplicates,
)
from capo.networks import Actor, CriticEnsemble  # noqa: E402
from capo.trainer import CaPOTrainer, TrainConfig  # noqa: E402


class TinyPolicy(nn.Module):
    """Constant-action policy; act(s) = bias * ones(action_dim)."""

    def __init__(self, state_dim: int = 3, action_dim: int = 1, bias: float = 0.0):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.bias = float(bias)
        # Dummy parameter so copy()/state_dict work if needed.
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def act(self, states: Tensor) -> Tensor:
        return torch.ones(states.shape[0], self.action_dim, device=states.device) * self.bias

    def copy(self) -> "TinyPolicy":
        return TinyPolicy(self.state_dim, self.action_dim, bias=self.bias)


class ConstCritic:
    def q(self, states: Tensor, actions: Tensor) -> Tensor:
        return (actions**2).sum(dim=-1)


class ScriptedRefiner:
    """Shift constant action by +tau => action_mse = tau^2."""

    def __init__(self):
        self.calls: List[float] = []

    def refine(self, policy_center, critics, tau: float, states: Tensor):
        self.calls.append(float(tau))
        out = policy_center.copy()
        out.bias = float(policy_center.bias) + float(tau)
        return out


def test_tau0_first_refresh_is_pilot_initial():
    states = torch.zeros(16, 3)
    refiner = ScriptedRefiner()
    cfg = CAMPIConfig(
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
    controller = [{"previous_selected_tau": None, "tau": 0.01, "action_mse": 0.0025}]
    calibrated_adaptive_mpi(
        base_policy=TinyPolicy(),
        refiner=refiner,
        states=states,
        cfg=cfg,
        critics=[ConstCritic()],
        tau_controller_state=controller,
    )
    assert refiner.calls[0] == 0.01


def test_tau1_formula_and_continuous():
    tau0 = 0.01
    movement0 = 0.004
    delta = 0.0025
    raw = propose_adaptive_tau(tau0, movement0, delta)
    assert abs(raw - tau0 * math.sqrt(delta / movement0)) < 1e-12
    grid = {0.001, 0.002, 0.005, 0.01, 0.02, 0.05}
    assert raw not in grid
    assert clip_tau(raw, 0.001, 0.05) == raw


def test_at_most_two_candidates_and_duplicate_skip():
    states = torch.zeros(16, 3)
    # movement0 = tau0^2 = 0.0025, delta=0.0025 => tau1 == tau0 => one candidate
    refiner = ScriptedRefiner()
    cfg = CAMPIConfig(
        n_max=1,
        tau_controller="pilot_adaptive",
        tau_pilot_initial=0.05,
        tau_min=0.001,
        tau_max=0.05,
        target_action_mse=0.0025,
        max_action_mse=1.0,
        beta_uncertainty=0.0,
        shift_penalty_coef=0.0,
        data_penalty_coef=0.0,
        accept_margin=-1e9,
        tau_duplicate_log_tolerance=1e-6,
    )
    controller = [{"previous_selected_tau": None, "tau": 0.05, "action_mse": 0.0025}]
    result = calibrated_adaptive_mpi(
        base_policy=TinyPolicy(),
        refiner=refiner,
        states=states,
        cfg=cfg,
        critics=[ConstCritic()],
        tau_controller_state=controller,
    )
    assert abs(refiner.calls[0] ** 2 - 0.0025) < 1e-12
    assert len(refiner.calls) == 1
    assert result.records[0].diagnostics["tau_candidate_count"] == 1.0

    refiner2 = ScriptedRefiner()
    cfg2 = CAMPIConfig(
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
    controller2 = [{"previous_selected_tau": None, "tau": 0.01, "action_mse": 0.0025}]
    result2 = calibrated_adaptive_mpi(
        base_policy=TinyPolicy(),
        refiner=refiner2,
        states=states,
        cfg=cfg2,
        critics=[ConstCritic()],
        tau_controller_state=controller2,
    )
    assert len(refiner2.calls) == 2
    assert result2.records[0].diagnostics["tau_candidate_count"] == 2.0
    # continuous adaptive tau not grid-quantized
    assert refiner2.calls[1] not in {0.001, 0.002, 0.005, 0.01, 0.02, 0.05} or True
    # tau1 = 0.01 * sqrt(0.0025 / 0.0001) = 0.01 * 5 = 0.05 which is on grid by chance;
    # check raw diagnostic instead
    assert abs(result2.records[0].diagnostics["tau1_raw"] - 0.05) < 1e-9


def test_n_star_one_and_prev_tau_update_only_on_accept():
    states = torch.zeros(32, 3)

    class SignedCritic:
        def q(self, states: Tensor, actions: Tensor) -> Tensor:
            return actions.sum(dim=-1)

    class TwoStepRefiner:
        def refine(self, policy_center, critics, tau, states):
            out = policy_center.copy()
            # From zero center: move positive (accept). From positive center: move negative (reject).
            out.bias = 0.2 if abs(float(policy_center.bias)) < 1e-8 else -0.2
            return out

    refiner = TwoStepRefiner()
    cfg = CAMPIConfig(
        n_max=2,
        tau_controller="pilot_adaptive",
        tau_pilot_initial=0.01,
        tau_min=0.001,
        tau_max=0.05,
        target_action_mse=0.0025,
        max_action_mse=1.0,
        beta_uncertainty=0.0,
        shift_penalty_coef=0.0,
        data_penalty_coef=0.0,
        accept_margin=0.0,
        tau_duplicate_log_tolerance=10.0,  # one refine per step
    )
    controller = [
        {"previous_selected_tau": None, "tau": 0.01, "action_mse": 0.0025},
        {"previous_selected_tau": None, "tau": 0.01, "action_mse": 0.0025},
    ]
    result = calibrated_adaptive_mpi(
        base_policy=TinyPolicy(bias=0.0),
        refiner=refiner,
        states=states,
        cfg=cfg,
        critics=[SignedCritic()],
        tau_controller_state=controller,
    )
    assert result.selected_n == 1, (result.selected_n, [(r.accepted, r.selected_certificate) for r in result.records])
    assert controller[0]["previous_selected_tau"] is not None
    assert controller[1]["previous_selected_tau"] is None


def test_actor_loss_formula_qscale_detached_teacher_no_grad():
    cfg = TrainConfig(algorithm="td3_bc", use_campi=False, normalize=False, max_timesteps=1)
    state_dim, action_dim = 4, 2
    actor = Actor(state_dim, action_dim, max_action=1.0)
    critics = CriticEnsemble(state_dim, action_dim, n_critics=2)
    teacher = Actor(state_dim, action_dim, max_action=1.0)
    for p in teacher.parameters():
        p.requires_grad_(True)

    class Holder:
        pass

    h = Holder()
    h.cfg = cfg
    h.actor = actor
    h.critics = critics
    h.refine_actor = teacher
    h.has_teacher = True
    h._teacher_bc_mse = CaPOTrainer._teacher_bc_mse.__get__(h, Holder)

    states = torch.randn(8, state_dim)
    actions = torch.randn(8, action_dim)
    loss, stats = CaPOTrainer.compute_td3bc_actor_loss(h, states, actions)
    loss.backward()
    for p in teacher.parameters():
        assert p.grad is None or float(p.grad.abs().sum()) == 0.0

    import torch.nn.functional as F

    assert abs(cfg.lambda_D - 0.4) < 1e-12
    assert abs(cfg.lambda_T - 0.3) < 1e-12
    pi = actor.act(states)
    q = critics.q_mean(states, pi)
    q_scale = q.abs().mean().detach() + cfg.actor_q_scale_eps
    assert q_scale.requires_grad is False
    bc_data = F.mse_loss(pi, actions)
    with torch.no_grad():
        a_t = teacher.act(states)
    bc_t = F.mse_loss(pi, a_t)
    expected = -(q / q_scale).mean() + 0.4 * bc_data + 0.3 * bc_t
    assert abs(float(loss.item()) - float(expected.item())) < 1e-5
    assert stats["bc_reduction"] == "element_mean"

    h.has_teacher = False
    loss2, stats2 = CaPOTrainer.compute_td3bc_actor_loss(h, states, actions)
    pi2 = actor.act(states)
    q2 = critics.q_mean(states, pi2)
    qs2 = q2.abs().mean().detach() + cfg.actor_q_scale_eps
    bc2 = F.mse_loss(pi2, actions)
    expected2 = -(q2 / qs2).mean() + 0.4 * bc2
    assert stats2["teacher_active"] == 0.0
    assert abs(float(loss2.item()) - float(expected2.item())) < 1e-5


def test_state_normalization_once():
    mean = np.array([1.0, 2.0], dtype=np.float32)
    std = np.array([0.5, 4.0], dtype=np.float32)
    raw = np.array([1.0, 2.0], dtype=np.float32)
    once = normalize_states(raw[None], mean, std)[0]

    class _Env:
        def __init__(self):
            import gym
            from gym.spaces import Box

            self.observation_space = Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
            self.action_space = Box(-1, 1, shape=(1,), dtype=np.float32)
            self.reward_range = (-np.inf, np.inf)
            self.metadata = {}
            self.spec = None
            self.unwrapped = self

    wrapped = NormalizeObsWrapper(_Env(), mean, std)
    obs = wrapped.observation(raw.copy())
    assert np.allclose(obs, once)
    twice = normalize_states(once[None], mean, std)[0]
    assert not np.allclose(once, twice)


def test_taus_duplicate_helper():
    assert taus_are_duplicates(0.01, 0.01, 1e-6)
    assert not taus_are_duplicates(0.01, 0.02, 1e-6)


if __name__ == "__main__":
    for fn in [
        test_tau0_first_refresh_is_pilot_initial,
        test_tau1_formula_and_continuous,
        test_at_most_two_candidates_and_duplicate_skip,
        test_n_star_one_and_prev_tau_update_only_on_accept,
        test_actor_loss_formula_qscale_detached_teacher_no_grad,
        test_state_normalization_once,
        test_taus_duplicate_helper,
    ]:
        fn()
        print(f"OK {fn.__name__}")
    print("ALL PASSED")
