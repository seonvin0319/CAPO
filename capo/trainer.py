"""CAPO trainer — teacher-guided proximal refinement for offline RL.

θL = learning actor (critic targets always use θL / θL-target)
θR = CAPO refinement actor / soft teacher
When Cert accepts (N*>0), θR becomes a BC teacher for θL:
    L_actor = -mean(Q/q_scale) + λ_D BC_data(θL, a_D)
              + teacher_active * λ_T BC_teacher(θL, θR)
    with BC_* = element-wise MSE (F.mse_loss) and
    q_scale = stopgrad(mean(|Q|)+eps).
Critic never bootstraps from θR → no overwrite feedback loop.
"""
from __future__ import annotations

import copy
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR

from .buffer import (
    NormalizeObsWrapper,
    ReplayBuffer,
    load_d4rl_dataset,
    make_env,
)
from .core import (
    CAPOConfig,
    calibrated_adaptive_mpi,
    candidate_certificate,
    dataset_action_mse,
    estimate_q_scale,
)
from .networks import Actor, CriticEnsemble, ValueFunction
from .refiner import ProximalW2Refiner

EXP_ADV_MAX = 100.0


class _Tee:
    """Duplicate stdout/stderr to an additional stream (train.log)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        fn = getattr(self.streams[0], "isatty", None)
        return fn() if fn else False


@dataclass
class TrainConfig:
    algorithm: str = "td3_bc"  # td3_bc | iql | cql
    env: str = "hopper-medium-v2"
    seed: int = 0
    device: str = "cuda"
    max_timesteps: int = 1_000_000
    eval_freq: int = 5_000
    n_episodes: int = 10
    batch_size: int = 256
    buffer_size: int = 2_000_000

    discount: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    normalize: bool = True
    normalize_reward: bool = True
    n_critics: int = 4
    hidden: int = 256

    # TD3+BC
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    alpha: float = 2.5  # legacy TD3+BC coefficient (unused by new actor loss)
    # Student actor loss (TD3+BC path), BC = element-wise mean MSE:
    #   -mean(Q/q_scale) + lambda_D * BC_data + teacher_active * lambda_T * BC_teacher
    lambda_D: float = 0.4
    lambda_T: float = 0.3
    actor_q_scale_eps: float = 1e-4
    bc_reduction: str = "element_mean"  # diagnostic label; loss uses F.mse_loss

    # IQL
    iql_tau: float = 0.7
    iql_beta: float = 3.0
    vf_lr: float = 3e-4

    # CQL
    cql_alpha: float = 10.0
    cql_n_actions: int = 10
    cql_temp: float = 1.0
    cql_policy_lr: float = 3e-5
    bc_steps: int = 0
    bc_coef: float = 1.0

    # CAPO (CAPO teacher path)
    use_capo: bool = True

    n_max: int = 2
    beta_uncertainty: float = 1.0
    shift_penalty_coef: float = 0.25
    data_penalty_coef: float = 0.5
    accept_margin: float = 0.0
    tau_candidates: Tuple[float, ...] = (1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2)
    tau_max: float = 5e-2
    tau_min: float = 1e-3
    max_action_mse: Optional[float] = 0.15
    normalize_delta_q: bool = True
    split_critics_for_certification: bool = True
    refine_steps: int = 2
    refine_lr: float = 3e-4
    capo_eval_batch: int = 512
    q_scale_ema: float = 0.99
    # Tau controller: pilot_adaptive (default) | full_grid | movement_warm_start
    tau_controller: str = "pilot_adaptive"
    target_action_mse: float = 0.0025
    initial_tau: float = 0.01
    tau_pilot_initial: float = 0.01
    tau_duplicate_log_tolerance: float = 1e-6

    # Teacher-guided training schedule
    capo_period: int = 100_000  # refresh θR from θL every this many env steps
    capo_start_step: int = 100_000  # warm up critic before enabling CAPO
    teacher_hold: bool = True  # keep last teacher between refreshes if still useful
    # If N*=0 (no challenger), keep the incumbent teacher instead of disabling.
    hold_teacher_on_nstar_zero: bool = True
    # Incumbent–challenger replacement (same-time, same-critic pairwise cert).
    use_replace_gate: bool = True
    replace_cert_margin: float = 0.01  # require C^{O→N} > margin

    eval_base_actor: bool = True  # legacy name; evaluates student θL
    eval_teacher_actor: bool = True
    # Paired CRN eval at each CAPO teacher refresh (certificate calibration).
    paired_eval_episodes: int = 40
    paired_eval_seed0: int = 10_000

    # IO
    out_dir: str = "results"
    # Optional tag in run folder name, e.g. "capo" / "baseline"
    run_tag: str = ""
    save_best: bool = True
    log_interval: int = 1000
    use_wandb: bool = False
    project: str = "CAPO"
    group: str = "d4rl"

    def __post_init__(self):
        if isinstance(self.tau_candidates, list):
            self.tau_candidates = tuple(float(x) for x in self.tau_candidates)
        self.algorithm = self.algorithm.lower().replace("-", "_").replace("+", "")
        if self.algorithm in ("td3bc", "td3"):
            self.algorithm = "td3_bc"
        self.tau_controller = "pilot_adaptive"


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def asymmetric_l2_loss(u: Tensor, tau: float) -> Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


def _seed_env(env, seed: int) -> None:
    """Best-effort seed for gym / wrapped D4RL envs."""
    cur = env
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "seed"):
            try:
                cur.seed(seed)
            except Exception:
                pass
        if hasattr(cur, "action_space") and hasattr(cur.action_space, "seed"):
            try:
                cur.action_space.seed(seed)
            except Exception:
                pass
        if hasattr(cur, "observation_space") and hasattr(cur.observation_space, "seed"):
            try:
                cur.observation_space.seed(seed)
            except Exception:
                pass
        cur = getattr(cur, "env", None)


@torch.no_grad()
def eval_actor(
    env,
    actor: Actor,
    device: str,
    n_episodes: int,
    episode_seeds: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    returns = []
    if episode_seeds is None:
        episode_seeds = list(range(n_episodes))
    for seed in episode_seeds:
        _seed_env(env, int(seed))
        state = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = actor.act(s).cpu().numpy()[0]
            state, reward, done, info = env.step(action)
            ep_ret += float(reward)
        returns.append(ep_ret)
    arr = np.asarray(returns, dtype=np.float64)
    out = {
        "return_mean": float(arr.mean()),
        "return_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "returns": arr,
    }
    try:
        out["d4rl_score"] = float(env.get_normalized_score(arr.mean()) * 100.0)
    except Exception:
        pass
    return out


def _t_crit_975(df: int) -> float:
    """Two-sided 95% t critical value (fallback to normal if scipy missing)."""
    if df <= 0:
        return 1.96
    try:
        from scipy import stats  # type: ignore

        return float(stats.t.ppf(0.975, df))
    except Exception:
        # Common values; else normal approx.
        table = {1: 12.706, 2: 4.303, 5: 2.571, 9: 2.262, 19: 2.093, 29: 2.045, 39: 2.023, 49: 2.010}
        for k in sorted(table, reverse=True):
            if df >= k:
                return table[k]
        return 1.96


@torch.no_grad()
def paired_eval_actors(
    env,
    student: Actor,
    teacher: Actor,
    device: str,
    episode_seeds: Sequence[int],
) -> Dict:
    """Common-random-number eval of teacher vs frozen student snapshot.

    Diagnostics only — never used for CAPO accept/reject.
    """
    st = eval_actor(env, student, device, len(episode_seeds), episode_seeds=episode_seeds)
    te = eval_actor(env, teacher, device, len(episode_seeds), episode_seeds=episode_seeds)
    d = te["returns"] - st["returns"]
    mean_d = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    tcrit = _t_crit_975(len(d) - 1)
    ci_low, ci_high = mean_d - tcrit * se, mean_d + tcrit * se
    if ci_low > 0:
        better = True
    elif ci_high < 0:
        better = False
    else:
        better = "uncertain"

    out: Dict = {
        "episode_seeds": [int(s) for s in episode_seeds],
        "student_snapshot_returns": st["returns"].tolist(),
        "teacher_returns": te["returns"].tolist(),
        "paired_delta_returns": d.tolist(),
        "paired_delta_mean": mean_d,
        "paired_delta_se": se,
        "paired_delta_ci_low": float(ci_low),
        "paired_delta_ci_high": float(ci_high),
        "teacher_better_by_eval": better,
        "student_return_mean": float(st["return_mean"]),
        "teacher_return_mean": float(te["return_mean"]),
        "n_episodes": int(len(episode_seeds)),
    }
    if "d4rl_score" in st and "d4rl_score" in te:
        try:
            j_s = float(env.get_normalized_score(st["return_mean"]) * 100.0)
            j_t = float(env.get_normalized_score(te["return_mean"]) * 100.0)
            scale = abs(j_t - j_s) / max(abs(te["return_mean"] - st["return_mean"]), 1e-6)
            out["student_d4rl_score"] = j_s
            out["teacher_d4rl_score"] = j_t
            out["paired_delta_d4rl"] = j_t - j_s
            out["paired_delta_d4rl_se"] = float(se * scale)
            out["paired_delta_d4rl_ci_low"] = float(j_t - j_s - tcrit * se * scale)
            out["paired_delta_d4rl_ci_high"] = float(j_t - j_s + tcrit * se * scale)
        except Exception:
            pass
    return out


class CAPOTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        set_seed(cfg.seed)
        if cfg.algorithm not in ("td3_bc", "iql", "cql"):
            raise ValueError(f"Unknown algorithm: {cfg.algorithm}")

        data, stats, raw_env = load_d4rl_dataset(
            cfg.env,
            normalize=cfg.normalize,
            normalize_reward=cfg.normalize_reward,
            device=str(self.device),
        )
        self.stats = stats
        self.state_dim = data["observations"].shape[1]
        self.action_dim = data["actions"].shape[1]
        self.max_action = stats.max_action

        self.eval_env = make_env(cfg.env, seed=cfg.seed + 100)
        if cfg.normalize:
            self.eval_env = NormalizeObsWrapper(
                self.eval_env, stats.state_mean, stats.state_std
            )

        std = np.asarray(stats.state_std, dtype=np.float64)
        print(
            "[CAPO] state_normalization "
            f"enabled={bool(cfg.normalize)} "
            f"state_mean_shape={tuple(np.asarray(stats.state_mean).shape)} "
            f"state_std_min={float(std.min()):.6g} "
            f"state_std_max={float(std.max()):.6g}",
            flush=True,
        )

        self.buffer = ReplayBuffer(self.state_dim, self.action_dim, cfg.buffer_size, str(self.device))
        self.buffer.load_d4rl(data)

        n_hidden = 3 if cfg.algorithm == "cql" else 2
        # θL: learning actor (critic always bootstraps from this)
        self.actor = Actor(self.state_dim, self.action_dim, self.max_action, cfg.hidden).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        # θR: CAPO refinement / teacher (never used in critic targets)
        self.refine_actor = copy.deepcopy(self.actor)
        self.deploy_actor = copy.deepcopy(self.actor)
        self.has_teacher = False
        self.teacher_n = 0
        self.last_capo_info: Dict[str, float] = {}
        self.last_capo_step = -10**9

        self.critics = CriticEnsemble(
            self.state_dim,
            self.action_dim,
            n_critics=cfg.n_critics,
            hidden=cfg.hidden,
            n_hidden=n_hidden,
        ).to(self.device)
        self.critics_target = copy.deepcopy(self.critics)

        actor_lr = cfg.cql_policy_lr if cfg.algorithm == "cql" else cfg.actor_lr
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=cfg.critic_lr)

        self.vf: Optional[ValueFunction] = None
        self.vf_opt = None
        self.actor_lr_schedule = None
        if cfg.algorithm == "iql":
            self.vf = ValueFunction(self.state_dim, cfg.hidden).to(self.device)
            self.vf_opt = torch.optim.Adam(self.vf.parameters(), lr=cfg.vf_lr)
            self.actor_lr_schedule = CosineAnnealingLR(self.actor_opt, cfg.max_timesteps)
            if cfg.tau == 0.005:
                cfg.tau = 0.001

        self.refiner = ProximalW2Refiner(lr=cfg.refine_lr, n_steps=cfg.refine_steps)
        self.capo_cfg = CAPOConfig(
            n_max=cfg.n_max,
            beta_uncertainty=cfg.beta_uncertainty,
            shift_penalty_coef=cfg.shift_penalty_coef,
            data_penalty_coef=cfg.data_penalty_coef,
            accept_margin=cfg.accept_margin,
            tau_candidates=tuple(cfg.tau_candidates),
            tau_max=cfg.tau_max,
            tau_min=cfg.tau_min,
            max_action_mse=cfg.max_action_mse,
            normalize_delta_q=cfg.normalize_delta_q,
            tau_controller=cfg.tau_controller,
            target_action_mse=cfg.target_action_mse,
            initial_tau=cfg.initial_tau,
            tau_pilot_initial=cfg.tau_pilot_initial,
            tau_duplicate_log_tolerance=cfg.tau_duplicate_log_tolerance,
        )
        # Per ladder-index τ controller state (shared across refreshes).
        self.tau_controller_state: List[dict] = [
            {
                "previous_selected_tau": None,
                "tau": float(cfg.tau_pilot_initial),
                "action_mse": float(cfg.target_action_mse),
            }
            for _ in range(cfg.n_max)
        ]
        self.q_scale = 1.0

        self.total_it = 0
        self.actor_updates = 0
        self.best_score = -1e9
        self.best_base_score = -1e9
        stamp = time.strftime("%m%d_%H%M")
        tag = (cfg.run_tag or "").strip().replace(" ", "_")
        mid = f"{tag}_{cfg.algorithm}" if tag else cfg.algorithm
        run_name = f"{stamp}_{mid}_{cfg.env}_s{cfg.seed}"
        self.run_dir = Path(cfg.out_dir) / cfg.env / f"s{cfg.seed}" / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=2)
        self._log_fp = open(self.run_dir / "train.log", "w", buffering=1)
        sys.stdout = _Tee(sys.stdout, self._log_fp)
        sys.stderr = _Tee(sys.stderr, self._log_fp)
        raw_env.close()

    def _soft_update(self, net, target):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.mul_(1.0 - self.cfg.tau)
            tp.data.add_(self.cfg.tau * p.data)

    def _capo_allowed(self) -> bool:
        cfg = self.cfg
        if not cfg.use_capo:
            return False
        return self.total_it >= cfg.capo_start_step

    def _teacher_bc_mse(self, states: Tensor, pi: Tensor) -> Tuple[Tensor, float, float]:
        """Detached teacher BC via element-wise MSE. Returns (mse, raw, active)."""
        if not self.has_teacher or self.cfg.lambda_T <= 0:
            return pi.new_zeros(()), 0.0, 0.0
        with torch.no_grad():
            a_t = self.refine_actor.act(states)
        a_t = a_t.detach()
        mse = F.mse_loss(pi, a_t)
        return mse, float(mse.detach().item()), 1.0

    def _teacher_bc_loss(self, states: Tensor, pi: Tensor) -> Tuple[Tensor, float]:
        """Legacy helper for IQL/CQL: weighted teacher BC term."""
        mse, raw, active = self._teacher_bc_mse(states, pi)
        return self.cfg.lambda_T * mse * active, raw

    def compute_td3bc_actor_loss(
        self,
        states: Tensor,
        actions: Tensor,
        pi: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Q-normalized student actor loss (diagnostics / tests).

        L = -mean(Q / q_scale) + λ_D * BC_data + teacher_active * λ_T * BC_teacher
        with BC_* = F.mse_loss (element-wise mean) and
        q_scale = stopgrad(mean(|Q|)+eps).
        """
        cfg = self.cfg
        if pi is None:
            pi = self.actor.act(states)
        q_values = self.critics.q_mean(states, pi)
        q_scale = q_values.abs().mean().detach() + float(cfg.actor_q_scale_eps)
        q_term = -(q_values / q_scale).mean()
        bc_data = F.mse_loss(pi, actions)
        bc_teacher, bc_teacher_raw, teacher_active = self._teacher_bc_mse(states, pi)
        actor_loss = q_term + cfg.lambda_D * bc_data + teacher_active * cfg.lambda_T * bc_teacher
        stats = {
            "actor_loss": float(actor_loss.detach().item()),
            "q_term": float(q_term.detach().item()),
            "bc_data": float(bc_data.detach().item()),
            "bc_teacher": bc_teacher_raw,
            "teacher_active": float(teacher_active),
            "actor_q_scale": float(q_scale.detach().item()),
            "has_teacher": float(self.has_teacher),
            "lambda_D": float(cfg.lambda_D),
            "lambda_T": float(cfg.lambda_T),
            "bc_reduction": str(cfg.bc_reduction),
        }
        return actor_loss, stats

    # ------------------------------------------------------------------ TD3+BC
    def _update_td3bc(self, states, actions, rewards, next_states, dones) -> Dict[str, float]:
        cfg = self.cfg
        with torch.no_grad():
            noise = (torch.randn_like(actions) * cfg.policy_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
            # Critic always bootstraps from learning actor target θL̄
            next_actions = (self.actor_target.act(next_states) + noise).clamp(-self.max_action, self.max_action)
            target_q = self.critics_target.q_min(next_states, next_actions)
            target = rewards.squeeze(-1) + (1.0 - dones.squeeze(-1)) * cfg.discount * target_q

        q_pred = self.critics(states, actions)
        critic_loss = F.mse_loss(q_pred, target.unsqueeze(0).expand_as(q_pred))
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        logs = {
            "critic_loss": float(critic_loss.item()),
        }

        if self.total_it % cfg.policy_freq == 0:
            pi = self.actor.act(states)
            actor_loss, actor_stats = self.compute_td3bc_actor_loss(states, actions, pi=pi)
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            logs.update(actor_stats)
            self.actor_updates += 1
            self._maybe_capo(logs)
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critics, self.critics_target)
        return logs

    # --------------------------------------------------------------------- IQL
    def _update_iql(self, states, actions, rewards, next_states, dones) -> Dict[str, float]:
        assert self.vf is not None and self.vf_opt is not None
        cfg = self.cfg
        with torch.no_grad():
            next_v = self.vf(next_states)
            target_q = self.critics_target.q_min(states, actions)

        v = self.vf(states)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, cfg.iql_tau)
        self.vf_opt.zero_grad(set_to_none=True)
        v_loss.backward()
        self.vf_opt.step()

        targets = rewards.squeeze(-1) + (1.0 - dones.squeeze(-1)) * cfg.discount * next_v
        q_pred = self.critics(states, actions)
        q_loss = F.mse_loss(q_pred, targets.unsqueeze(0).expand_as(q_pred))
        self.critic_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.critic_opt.step()
        self._soft_update(self.critics, self.critics_target)

        exp_adv = torch.exp(cfg.iql_beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        pi = self.actor.act(states)
        bc = torch.sum((pi - actions) ** 2, dim=-1)
        teacher_term, teacher_bc = self._teacher_bc_loss(states, pi)
        actor_loss = torch.mean(exp_adv * bc) + teacher_term
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        if self.actor_lr_schedule is not None:
            self.actor_lr_schedule.step()

        logs = {
            "critic_loss": float(q_loss.item()),
            "value_loss": float(v_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "bc_teacher": teacher_bc,
            "has_teacher": float(self.has_teacher),
        }
        self.actor_updates += 1
        self._maybe_capo(logs)
        return logs

    # --------------------------------------------------------------------- CQL
    def _update_cql(self, states, actions, rewards, next_states, dones) -> Dict[str, float]:
        cfg = self.cfg
        bsz, act_dim = actions.shape
        with torch.no_grad():
            next_actions = self.actor.act(next_states)  # θL only
            target_q = self.critics_target.q_min(next_states, next_actions)
            td_target = rewards.squeeze(-1) + (1.0 - dones.squeeze(-1)) * cfg.discount * target_q

        q_data = self.critics(states, actions)
        td_loss = F.mse_loss(q_data, td_target.unsqueeze(0).expand_as(q_data))

        rand_actions = actions.new_empty(bsz, cfg.cql_n_actions, act_dim).uniform_(-1.0, 1.0)
        q_rand = self.critics(states, rand_actions)
        with torch.no_grad():
            pi_actions = self.actor.act(states)
        q_pi = self.critics(states, pi_actions)
        random_density = math.log(0.5**act_dim)
        cat = torch.cat([q_rand - random_density, q_pi.unsqueeze(-1)], dim=-1)
        cql_ood = torch.logsumexp(cat / cfg.cql_temp, dim=-1) * cfg.cql_temp
        cql_diff = (cql_ood - q_data).mean()
        critic_loss = td_loss + cfg.cql_alpha * cql_diff

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self._soft_update(self.critics, self.critics_target)

        pi = self.actor.act(states)
        q_pi_det = self.critics.q_mean(states, pi)
        if self.total_it <= cfg.bc_steps:
            actor_loss = F.mse_loss(pi, actions)
            teacher_bc = 0.0
        else:
            bc_data = cfg.bc_coef * F.mse_loss(pi, actions)
            teacher_term, teacher_bc = self._teacher_bc_loss(states, pi)
            actor_loss = -q_pi_det.mean() + bc_data + teacher_term

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        logs = {
            "critic_loss": float(critic_loss.item()),
            "cql_diff": float(cql_diff.item()),
            "actor_loss": float(actor_loss.item()),
            "bc_teacher": teacher_bc,
            "has_teacher": float(self.has_teacher),
        }
        self.actor_updates += 1
        self._maybe_capo(logs)
        return logs

    def _split_critics(self):
        adapters = self.critics.as_adapters()
        if not self.cfg.split_critics_for_certification or len(adapters) < 2:
            return adapters, adapters
        mid = max(1, len(adapters) // 2)
        return adapters[:mid], adapters[mid:]

    def _maybe_capo(self, logs: Dict[str, float]):
        cfg = self.cfg
        if not cfg.use_capo:
            self.deploy_actor.load_state_dict(self.actor.state_dict())
            return

        # Only schedule after warm-start; first eligible step always refreshes.
        if self.total_it < cfg.capo_start_step:
            logs["has_teacher"] = float(self.has_teacher)
            logs["capo_selected_n"] = float(self.teacher_n if self.has_teacher else 0)
            return
        first = self.last_capo_step < cfg.capo_start_step
        due = first or (self.total_it - self.last_capo_step) >= cfg.capo_period
        if not due:
            logs["has_teacher"] = float(self.has_teacher)
            logs["capo_selected_n"] = float(self.teacher_n if self.has_teacher else 0)
            return

        cert_states, cert_actions, _, _, _ = self.buffer.sample(cfg.capo_eval_batch)
        info = self._run_capo(cert_states, cert_actions)
        self.last_capo_info = info
        logs.update(info)
        self.last_capo_step = self.total_it
        acc_c = info.get("capo_accepted_cert", float("nan"))
        stop_c = info.get("capo_stop_cert", float("nan"))
        delta = info.get("paired_delta_d4rl", float("nan"))
        print(
            f"[CAPO] step={self.total_it} N*={int(info.get('capo_selected_n', 0))} "
            f"accepted={int(info.get('capo_accepted', 0))} "
            f"accepted_cert={acc_c:.5f} stop_cert={stop_c:.5f} "
            f"tau*={info.get('capo_selected_tau', float('nan'))} "
            f"pairedΔJ={delta if delta == delta else float('nan'):.2f} "
            f"teacher={int(self.has_teacher)}",
            flush=True,
        )
        ladder_path = self.run_dir / "capo_ladder.jsonl"
        if ladder_path.exists():
            last_line = ladder_path.read_text().strip().splitlines()[-1]
            ladder = json.loads(last_line)
            for rec in ladder.get("records", []):
                parts = [
                    f"τ={c['tau']:g}:C={c['cert']:+.4f}"
                    for c in rec.get("candidates", [])
                ]
                flag = "ACCEPT" if rec["accepted"] else "STOP"
                print(
                    f"  n={rec['n']} {flag} selected_τ={rec.get('selected_tau')} "
                    f"[{' | '.join(parts)}]",
                    flush=True,
                )

    def _pairwise_lcb_cert(
        self,
        current_policy,
        candidate_policy,
        states: Tensor,
        data_actions: Tensor,
        cert_critics,
        q_scale: float,
    ) -> float:
        """LCB certificate for Q(candidate) - Q(current) under current cert critics."""
        stats = candidate_certificate(
            cert_critics=cert_critics,
            current_policy=current_policy,
            candidate_policy=candidate_policy,
            states=states,
            tau=0.0,
            cfg=self.capo_cfg,
            q_scale=q_scale,
            data_actions=data_actions,
        )
        return float(stats.certificate)

    def _run_capo(self, states: Tensor, data_actions: Tensor) -> Dict[str, float]:
        gen_critics, cert_critics = self._split_critics()
        batch_scale = estimate_q_scale(
            cert_critics, states, actions=data_actions, eps=self.capo_cfg.q_scale_eps
        )
        ema = float(self.cfg.q_scale_ema)
        self.q_scale = ema * self.q_scale + (1.0 - ema) * batch_scale

        # Always refine from current learning actor θL
        result = calibrated_adaptive_mpi(
            base_policy=self.actor,
            refiner=self.refiner,
            states=states,
            cfg=self.capo_cfg,
            gen_critics=gen_critics,
            cert_critics=cert_critics,
            data_actions=data_actions,
            q_scale=self.q_scale,
            tau_controller_state=self.tau_controller_state,
        )
        # --- Certificate bookkeeping (do NOT use last rejected step as "best") ---
        accepted_recs = [r for r in result.records if r.accepted]
        stop_rec = None
        if result.records and not result.records[-1].accepted:
            stop_rec = result.records[-1]
        info = {
            "capo_selected_n": float(result.selected_n),
            "capo_accepted": float(result.accepted),
            "capo_q_scale": float(self.q_scale),
            "capo_ladder": float(result.records[-1].ladder_value) if result.records else 0.0,
            "capo_gated": 0.0,
            "capo_n_records": float(len(result.records)),
        }
        if accepted_recs:
            # Cert of the last accepted ladder step (= cert of selected policy vs prior).
            last_acc = accepted_recs[-1]
            info["capo_accepted_cert"] = float(last_acc.selected_certificate)
            info["capo_selected_tau"] = float(last_acc.selected_tau) if last_acc.selected_tau is not None else float("nan")
            info["capo_sum_accepted_cert"] = float(sum(r.selected_certificate for r in accepted_recs))
            # Backward-compatible aliases: these now mean accepted, not last attempt.
            info["capo_best_cert"] = float(last_acc.selected_certificate)
            info["capo_last_cert"] = float(last_acc.selected_certificate)
            if last_acc.candidates:
                best_acc = max(last_acc.candidates, key=lambda c: c.certificate)
                info["capo_best_move"] = float(best_acc.movement)
                info["capo_best_amse"] = float(best_acc.action_mse)
        else:
            info["capo_accepted_cert"] = float("nan")
            info["capo_sum_accepted_cert"] = 0.0
        if stop_rec is not None:
            info["capo_stop_cert"] = float(stop_rec.selected_certificate)
            if stop_rec.candidates:
                stop_best = max(stop_rec.candidates, key=lambda c: c.certificate)
                info["capo_stop_best_cert"] = float(stop_best.certificate)
                info["capo_stop_tau"] = float(stop_best.tau)
        else:
            info["capo_stop_cert"] = float("nan")
        if result.movements:
            info["capo_movement"] = float(result.movements[-1])
        if result.selected_tau:
            info["capo_tau_max_selected"] = float(max(result.selected_tau))

        # Persist full (n, τ) ladder for adaptive-selection analysis.
        ladder_path = self.run_dir / "capo_ladder.jsonl"
        ladder_row = {
            "step": int(self.total_it),
            "selected_n": int(result.selected_n),
            "accepted": bool(result.accepted),
            "selected_taus": list(result.selected_tau),
            "accepted_certificates": list(result.certificates),
            "records": [
                {
                    "n": int(rec.step) + 1,
                    "accepted": bool(rec.accepted),
                    "selected_tau": rec.selected_tau,
                    "selected_certificate": float(rec.selected_certificate),
                    "diagnostics": {
                        k: (None if isinstance(v, float) and v != v else v)
                        for k, v in (rec.diagnostics or {}).items()
                    },
                    "candidates": [
                        {
                            "tau": float(c.tau),
                            "cert": float(c.certificate),
                            "gain": float(c.estimated_gain),
                            "unc": float(c.uncertainty),
                            "shift": float(c.shift_penalty),
                            "data": float(c.data_penalty),
                            "move": float(c.movement),
                            "amse": float(c.action_mse),
                        }
                        for c in rec.candidates
                    ],
                }
                for rec in result.records
            ],
        }
        with open(ladder_path, "a") as f:
            f.write(json.dumps(ladder_row) + "\n")

        # Teacher path — incumbent–challenger replacement (offline cert only)
        self._apply_teacher_replace_gate(
            result=result,
            states=states,
            data_actions=data_actions,
            cert_critics=cert_critics,
            info=info,
        )
        self.deploy_actor.load_state_dict(self.actor.state_dict())

        info["has_teacher"] = float(self.has_teacher)
        info["teacher_n"] = float(self.teacher_n)
        return info

    def _apply_teacher_replace_gate(
        self,
        result,
        states: Tensor,
        data_actions: Tensor,
        cert_critics,
        info: Dict[str, float],
    ) -> None:
        """Incumbent–challenger gate (offline cert only). Paired CRN = diagnostics.

        When N*=0 (no challenger) and an incumbent exists, default is keep_old
        (hold_teacher_on_nstar_zero) so soft-BC guidance is not hard-disabled.
        """
        cfg = self.cfg
        margin = float(cfg.replace_cert_margin)
        has_new = bool(result.selected_n > 0 and result.accepted)
        has_old = bool(self.has_teacher)
        pi_new = result.final_policy if has_new else None
        # Snapshot incumbent before any weight change.
        pi_old = copy.deepcopy(self.refine_actor) if has_old else None
        old_n = int(self.teacher_n) if has_old else 0

        c_sn = float("nan")
        c_so = float("nan")
        c_on = float("nan")
        if has_new:
            c_sn = self._pairwise_lcb_cert(
                self.actor, pi_new, states, data_actions, cert_critics, self.q_scale
            )
        if has_old and pi_old is not None:
            c_so = self._pairwise_lcb_cert(
                self.actor, pi_old, states, data_actions, cert_critics, self.q_scale
            )
        if has_old and has_new and pi_old is not None:
            c_on = self._pairwise_lcb_cert(
                pi_old, pi_new, states, data_actions, cert_critics, self.q_scale
            )

        if not cfg.use_replace_gate:
            if has_new:
                decision = "replace_new"
            elif has_old and cfg.teacher_hold:
                decision = "keep_old"
            else:
                decision = "disable_teacher"
        elif not has_new:
            # N*=0: no certified challenger.
            if has_old and cfg.hold_teacher_on_nstar_zero:
                decision = "keep_old"
            elif has_old and cfg.teacher_hold and c_so > margin:
                decision = "keep_old"
            else:
                decision = "disable_teacher"
        elif not has_old:
            decision = "replace_new" if c_sn > margin else "disable_teacher"
        else:
            if c_sn > margin and c_on > margin:
                decision = "replace_new"
            elif c_so > margin:
                decision = "keep_old"
            elif c_sn > margin and c_so <= margin:
                decision = "replace_new"  # stale incumbent
            else:
                decision = "disable_teacher"

        # Diagnostics BEFORE applying weights (paired never feeds the gate).
        refresh_row: Dict = {
            "refresh_step": int(self.total_it),
            "challenger_n": int(result.selected_n) if has_new else 0,
            "incumbent_n": old_n,
            "selected_tau_per_step": list(result.selected_tau) if has_new else [],
            "accepted_cert_per_step": list(result.certificates) if has_new else [],
            "ladder_accepted": has_new,
            "stop_cert": info.get("capo_stop_cert"),
            "student_to_new_cert": None if c_sn != c_sn else float(c_sn),
            "student_to_old_cert": None if c_so != c_so else float(c_so),
            "old_to_new_replace_cert": None if c_on != c_on else float(c_on),
            "replacement_decision": decision,
            "q_scale": float(self.q_scale),
            "use_replace_gate": bool(cfg.use_replace_gate),
            "replace_cert_margin": margin,
            "movement": float(result.movements[-1]) if result.movements else None,
        }
        if has_new and pi_new is not None:
            with torch.no_grad():
                refresh_row["dataset_amse"] = float(
                    dataset_action_mse(pi_new, states, data_actions)
                )

        if cfg.paired_eval_episodes > 0:
            seeds = [cfg.paired_eval_seed0 + i for i in range(cfg.paired_eval_episodes)]
            if has_new and pi_new is not None:
                paired_new = paired_eval_actors(
                    self.eval_env, self.actor, pi_new, str(self.device), seeds
                )
                refresh_row["paired_delta_d4rl_new"] = paired_new.get("paired_delta_d4rl")
                refresh_row["paired_delta_mean_new"] = paired_new.get("paired_delta_mean")
                refresh_row["paired_delta_ci_low_new"] = paired_new.get("paired_delta_ci_low")
                refresh_row["paired_delta_ci_high_new"] = paired_new.get("paired_delta_ci_high")
                refresh_row["teacher_better_by_eval_new"] = paired_new.get("teacher_better_by_eval")
                refresh_row["new_teacher_d4rl"] = paired_new.get("teacher_d4rl_score")
                refresh_row["student_snapshot_d4rl"] = paired_new.get("student_d4rl_score")
                refresh_row["student_snapshot_returns"] = paired_new.get("student_snapshot_returns")
                refresh_row["new_teacher_returns"] = paired_new.get("teacher_returns")
                refresh_row["paired_delta_returns_new"] = paired_new.get("paired_delta_returns")
            if has_old and pi_old is not None:
                paired_old = paired_eval_actors(
                    self.eval_env, self.actor, pi_old, str(self.device), seeds
                )
                refresh_row["paired_delta_d4rl_old"] = paired_old.get("paired_delta_d4rl")
                refresh_row["paired_delta_mean_old"] = paired_old.get("paired_delta_mean")
                refresh_row["paired_delta_ci_low_old"] = paired_old.get("paired_delta_ci_low")
                refresh_row["paired_delta_ci_high_old"] = paired_old.get("paired_delta_ci_high")
                refresh_row["teacher_better_by_eval_old"] = paired_old.get("teacher_better_by_eval")
                refresh_row["old_teacher_d4rl"] = paired_old.get("teacher_d4rl_score")
                refresh_row["old_teacher_returns"] = paired_old.get("teacher_returns")
                refresh_row["paired_delta_returns_old"] = paired_old.get("paired_delta_returns")
            # Oracle regret among available challengers (diagnostic).
            gains = []
            if refresh_row.get("paired_delta_d4rl_old") is not None:
                gains.append(("old", float(refresh_row["paired_delta_d4rl_old"])))
            if refresh_row.get("paired_delta_d4rl_new") is not None:
                gains.append(("new", float(refresh_row["paired_delta_d4rl_new"])))
            if gains:
                oracle = max(gains, key=lambda x: x[1])
                refresh_row["oracle_best"] = oracle[0]
                if decision == "replace_new":
                    selected_g = refresh_row.get("paired_delta_d4rl_new")
                elif decision == "keep_old":
                    selected_g = refresh_row.get("paired_delta_d4rl_old")
                else:
                    selected_g = 0.0
                if selected_g is not None:
                    refresh_row["replace_regret_d4rl"] = float(oracle[1] - float(selected_g))

        # Apply decision (offline only).
        if decision == "replace_new":
            assert pi_new is not None
            self.refine_actor.load_state_dict(pi_new.state_dict())
            self.has_teacher = True
            self.teacher_n = int(result.selected_n)
        elif decision == "keep_old":
            self.has_teacher = True
            self.teacher_n = old_n
        else:
            self.has_teacher = False
            self.teacher_n = 0

        refresh_row["accepted_by_cert"] = decision == "replace_new"
        refresh_row["has_teacher_after"] = bool(self.has_teacher)
        refresh_row["teacher_n_after"] = int(self.teacher_n)

        info["student_to_new_cert"] = float(c_sn) if c_sn == c_sn else float("nan")
        info["student_to_old_cert"] = float(c_so) if c_so == c_so else float("nan")
        info["old_to_new_replace_cert"] = float(c_on) if c_on == c_on else float("nan")
        info["replacement_decision_code"] = {
            "replace_new": 1.0,
            "keep_old": 0.0,
            "disable_teacher": -1.0,
        }[decision]
        if refresh_row.get("paired_delta_d4rl_new") is not None:
            info["paired_delta_d4rl"] = float(refresh_row["paired_delta_d4rl_new"])
        elif refresh_row.get("paired_delta_d4rl_old") is not None:
            info["paired_delta_d4rl"] = float(refresh_row["paired_delta_d4rl_old"])

        with open(self.run_dir / "capo_refresh.jsonl", "a") as f:
            f.write(json.dumps(refresh_row) + "\n")

        print(
            f"  replace_gate decision={decision} "
            f"C_S→N={c_sn if c_sn == c_sn else float('nan'):+.5f} "
            f"C_S→O={c_so if c_so == c_so else float('nan'):+.5f} "
            f"C_O→N={c_on if c_on == c_on else float('nan'):+.5f}",
            flush=True,
        )

    def train_step(self) -> Dict[str, float]:
        self.total_it += 1
        batch = self.buffer.sample(self.cfg.batch_size)
        if self.cfg.algorithm == "td3_bc":
            return self._update_td3bc(*batch)
        if self.cfg.algorithm == "iql":
            return self._update_iql(*batch)
        return self._update_cql(*batch)

    def train(self) -> Dict[str, float]:
        cfg = self.cfg
        metrics_path = self.run_dir / "metrics.jsonl"
        t0 = time.time()
        last_eval: Dict[str, float] = {}

        print(
            f"[CAPO] algo={cfg.algorithm} env={cfg.env} device={self.device} "
            f"n_max={cfg.n_max} period={cfg.capo_period} "
            f"start={cfg.capo_start_step} λ_D={cfg.lambda_D} λ_T={cfg.lambda_T} "
            f"bc_reduction={cfg.bc_reduction} "
            f"replace_gate={cfg.use_replace_gate} margin={cfg.replace_cert_margin} "
            f"tau_ctrl={cfg.tau_controller} δ={cfg.target_action_mse} "
            f"tau_pilot0={cfg.tau_pilot_initial}",
            flush=True,
        )
        print(f"[CAPO] run_dir={self.run_dir}", flush=True)

        for t in range(1, cfg.max_timesteps + 1):
            logs = self.train_step()
            if t % cfg.log_interval == 0:
                elapsed = time.time() - t0
                msg = (
                    f"t={t} critic={logs.get('critic_loss', 0):.4f}"
                    f" actor={logs.get('actor_loss', 0):.4f}"
                    f" teacher={int(logs.get('has_teacher', 0))}"
                    f" N*={int(logs.get('capo_selected_n', logs.get('teacher_n', 0)))}"
                    f" ({elapsed:.1f}s)"
                )
                print(msg, flush=True)

            if t % cfg.eval_freq == 0 or t == cfg.max_timesteps:
                # Curve eval: student θL (teacher-guided after CAPO starts — not a pure TD3+BC control)
                curve_seeds = [cfg.seed * 100_000 + t + i for i in range(cfg.n_episodes)]
                student_eval = eval_actor(
                    self.eval_env, self.actor, str(self.device), cfg.n_episodes, episode_seeds=curve_seeds
                )
                row = {
                    "step": t,
                    "return_mean": student_eval["return_mean"],
                    "return_std": student_eval["return_std"],
                    **{
                        k: float(v)
                        for k, v in logs.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)
                    },
                    **{
                        k: float(v)
                        for k, v in self.last_capo_info.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)
                    },
                    "has_teacher": float(self.has_teacher),
                    "capo_selected_n": float(
                        self.teacher_n
                        if self.has_teacher
                        else self.last_capo_info.get("capo_selected_n", 0.0)
                    ),
                }
                if "d4rl_score" in student_eval:
                    row["student_d4rl_score"] = student_eval["d4rl_score"]
                    # Legacy aliases (student ≠ pure TD3+BC once teacher loss is active).
                    row["d4rl_score"] = student_eval["d4rl_score"]
                    row["base_d4rl_score"] = student_eval["d4rl_score"]
                    self.best_base_score = max(self.best_base_score, student_eval["d4rl_score"])

                teacher_eval = None
                if cfg.eval_teacher_actor and self.has_teacher:
                    teacher_eval = eval_actor(
                        self.eval_env,
                        self.refine_actor,
                        str(self.device),
                        cfg.n_episodes,
                        episode_seeds=curve_seeds,
                    )
                    row["teacher_return_mean"] = teacher_eval["return_mean"]
                    if "d4rl_score" in teacher_eval:
                        row["teacher_d4rl_score"] = teacher_eval["d4rl_score"]
                        row["curve_delta_d4rl"] = (
                            teacher_eval["d4rl_score"] - student_eval.get("d4rl_score", float("nan"))
                        )

                with open(metrics_path, "a") as f:
                    f.write(json.dumps(row) + "\n")

                score = row.get("student_d4rl_score", row.get("d4rl_score", row["return_mean"]))
                msg = (
                    f"[eval] step={t} student_d4rl={row.get('student_d4rl_score', float('nan')):.2f}"
                    f" teacher={'on' if self.has_teacher else 'off'} N*={self.teacher_n}"
                )
                if teacher_eval is not None and "d4rl_score" in teacher_eval:
                    msg += f" teacher_d4rl={teacher_eval['d4rl_score']:.2f}"
                print(msg, flush=True)
                last_eval = row

                if cfg.save_best and score > self.best_score:
                    self.best_score = score
                    torch.save(
                        {
                            "actor": self.actor.state_dict(),
                            "refine_actor": self.refine_actor.state_dict(),
                            "critics": self.critics.state_dict(),
                            "vf": None if self.vf is None else self.vf.state_dict(),
                            "has_teacher": self.has_teacher,
                            "teacher_n": self.teacher_n,
                            "step": t,
                            "score": score,
                            "config": asdict(cfg),
                        },
                        self.run_dir / "best.pt",
                    )

        # Always persist the 1M (max_timesteps) weights — not only best.pt.
        final_payload = {
            "actor": self.actor.state_dict(),
            "refine_actor": self.refine_actor.state_dict(),
            "critics": self.critics.state_dict(),
            "vf": None if self.vf is None else self.vf.state_dict(),
            "has_teacher": self.has_teacher,
            "teacher_n": self.teacher_n,
            "step": int(cfg.max_timesteps),
            "score": last_eval.get(
                "student_d4rl_score", last_eval.get("d4rl_score", float("nan"))
            )
            if last_eval
            else float("nan"),
            "config": asdict(cfg),
        }
        final_path = self.run_dir / "final.pt"
        ckpt_path = self.run_dir / f"checkpoint_{int(cfg.max_timesteps)}.pt"
        torch.save(final_payload, final_path)
        torch.save(final_payload, ckpt_path)
        print(
            f"[ckpt] saved 1M/final weights → {final_path.name}, {ckpt_path.name} "
            f"(best.pt step/score may differ)",
            flush=True,
        )
        summary = {
            "algorithm": cfg.algorithm,
            "env": cfg.env,
            "seed": cfg.seed,
            "capo_mode": "teacher",
            "best_student_score": self.best_score,
            "best_learn_score": self.best_score,  # legacy alias
            "best_base_score": self.best_base_score,  # legacy alias (= student)
            "final_eval": last_eval,
            "run_dir": str(self.run_dir),
            "elapsed_sec": time.time() - t0,
            "note": (
                "student_d4rl_score is θL under teacher guidance after CAPO starts; "
                "use --no_capo for pure TD3+BC control. "
                "Certificate calibration: see capo_refresh.jsonl paired ΔJ."
            ),
        }
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[done] {summary}", flush=True)
        return summary
