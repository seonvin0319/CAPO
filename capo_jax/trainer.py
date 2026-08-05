"""CAPO trainer (JAX / Flax / Optax) — teacher-guided offline RL.

θL = learning actor (critic targets always use θL / θL-target)
θR = CAPO refinement actor / soft teacher
"""
from __future__ import annotations

import json
import math
import pickle
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct

from .buffer import (
    NormalizeObsWrapper,
    ReplayBuffer,
    load_d4rl_dataset,
    make_env,
    resolve_jax_device,
)
from .core import (
    CAPOConfig,
    calibrated_adaptive_mpi,
    candidate_certificate,
    dataset_action_mse,
    estimate_q_scale,
)
from .gate import teacher_bc_components
from .gate_runtime import apply_teacher_replace_gate
from .networks import (
    Actor,
    ActorPolicy,
    CriticEnsemble,
    CriticEnsembleAdapter,
    ValueFunction,
    q_mean,
    q_min,
    slice_ensemble_params,
)
from .refiner import ProximalW2Refiner
from .td3_objectives import td3bc_legacy_actor_components

EXP_ADV_MAX = 100.0


class _Tee:
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
    algorithm: str = "td3_bc"
    env: str = "hopper-medium-v2"
    seed: int = 0
    rng_seed: Optional[int] = None
    device: str = "cuda"
    max_timesteps: int = 1_000_000
    eval_freq: int = 5_000
    n_episodes: int = 10
    batch_size: int = 256
    # Fuse this many replay samples + optimizer updates into one XLA dispatch.
    # Logging, evaluation, and CAPO refresh boundaries still run at exact steps.
    jit_update_chunk: int = 32
    buffer_size: int = 2_000_000

    discount: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    normalize: bool = True
    normalize_reward: bool = True
    n_critics: int = 4
    hidden: int = 256

    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    alpha: float = 2.5
    td3_actor_objective: str = "capo_student"  # capo_student or td3bc_legacy
    lambda_D: float = 0.4
    lambda_T: float = 0.3
    actor_q_scale_eps: float = 1e-4
    bc_reduction: str = "element_mean"
    teacher_bc_mode: str = "uniform"  # uniform or statewise_lcb_mask

    iql_tau: float = 0.7
    iql_beta: float = 3.0
    vf_lr: float = 3e-4

    cql_alpha: float = 10.0
    cql_n_actions: int = 10
    cql_temp: float = 1.0
    cql_policy_lr: float = 3e-5
    bc_steps: int = 0
    bc_coef: float = 1.0

    use_capo: bool = True
    n_max: int = 2
    beta_uncertainty: float = 1.0
    shift_penalty_coef: float = 0.25
    data_penalty_coef: float = 0.5
    accept_margin: float = 0.0
    tau_max: float = 5e-2
    tau_min: float = 1e-3
    max_action_mse: Optional[float] = 0.15
    normalize_delta_q: bool = True
    split_critics_for_certification: bool = True
    refine_steps: int = 2
    refine_lr: float = 3e-4
    capo_eval_batch: int = 512
    q_scale_ema: float = 0.99
    tau_controller: str = "pilot_adaptive"
    target_action_mse: float = 0.0025
    initial_tau: float = 0.01
    tau_pilot_initial: float = 0.01
    tau_duplicate_log_tolerance: float = 1e-6

    capo_period: int = 100_000
    capo_start_step: int = 100_000
    teacher_hold: bool = True
    hold_teacher_on_nstar_zero: bool = True
    use_replace_gate: bool = True
    replace_cert_margin: float = 0.0
    # Stale means current old invalid and current new valid.
    stale_incumbent_action: str = "replace_new"  # disable|quarantine|keep_old|replace_new
    nstar_zero_action: str = "revalidate_current"
    save_refresh_actors: bool = False

    eval_base_actor: bool = True
    eval_teacher_actor: bool = True
    paired_eval_episodes: int = 40
    paired_eval_seed0: int = 10_000

    out_dir: str = "results"
    run_tag: str = ""
    run_id: str = ""
    sweep_name: str = ""
    resume_run_dir: str = ""
    heartbeat_freq: int = 10_000
    save_best: bool = True
    # Periodic actor snapshots consumed by post-hoc BC distillation. 0 disables.
    save_ckpt_freq: int = 50_000
    checkpoint_on_signal: bool = True
    log_interval: int = 1000
    use_wandb: bool = False
    project: str = "CAPO"
    group: str = "d4rl"

    def __post_init__(self):
        self.algorithm = self.algorithm.lower().replace("-", "_").replace("+", "")
        if self.algorithm in ("td3bc", "td3"):
            self.algorithm = "td3_bc"
        if self.stale_incumbent_action == "disable_teacher":
            self.stale_incumbent_action = "disable"
        if self.stale_incumbent_action not in (
            "disable",
            "quarantine",
            "replace_new",
            "keep_old",
        ):
            raise ValueError(
                "stale_incumbent_action must be disable, quarantine, keep_old, or replace_new"
            )
        if self.nstar_zero_action not in ("legacy_hold", "revalidate_current"):
            raise ValueError("nstar_zero_action must be legacy_hold or revalidate_current")
        if self.td3_actor_objective not in ("capo_student", "td3bc_legacy"):
            raise ValueError("td3_actor_objective must be capo_student or td3bc_legacy")
        if self.tau_controller != "pilot_adaptive":
            raise ValueError("tau_controller is fixed to pilot_adaptive")
        if self.teacher_bc_mode not in ("uniform", "statewise_lcb_mask"):
            raise ValueError("teacher_bc_mode must be uniform or statewise_lcb_mask")
        self.jit_update_chunk = max(1, int(self.jit_update_chunk))
        self.save_ckpt_freq = max(0, int(self.save_ckpt_freq))


@struct.dataclass
class TrainState:
    actor_params: Any
    actor_target_params: Any
    actor_opt_state: Any
    critic_params: Any
    critic_target_params: Any
    critic_opt_state: Any
    teacher_params: Any
    quarantined_params: Any
    vf_params: Any
    vf_opt_state: Any
    has_teacher: jnp.ndarray  # 0/1 scalar
    teacher_n: jnp.ndarray
    teacher_tau: jnp.ndarray
    has_quarantined: jnp.ndarray
    quarantined_n: jnp.ndarray
    quarantined_tau: jnp.ndarray
    q_scale: jnp.ndarray
    total_it: jnp.ndarray
    actor_updates: jnp.ndarray
    last_capo_step: jnp.ndarray
    rng: Any


def soft_update(online, target, tau: float):
    return jax.tree_util.tree_map(lambda o, t: (1.0 - tau) * t + tau * o, online, target)


def asymmetric_l2_loss(u, tau: float):
    return jnp.mean(jnp.abs(tau - (u < 0).astype(u.dtype)) * u**2)


def _seed_env(env, seed: int) -> None:
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


def eval_actor(env, actor_apply, actor_params, n_episodes: int, episode_seeds=None) -> Dict[str, float]:
    returns = []
    if episode_seeds is None:
        episode_seeds = list(range(n_episodes))
    for seed in episode_seeds:
        _seed_env(env, int(seed))
        state = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            s = jnp.asarray(state, dtype=jnp.float32)[None, :]
            action = np.asarray(actor_apply(actor_params, s))[0]
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
    if df <= 0:
        return 1.96
    try:
        from scipy import stats  # type: ignore

        return float(stats.t.ppf(0.975, df))
    except Exception:
        table = {1: 12.706, 2: 4.303, 5: 2.571, 9: 2.262, 19: 2.093, 29: 2.045, 39: 2.023, 49: 2.010}
        for k in sorted(table, reverse=True):
            if df >= k:
                return table[k]
        return 1.96


def paired_eval_actors(env, student_apply, student_params, teacher_apply, teacher_params, episode_seeds):
    st = eval_actor(env, student_apply, student_params, len(episode_seeds), episode_seeds=episode_seeds)
    te = eval_actor(env, teacher_apply, teacher_params, len(episode_seeds), episode_seeds=episode_seeds)
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


def _save_ckpt(path: Path, payload: dict):
    """Atomically publish a pickle so interruption cannot corrupt the old file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=5)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


class CAPOTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        if cfg.algorithm not in ("td3_bc", "iql", "cql"):
            raise ValueError(f"Unknown algorithm: {cfg.algorithm}")

        self.jax_device = resolve_jax_device(cfg.device)
        effective_seed = cfg.seed if cfg.rng_seed is None else int(cfg.rng_seed)
        self.rng = jax.device_put(jax.random.PRNGKey(effective_seed), self.jax_device)
        np.random.seed(effective_seed)

        data, stats, raw_env = load_d4rl_dataset(
            cfg.env,
            normalize=cfg.normalize,
            normalize_reward=cfg.normalize_reward,
            device=cfg.device,
        )
        self.stats = stats
        self.state_dim = int(data["observations"].shape[1])
        self.action_dim = int(data["actions"].shape[1])
        self.max_action = float(stats.max_action)

        self.eval_env = make_env(cfg.env, seed=cfg.seed + 100)
        if cfg.normalize:
            self.eval_env = NormalizeObsWrapper(self.eval_env, stats.state_mean, stats.state_std)

        std = np.asarray(stats.state_std, dtype=np.float64)
        print(
            "[CAPO-JAX] state_normalization "
            f"enabled={bool(cfg.normalize)} "
            f"state_mean_shape={tuple(np.asarray(stats.state_mean).shape)} "
            f"state_std_min={float(std.min()):.6g} "
            f"state_std_max={float(std.max()):.6g}",
            flush=True,
        )

        self.buffer = ReplayBuffer(
            self.state_dim, self.action_dim, cfg.buffer_size, self.jax_device
        )
        self.buffer.load_d4rl(data)

        n_hidden = 3 if cfg.algorithm == "cql" else 2
        self.actor_mod = Actor(action_dim=self.action_dim, max_action=self.max_action, hidden=cfg.hidden)
        self.critic_mod = CriticEnsemble(
            n_critics=cfg.n_critics, hidden=cfg.hidden, n_hidden=n_hidden
        )
        self.vf_mod = ValueFunction(hidden=cfg.hidden) if cfg.algorithm == "iql" else None

        with jax.default_device(self.jax_device):
            self.rng, k_actor, k_critic, k_vf = jax.random.split(self.rng, 4)
            dummy_s = jnp.zeros((1, self.state_dim), dtype=jnp.float32)
            dummy_a = jnp.zeros((1, self.action_dim), dtype=jnp.float32)
            actor_params = self.actor_mod.init(k_actor, dummy_s)["params"]
            critic_params = self.critic_mod.init(k_critic, dummy_s, dummy_a)["params"]
            vf_params = (
                self.vf_mod.init(k_vf, dummy_s)["params"]
                if self.vf_mod is not None
                else None
            )

        actor_lr = cfg.cql_policy_lr if cfg.algorithm == "cql" else cfg.actor_lr
        self.actor_tx = optax.adam(actor_lr)
        self.critic_tx = optax.adam(cfg.critic_lr)
        self.vf_tx = optax.adam(cfg.vf_lr) if cfg.algorithm == "iql" else None

        if cfg.algorithm == "iql" and cfg.tau == 0.005:
            cfg.tau = 0.001

        # IQL cosine schedule via optax
        if cfg.algorithm == "iql":
            self.actor_tx = optax.adam(
                optax.cosine_decay_schedule(init_value=actor_lr, decay_steps=cfg.max_timesteps)
            )

        self.state = TrainState(
            actor_params=actor_params,
            actor_target_params=jax.tree_util.tree_map(lambda x: x, actor_params),
            actor_opt_state=self.actor_tx.init(actor_params),
            critic_params=critic_params,
            critic_target_params=jax.tree_util.tree_map(lambda x: x, critic_params),
            critic_opt_state=self.critic_tx.init(critic_params),
            teacher_params=jax.tree_util.tree_map(lambda x: x, actor_params),
            quarantined_params=jax.tree_util.tree_map(lambda x: x, actor_params),
            vf_params=vf_params,
            vf_opt_state=self.vf_tx.init(vf_params) if self.vf_tx is not None else None,
            has_teacher=jnp.asarray(0.0),
            teacher_n=jnp.asarray(0),
            teacher_tau=jnp.asarray(float("nan")),
            has_quarantined=jnp.asarray(0.0),
            quarantined_n=jnp.asarray(0),
            quarantined_tau=jnp.asarray(float("nan")),
            q_scale=jnp.asarray(1.0),
            total_it=jnp.asarray(0),
            actor_updates=jnp.asarray(0),
            last_capo_step=jnp.asarray(-10**9),
            rng=self.rng,
        )
        self.state = jax.device_put(self.state, self.jax_device)

        def actor_apply(params, s):
            return self.actor_mod.apply({"params": params}, s)

        def critic_apply(params, s, a):
            return self.critic_mod.apply({"params": params}, s, a)

        def vf_apply(params, s):
            return self.vf_mod.apply({"params": params}, s)

        # These functions are also called outside the compiled update step by
        # evaluation and certificate code, so cache their executables here.
        self.actor_apply = jax.jit(actor_apply)
        self.critic_apply = jax.jit(critic_apply)
        self.vf_apply = jax.jit(vf_apply) if self.vf_mod is not None else None

        self._gen_critic_count = max(1, cfg.n_critics // 2)
        self._gen_critic_mod = CriticEnsemble(
            n_critics=self._gen_critic_count,
            hidden=cfg.hidden,
            n_hidden=n_hidden,
        )

        def gen_critic_apply(params, s, a):
            return self._gen_critic_mod.apply({"params": params}, s, a)

        self.gen_critic_apply = jax.jit(gen_critic_apply)

        self.refiner = ProximalW2Refiner(
            lr=cfg.refine_lr,
            n_steps=cfg.refine_steps,
            actor_apply=self.actor_apply,
            critic_apply=self.critic_apply,
        )
        self.capo_cfg = CAPOConfig(
            n_max=cfg.n_max,
            beta_uncertainty=cfg.beta_uncertainty,
            shift_penalty_coef=cfg.shift_penalty_coef,
            data_penalty_coef=cfg.data_penalty_coef,
            accept_margin=cfg.accept_margin,
            tau_max=cfg.tau_max,
            tau_min=cfg.tau_min,
            max_action_mse=cfg.max_action_mse,
            normalize_delta_q=cfg.normalize_delta_q,
            target_action_mse=cfg.target_action_mse,
            initial_tau=cfg.initial_tau,
            tau_pilot_initial=cfg.tau_pilot_initial,
            tau_duplicate_log_tolerance=cfg.tau_duplicate_log_tolerance,
        )
        self.tau_controller_state: List[dict] = [
            {
                "previous_selected_tau": None,
                "tau": float(cfg.tau_pilot_initial),
                "action_mse": float(cfg.target_action_mse),
            }
            for _ in range(cfg.n_max)
        ]
        self.last_capo_info: Dict[str, float] = {}
        self.best_score = -1e9
        self.best_base_score = -1e9
        self._host_total_it = 0
        self._host_last_capo_step = -10**9
        self.gate_counts = {
            "replace_count": 0,
            "disable_count": 0,
            "quarantine_count": 0,
            "reactivation_count": 0,
            "stale_count": 0,
        }
        self.last_gate_action = "remain_inactive"
        self.compile_and_first_update_sec = 0.0
        self.first_capo_refresh_wall_sec = None
        self._termination_signal = 0

        stamp = time.strftime("%m%d_%H%M")
        tag = (cfg.run_tag or "").strip().replace(" ", "_")
        mid = f"{tag}_{cfg.algorithm}" if tag else cfg.algorithm
        run_name = f"{stamp}_{mid}_jax_{cfg.env}_s{cfg.seed}"
        if cfg.resume_run_dir:
            self.run_dir = Path(cfg.resume_run_dir).expanduser().resolve()
        elif cfg.run_id:
            sweep_part = cfg.sweep_name or "sweeps"
            self.run_dir = Path(cfg.out_dir) / sweep_part / cfg.run_id
        else:
            self.run_dir = (
                Path(cfg.out_dir) / cfg.algorithm / cfg.env / f"s{cfg.seed}" / run_name
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=2)
        log_mode = "a" if cfg.resume_run_dir else "w"
        self._log_fp = open(self.run_dir / "train.log", log_mode, buffering=1)
        sys.stdout = _Tee(sys.stdout, self._log_fp)
        sys.stderr = _Tee(sys.stderr, self._log_fp)
        raw_env.close()
        self._start_step = 0
        if cfg.resume_run_dir:
            self._load_training_checkpoint()

        # Compile update steps
        self._update_td3bc_jit = jax.jit(self._td3bc_update_fn)
        self._update_iql_jit = jax.jit(self._iql_update_fn)
        self._update_cql_jit = jax.jit(self._cql_update_fn)

        cfg.jit_update_chunk = max(1, int(cfg.jit_update_chunk))
        self._buffer_arrays = (
            self.buffer.states, self.buffer.actions, self.buffer.rewards,
            self.buffer.next_states, self.buffer.dones,
        )
        update_fn = {
            "td3_bc": self._td3bc_update_fn,
            "iql": self._iql_update_fn,
            "cql": self._cql_update_fn,
        }[cfg.algorithm]

        def run_update_chunk(state, rng, buffer_arrays):
            def body(carry, _):
                state_i, rng_i = carry
                rng_i, sample_key, update_key = jax.random.split(rng_i, 3)
                idx = jax.random.randint(
                    sample_key, (cfg.batch_size,), 0, self.buffer.size
                )
                batch = tuple(array[idx] for array in buffer_arrays)
                state_i, logs_i = update_fn(state_i, batch, update_key)
                state_i = state_i.replace(rng=rng_i)
                return (state_i, rng_i), logs_i

            (state, rng), logs = jax.lax.scan(
                body, (state, rng), xs=None, length=cfg.jit_update_chunk
            )
            last_logs = jax.tree_util.tree_map(lambda x: x[-1], logs)
            return state, rng, last_logs

        self._run_update_chunk_jit = jax.jit(run_update_chunk)

    # ------------------------------------------------------------------ helpers
    def _policy(self, params) -> ActorPolicy:
        return ActorPolicy(params, self.actor_apply)

    def _split_critics(self):
        n_critics = self.cfg.n_critics
        if not self.cfg.split_critics_for_certification or n_critics < 2:
            self.refiner.critic_apply = self.critic_apply
            adapter = CriticEnsembleAdapter(
                self.state.critic_params, self.critic_apply, 0, n_critics
            )
            return (adapter,), (adapter,), self.state.critic_params
        mid = self._gen_critic_count
        gen_params = slice_ensemble_params(self.state.critic_params, 0, mid)
        gen_adapter = CriticEnsembleAdapter(
            self.state.critic_params, self.critic_apply, 0, mid
        )
        cert_adapter = CriticEnsembleAdapter(
            self.state.critic_params, self.critic_apply, mid, n_critics
        )
        self.refiner.critic_apply = self.gen_critic_apply
        return (gen_adapter,), (cert_adapter,), gen_params

    # ----------------------------------------------------------- JIT updates
    def _td3bc_update_fn(self, state: TrainState, batch, rng):
        cfg = self.cfg
        states, actions, rewards, next_states, dones = batch
        actor_apply = self.actor_apply
        critic_apply = self.critic_apply

        def critic_loss_fn(critic_params):
            noise = jax.random.normal(rng, actions.shape) * cfg.policy_noise
            noise = jnp.clip(noise, -cfg.noise_clip, cfg.noise_clip)
            next_actions = jnp.clip(
                actor_apply(state.actor_target_params, next_states) + noise,
                -self.max_action,
                self.max_action,
            )
            target_q = q_min(critic_apply(state.critic_target_params, next_states, next_actions))
            target = rewards.squeeze(-1) + (1.0 - dones.squeeze(-1)) * cfg.discount * target_q
            q_pred = critic_apply(critic_params, states, actions)
            return jnp.mean((q_pred - target[None, :]) ** 2)

        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(state.critic_params)
        critic_updates, critic_opt_state = self.critic_tx.update(
            critic_grads, state.critic_opt_state, state.critic_params
        )
        critic_params = optax.apply_updates(state.critic_params, critic_updates)

        do_actor = (state.total_it + 1) % cfg.policy_freq == 0

        def actor_loss_fn(actor_params):
            pi = actor_apply(actor_params, states)
            q_student = critic_apply(critic_params, states, pi)
            q_values = q_mean(q_student)
            q_scale = jax.lax.stop_gradient(
                jnp.abs(q_values).mean() + cfg.actor_q_scale_eps
            )
            if cfg.td3_actor_objective == "td3bc_legacy":
                legacy = td3bc_legacy_actor_components(
                    q_student, pi, actions, alpha=cfg.alpha,
                    eps=cfg.actor_q_scale_eps,
                )
                q_scale = legacy["q_scale"]
                td3bc_scale = legacy["td3bc_scale"]
                q_term = legacy["q_actor_term"]
                data_bc_weighted = legacy["data_bc_loss"]
            else:
                td3bc_scale = 1.0 / q_scale
                q_term = -(q_values / q_scale).mean()
                data_bc_weighted = cfg.lambda_D * jnp.mean((pi - actions) ** 2)
            bc_data = jnp.mean((pi - actions) ** 2)
            a_t = jax.lax.stop_gradient(actor_apply(state.teacher_params, states))
            q_teacher = jax.lax.stop_gradient(
                critic_apply(critic_params, states, a_t)
            )
            teacher_stats = teacher_bc_components(
                pi,
                a_t,
                q_student,
                q_teacher,
                teacher_active=state.has_teacher,
                lambda_t=cfg.lambda_T,
                beta_uncertainty=cfg.beta_uncertainty,
                mode=cfg.teacher_bc_mode,
            )
            teacher_weighted = (
                jnp.asarray(0.0, dtype=bc_data.dtype)
                if cfg.td3_actor_objective == "td3bc_legacy"
                else teacher_stats["teacher_bc_loss_weighted"]
            )
            loss = q_term + data_bc_weighted + teacher_weighted
            stats = {
                "actor_loss": loss,
                "q_actor_term": q_term,
                "data_bc_loss": bc_data,
                "dataset_action_mse": bc_data,
                "student_teacher_action_mse": teacher_stats["teacher_bc_uniform"],
                "q_scale": q_scale,
                "td3bc_scale": td3bc_scale,
                "data_bc_loss_weighted": data_bc_weighted,
                **teacher_stats,
                "teacher_bc_loss_weighted": teacher_weighted,
            }
            return loss, stats

        def actor_branch(carry):
            st = carry
            (actor_loss, astats), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
                st.actor_params
            )
            actor_updates, actor_opt_state = self.actor_tx.update(
                actor_grads, st.actor_opt_state, st.actor_params
            )
            actor_params = optax.apply_updates(st.actor_params, actor_updates)
            actor_target = soft_update(actor_params, st.actor_target_params, cfg.tau)
            critic_target = soft_update(critic_params, st.critic_target_params, cfg.tau)
            st = st.replace(
                actor_params=actor_params,
                actor_target_params=actor_target,
                actor_opt_state=actor_opt_state,
                critic_params=critic_params,
                critic_target_params=critic_target,
                critic_opt_state=critic_opt_state,
                total_it=st.total_it + 1,
                actor_updates=st.actor_updates + 1,
            )
            logs = {
                "critic_loss": critic_loss,
                **astats,
                "bc_teacher": astats["teacher_bc_loss_unweighted"],
                "has_teacher": st.has_teacher,
                "capo_selected_n": st.teacher_n.astype(jnp.float32),
                "did_actor": jnp.asarray(1.0),
            }
            return st, logs

        def critic_only(carry):
            st = carry
            st = st.replace(
                critic_params=critic_params,
                critic_opt_state=critic_opt_state,
                total_it=st.total_it + 1,
            )
            zero = jnp.asarray(0.0)
            logs = {
                "critic_loss": critic_loss,
                "actor_loss": zero,
                "q_actor_term": zero,
                "data_bc_loss": zero,
                "data_bc_loss_weighted": zero,
                "dataset_action_mse": zero,
                "student_teacher_action_mse": zero,
                "q_scale": state.q_scale,
                "td3bc_scale": zero,
                "teacher_bc_uniform": zero,
                "teacher_bc_masked": zero,
                "teacher_bc_loss_unweighted": zero,
                "teacher_bc_loss_weighted": zero,
                "teacher_mask_fraction": zero,
                "teacher_lcb_mean": zero,
                "teacher_lcb_std": zero,
                "bc_teacher": zero,
                "has_teacher": st.has_teacher,
                "capo_selected_n": st.teacher_n.astype(jnp.float32),
                "did_actor": zero,
            }
            return st, logs

        return jax.lax.cond(do_actor, actor_branch, critic_only, state)

    def _iql_update_fn(self, state: TrainState, batch, rng):
        del rng
        cfg = self.cfg
        states, actions, rewards, next_states, dones = batch
        actor_apply = self.actor_apply
        critic_apply = self.critic_apply
        vf_apply = self.vf_apply

        def vf_loss_fn(vf_params):
            target_q = q_min(critic_apply(state.critic_target_params, states, actions))
            v = vf_apply(vf_params, states)
            adv = target_q - v
            return asymmetric_l2_loss(adv, cfg.iql_tau), adv

        (v_loss, adv), vf_grads = jax.value_and_grad(vf_loss_fn, has_aux=True)(state.vf_params)
        vf_updates, vf_opt_state = self.vf_tx.update(vf_grads, state.vf_opt_state, state.vf_params)
        vf_params = optax.apply_updates(state.vf_params, vf_updates)

        def critic_loss_fn(critic_params):
            # Match the PyTorch update order: advantage and TD target both use
            # the value-network snapshot from the beginning of this step.
            next_v = vf_apply(state.vf_params, next_states)
            targets = rewards.squeeze(-1) + (1.0 - dones.squeeze(-1)) * cfg.discount * next_v
            q_pred = critic_apply(critic_params, states, actions)
            return jnp.mean((q_pred - targets[None, :]) ** 2)

        q_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(state.critic_params)
        critic_updates, critic_opt_state = self.critic_tx.update(
            critic_grads, state.critic_opt_state, state.critic_params
        )
        critic_params = optax.apply_updates(state.critic_params, critic_updates)
        critic_target = soft_update(critic_params, state.critic_target_params, cfg.tau)

        def actor_loss_fn(actor_params):
            exp_adv = jnp.clip(jnp.exp(cfg.iql_beta * jax.lax.stop_gradient(adv)), a_max=EXP_ADV_MAX)
            pi = actor_apply(actor_params, states)
            bc = jnp.sum((pi - actions) ** 2, axis=-1)
            a_t = jax.lax.stop_gradient(actor_apply(state.teacher_params, states))
            bc_teacher_raw = jnp.mean((pi - a_t) ** 2)
            teacher_active = state.has_teacher * (1.0 if cfg.lambda_T > 0 else 0.0)
            teacher_term = teacher_active * cfg.lambda_T * bc_teacher_raw
            loss = jnp.mean(exp_adv * bc) + teacher_term
            return loss, bc_teacher_raw * teacher_active

        (actor_loss, teacher_bc), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
            state.actor_params
        )
        actor_updates, actor_opt_state = self.actor_tx.update(
            actor_grads, state.actor_opt_state, state.actor_params
        )
        actor_params = optax.apply_updates(state.actor_params, actor_updates)

        new_state = state.replace(
            actor_params=actor_params,
            actor_opt_state=actor_opt_state,
            critic_params=critic_params,
            critic_target_params=critic_target,
            critic_opt_state=critic_opt_state,
            vf_params=vf_params,
            vf_opt_state=vf_opt_state,
            total_it=state.total_it + 1,
            actor_updates=state.actor_updates + 1,
        )
        logs = {
            "critic_loss": q_loss,
            "value_loss": v_loss,
            "actor_loss": actor_loss,
            "bc_teacher": teacher_bc,
            "has_teacher": new_state.has_teacher,
            "capo_selected_n": new_state.teacher_n.astype(jnp.float32),
            "did_actor": jnp.asarray(1.0),
        }
        return new_state, logs

    def _cql_update_fn(self, state: TrainState, batch, rng):
        cfg = self.cfg
        states, actions, rewards, next_states, dones = batch
        bsz, act_dim = actions.shape
        actor_apply = self.actor_apply
        critic_apply = self.critic_apply
        rng, rng_rand = jax.random.split(rng)

        def critic_loss_fn(critic_params):
            next_actions = actor_apply(state.actor_params, next_states)
            target_q = q_min(critic_apply(state.critic_target_params, next_states, next_actions))
            td_target = rewards.squeeze(-1) + (1.0 - dones.squeeze(-1)) * cfg.discount * target_q
            q_data = critic_apply(critic_params, states, actions)
            td_loss = jnp.mean((q_data - td_target[None, :]) ** 2)

            rand_actions = jax.random.uniform(
                rng_rand, (bsz, cfg.cql_n_actions, act_dim), minval=-1.0, maxval=1.0
            )
            q_rand = critic_apply(critic_params, states, rand_actions)
            pi_actions = jax.lax.stop_gradient(actor_apply(state.actor_params, states))
            q_pi = critic_apply(critic_params, states, pi_actions)
            random_density = math.log(0.5**act_dim)
            cat = jnp.concatenate([q_rand - random_density, q_pi[..., None]], axis=-1)
            cql_ood = jax.nn.logsumexp(cat / cfg.cql_temp, axis=-1) * cfg.cql_temp
            cql_diff = (cql_ood - q_data).mean()
            loss = td_loss + cfg.cql_alpha * cql_diff
            return loss, cql_diff

        (critic_loss, cql_diff), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
            state.critic_params
        )
        critic_updates, critic_opt_state = self.critic_tx.update(
            critic_grads, state.critic_opt_state, state.critic_params
        )
        critic_params = optax.apply_updates(state.critic_params, critic_updates)
        critic_target = soft_update(critic_params, state.critic_target_params, cfg.tau)

        def actor_loss_fn(actor_params):
            pi = actor_apply(actor_params, states)
            q_pi_det = q_mean(critic_apply(critic_params, states, pi))
            bc_mse = jnp.mean((pi - actions) ** 2)
            a_t = jax.lax.stop_gradient(actor_apply(state.teacher_params, states))
            bc_teacher_raw = jnp.mean((pi - a_t) ** 2)
            teacher_active = state.has_teacher * (1.0 if cfg.lambda_T > 0 else 0.0)
            teacher_term = teacher_active * cfg.lambda_T * bc_teacher_raw
            q_bc_loss = -q_pi_det.mean() + cfg.bc_coef * bc_mse + teacher_term
            pure_bc = bc_mse
            use_bc_only = (state.total_it + 1) <= cfg.bc_steps
            loss = jnp.where(use_bc_only, pure_bc, q_bc_loss)
            teacher_bc = jnp.where(use_bc_only, 0.0, bc_teacher_raw * teacher_active)
            return loss, teacher_bc

        (actor_loss, teacher_bc), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
            state.actor_params
        )
        actor_updates, actor_opt_state = self.actor_tx.update(
            actor_grads, state.actor_opt_state, state.actor_params
        )
        actor_params = optax.apply_updates(state.actor_params, actor_updates)

        new_state = state.replace(
            actor_params=actor_params,
            actor_opt_state=actor_opt_state,
            critic_params=critic_params,
            critic_target_params=critic_target,
            critic_opt_state=critic_opt_state,
            total_it=state.total_it + 1,
            actor_updates=state.actor_updates + 1,
            rng=rng,
        )
        logs = {
            "critic_loss": critic_loss,
            "cql_diff": cql_diff,
            "actor_loss": actor_loss,
            "bc_teacher": teacher_bc,
            "has_teacher": new_state.has_teacher,
            "capo_selected_n": new_state.teacher_n.astype(jnp.float32),
            "did_actor": jnp.asarray(1.0),
        }
        return new_state, logs

    # --------------------------------------------------------------- CAPO path
    def _maybe_capo(self, logs: Dict[str, Any], total_it: int):
        cfg = self.cfg
        if not cfg.use_capo:
            return
        if total_it < cfg.capo_start_step:
            return
        first = self._host_last_capo_step < cfg.capo_start_step
        due = first or (total_it - self._host_last_capo_step) >= cfg.capo_period
        if not due:
            return

        self.rng, key = jax.random.split(self.state.rng if self.state.rng is not None else self.rng)
        cert_states, cert_actions, _, _, _ = self.buffer.sample(key, cfg.capo_eval_batch)
        capo_wall_start = time.time()
        info = self._run_capo(cert_states, cert_actions)
        capo_wall = time.time() - capo_wall_start
        if self.first_capo_refresh_wall_sec is None:
            self.first_capo_refresh_wall_sec = capo_wall
        self.last_capo_info = info
        logs.update(info)
        self.state = self.state.replace(last_capo_step=jnp.asarray(total_it), rng=self.rng)
        self._host_last_capo_step = total_it
        acc_c = info.get("capo_accepted_cert", float("nan"))
        stop_c = info.get("capo_stop_cert", float("nan"))
        delta = info.get("paired_delta_d4rl", float("nan"))
        print(
            f"[CAPO] step={total_it} N*={int(info.get('capo_selected_n', 0))} "
            f"accepted={int(info.get('capo_accepted', 0))} "
            f"accepted_cert={acc_c:.5f} stop_cert={stop_c:.5f} "
            f"tau*={info.get('capo_selected_tau', float('nan'))} "
            f"pairedΔJ={delta if delta == delta else float('nan'):.2f} "
            f"teacher={int(float(self.state.has_teacher))}",
            flush=True,
        )

    def _pairwise_lcb_cert(self, current_policy, candidate_policy, states, data_actions, cert_critics, q_scale):
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

    def _run_capo(self, states, data_actions) -> Dict[str, float]:
        gen_critics, cert_critics, gen_params = self._split_critics()
        # Restore full critic apply on refiner for cases without split
        if not self.cfg.split_critics_for_certification:
            self.refiner.critic_apply = self.critic_apply
            gen_params = self.state.critic_params

        batch_scale = estimate_q_scale(
            cert_critics, states, actions=data_actions, eps=self.capo_cfg.q_scale_eps
        )
        ema = float(self.cfg.q_scale_ema)
        q_scale = float(ema * float(self.state.q_scale) + (1.0 - ema) * batch_scale)
        self.state = self.state.replace(q_scale=jnp.asarray(q_scale))

        result = calibrated_adaptive_mpi(
            base_policy=self._policy(self.state.actor_params),
            refiner=self.refiner,
            states=states,
            cfg=self.capo_cfg,
            gen_critics=gen_critics,
            cert_critics=cert_critics,
            data_actions=data_actions,
            q_scale=q_scale,
            tau_controller_state=self.tau_controller_state,
            gen_critic_params=gen_params,
        )

        accepted_recs = [r for r in result.records if r.accepted]
        stop_rec = None
        if result.records and not result.records[-1].accepted:
            stop_rec = result.records[-1]
        info = {
            "capo_selected_n": float(result.selected_n),
            "capo_accepted": float(result.accepted),
            "capo_q_scale": float(self.state.q_scale),
            "capo_ladder": float(result.records[-1].ladder_value) if result.records else 0.0,
            "capo_gated": 0.0,
            "capo_n_records": float(len(result.records)),
        }
        if accepted_recs:
            last_acc = accepted_recs[-1]
            info["capo_accepted_cert"] = float(last_acc.selected_certificate)
            info["capo_selected_tau"] = (
                float(last_acc.selected_tau) if last_acc.selected_tau is not None else float("nan")
            )
            info["capo_sum_accepted_cert"] = float(sum(r.selected_certificate for r in accepted_recs))
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

        ladder_path = self.run_dir / "capo_ladder.jsonl"
        ladder_row = {
            "step": int(self.state.total_it),
            "selected_n": int(result.selected_n),
            "accepted": bool(result.accepted),
            "selected_taus": list(result.selected_tau),
            "accepted_certificates": list(result.certificates),
            "backend": "jax",
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

        self._apply_teacher_replace_gate(result, states, data_actions, cert_critics, info)
        info["has_teacher"] = float(self.state.has_teacher)
        info["teacher_n"] = float(self.state.teacher_n)
        info["capo_q_scale"] = float(self.state.q_scale)
        return info

    def _apply_teacher_replace_gate(self, result, states, data_actions, cert_critics, info):
        return apply_teacher_replace_gate(
            self,
            result,
            states,
            data_actions,
            cert_critics,
            info,
            paired_eval_actors,
        )

    def train_step(self, *, sync_logs: bool = True) -> Dict[str, Any]:
        self.rng, sk, uk = jax.random.split(
            self.state.rng if self.state.rng is not None else self.rng, 3
        )
        batch = self.buffer.sample(sk, self.cfg.batch_size)
        if self.cfg.algorithm == "td3_bc":
            self.state, logs = self._update_td3bc_jit(self.state, batch, uk)
        elif self.cfg.algorithm == "iql":
            self.state, logs = self._update_iql_jit(self.state, batch, uk)
        else:
            self.state, logs = self._update_cql_jit(self.state, batch, uk)
        self.state = self.state.replace(rng=self.rng)

        self._host_total_it += 1
        did_actor = (
            self.cfg.algorithm != "td3_bc"
            or self._host_total_it % self.cfg.policy_freq == 0
        )
        if did_actor:
            self._maybe_capo(logs, self._host_total_it)

        if not sync_logs:
            return logs
        host_logs = jax.device_get(logs)
        return {k: float(v) for k, v in host_logs.items()}

    def train_chunk(self, *, sync_logs: bool = True) -> Dict[str, Any]:
        """Run ``jit_update_chunk`` updates in one compiled XLA dispatch."""
        chunk = self.cfg.jit_update_chunk
        if chunk <= 1:
            return self.train_step(sync_logs=sync_logs)
        rng = self.state.rng if self.state.rng is not None else self.rng
        self.state, self.rng, logs = self._run_update_chunk_jit(
            self.state, rng, self._buffer_arrays
        )
        self._host_total_it += chunk
        did_actor = (
            self.cfg.algorithm != "td3_bc"
            or self._host_total_it % self.cfg.policy_freq == 0
        )
        if did_actor:
            self._maybe_capo(logs, self._host_total_it)
        if not sync_logs:
            return logs
        host_logs = jax.device_get(logs)
        return {k: float(v) for k, v in host_logs.items()}

    def _next_capo_step(self) -> int:
        """Next exact actor-update step at which CAPO may refresh."""
        cfg = self.cfg
        if not cfg.use_capo:
            return cfg.max_timesteps + 1
        if self._host_last_capo_step < cfg.capo_start_step:
            threshold = cfg.capo_start_step
        else:
            threshold = self._host_last_capo_step + cfg.capo_period
        if cfg.algorithm == "td3_bc":
            freq = max(1, int(cfg.policy_freq))
            threshold = ((threshold + freq - 1) // freq) * freq
        return int(threshold)

    def _checkpoint_payload(self, step: int, score: float) -> Dict[str, Any]:
        """Complete resumable state plus stable actor aliases for distillation."""
        return {
            "train_state": jax.device_get(self.state),
            "actor": jax.device_get(self.state.actor_params),
            "teacher": jax.device_get(self.state.teacher_params),
            "quarantined_actor": jax.device_get(self.state.quarantined_params),
            "has_teacher": bool(float(self.state.has_teacher) > 0.5),
            "has_quarantined": bool(float(self.state.has_quarantined) > 0.5),
            "teacher_n": int(self.state.teacher_n),
            "teacher_tau": float(self.state.teacher_tau),
            "quarantined_n": int(self.state.quarantined_n),
            "quarantined_tau": float(self.state.quarantined_tau),
            "tau_controller_state": self.tau_controller_state,
            "gate_counts": dict(self.gate_counts),
            "last_gate_action": self.last_gate_action,
            "last_capo_info": self.last_capo_info,
            "host_total_it": int(self._host_total_it),
            "host_last_capo_step": int(self._host_last_capo_step),
            "best_score": float(self.best_score),
            "best_base_score": float(self.best_base_score),
            "step": int(step),
            "score": float(score),
            "config": asdict(self.cfg),
            "backend": "jax",
            "checkpoint_version": 2,
        }

    def _save_training_checkpoint(self, step: int, score: float) -> Path:
        payload = self._checkpoint_payload(step, score)
        path = self.run_dir / f"checkpoint_{int(step)}.pkl"
        _save_ckpt(path, payload)
        _save_ckpt(self.run_dir / "latest.pkl", payload)
        print(f"[ckpt] resumable step={step} → {path.name}, latest.pkl", flush=True)
        return path

    def _find_resume_checkpoint(self) -> Path:
        latest = self.run_dir / "latest.pkl"
        if latest.is_file():
            return latest
        candidates = list(self.run_dir.glob("checkpoint_*.pkl"))
        if not candidates:
            raise FileNotFoundError(f"no resumable checkpoint under {self.run_dir}")
        return max(candidates, key=lambda p: int(p.stem.split("_")[-1]))

    def _load_training_checkpoint(self) -> None:
        path = self._find_resume_checkpoint()
        with open(path, "rb") as stream:
            payload = pickle.load(stream)
        if "train_state" not in payload:
            raise ValueError(f"checkpoint is actor-only and cannot resume: {path}")
        self.state = jax.device_put(payload["train_state"], self.jax_device)
        self.rng = self.state.rng
        self.tau_controller_state = payload.get(
            "tau_controller_state", self.tau_controller_state
        )
        self.gate_counts.update(payload.get("gate_counts") or {})
        self.last_gate_action = payload.get("last_gate_action", "remain_inactive")
        self.last_capo_info = payload.get("last_capo_info") or {}
        self._host_total_it = int(payload.get("host_total_it", payload["step"]))
        self._host_last_capo_step = int(
            payload.get("host_last_capo_step", int(self.state.last_capo_step))
        )
        self.best_score = float(payload.get("best_score", payload.get("score", -1e9)))
        self.best_base_score = float(
            payload.get("best_base_score", payload.get("score", -1e9))
        )
        self._start_step = int(payload["step"])
        print(f"[resume] loaded {path} at step={self._start_step}", flush=True)

    def _write_heartbeat(self, step: int, status: str = "running") -> None:
        payload = {
            "run_id": self.cfg.run_id,
            "env": self.cfg.env,
            "seed": self.cfg.seed,
            "step": int(step),
            "max_timesteps": int(self.cfg.max_timesteps),
            "status": status,
            "updated_unix": time.time(),
            "teacher_state": (
                "active"
                if float(self.state.has_teacher) > 0.5
                else (
                    "quarantined"
                    if float(self.state.has_quarantined) > 0.5
                    else "disabled"
                )
            ),
        }
        tmp = self.run_dir / "heartbeat.json.tmp"
        with open(tmp, "w") as stream:
            json.dump(payload, stream, indent=2)
        tmp.replace(self.run_dir / "heartbeat.json")

    def _install_signal_handlers(self) -> None:
        if not self.cfg.checkpoint_on_signal:
            return

        def request_checkpoint(signum, _frame):
            self._termination_signal = int(signum)
            print(
                f"[signal] received {signal.Signals(signum).name}; "
                "checkpoint requested at next safe boundary",
                flush=True,
            )

        signal.signal(signal.SIGTERM, request_checkpoint)
        signal.signal(signal.SIGINT, request_checkpoint)

    def _on_ckpt_schedule(self, step: int) -> bool:
        freq = int(self.cfg.save_ckpt_freq or 0)
        return freq > 0 and (
            step % freq == 0 or step == int(self.cfg.max_timesteps)
        )

    def train(self) -> Dict[str, float]:
        cfg = self.cfg
        metrics_path = self.run_dir / "metrics.jsonl"
        t0 = time.time()
        last_eval: Dict[str, float] = {}
        self._install_signal_handlers()

        print(
            f"[CAPO-JAX] algo={cfg.algorithm} env={cfg.env} device={self.jax_device} "
            f"n_max={cfg.n_max} period={cfg.capo_period} "
            f"start={cfg.capo_start_step} λ_D={cfg.lambda_D} λ_T={cfg.lambda_T} "
            f"bc_reduction={cfg.bc_reduction} "
            f"replace_gate={cfg.use_replace_gate} margin={cfg.replace_cert_margin} "
            f"stale_action={cfg.stale_incumbent_action} "
            f"tau_ctrl=pilot_adaptive δ={cfg.target_action_mse} "
            f"tau_pilot0={cfg.tau_pilot_initial}",
            flush=True,
        )
        print(f"[CAPO-JAX] run_dir={self.run_dir}", flush=True)

        t = int(self._start_step)
        self._write_heartbeat(t)
        first_update_pending = t == 0
        while t < cfg.max_timesteps:
            next_eval = min(
                ((t // cfg.eval_freq) + 1) * cfg.eval_freq, cfg.max_timesteps
            )
            next_log = ((t // cfg.log_interval) + 1) * cfg.log_interval
            next_ckpt = (
                ((t // cfg.save_ckpt_freq) + 1) * cfg.save_ckpt_freq
                if cfg.save_ckpt_freq > 0
                else cfg.max_timesteps + 1
            )
            next_heartbeat = (
                ((t // cfg.heartbeat_freq) + 1) * cfg.heartbeat_freq
                if cfg.heartbeat_freq > 0
                else cfg.max_timesteps + 1
            )
            boundary = min(
                next_eval,
                next_log,
                next_ckpt,
                next_heartbeat,
                self._next_capo_step(),
                cfg.max_timesteps,
            )
            steps_to_boundary = max(1, boundary - t)
            use_chunk = (
                cfg.jit_update_chunk > 1
                and steps_to_boundary >= cfg.jit_update_chunk
            )
            t += cfg.jit_update_chunk if use_chunk else 1
            is_eval = t % cfg.eval_freq == 0 or t == cfg.max_timesteps
            is_log = t % cfg.log_interval == 0
            update_wall_start = time.time() if first_update_pending else None
            if use_chunk:
                logs = self.train_chunk(sync_logs=is_log or is_eval)
            else:
                logs = self.train_step(sync_logs=is_log or is_eval)
            if first_update_pending:
                leaves = jax.tree_util.tree_leaves(logs)
                if leaves:
                    jax.block_until_ready(leaves[0])
                self.compile_and_first_update_sec = time.time() - update_wall_start
                first_update_pending = False
                print(
                    f"[compile] update_compile_plus_first_dispatch_sec="
                    f"{self.compile_and_first_update_sec:.3f}",
                    flush=True,
                )
            if is_log:
                elapsed = time.time() - t0
                msg = (
                    f"t={t} critic={logs.get('critic_loss', 0):.4f}"
                    f" actor={logs.get('actor_loss', 0):.4f}"
                    f" teacher={int(logs.get('has_teacher', 0))}"
                    f" N*={int(logs.get('capo_selected_n', logs.get('teacher_n', 0)))}"
                    f" ({elapsed:.1f}s)"
                )
                print(msg, flush=True)

            if is_eval:
                teacher_state = (
                    "active"
                    if float(self.state.has_teacher) > 0.5
                    else (
                        "quarantined"
                        if float(self.state.has_quarantined) > 0.5
                        else "disabled"
                    )
                )
                curve_seeds = [cfg.seed * 100_000 + t + i for i in range(cfg.n_episodes)]
                student_eval = eval_actor(
                    self.eval_env,
                    self.actor_apply,
                    self.state.actor_params,
                    cfg.n_episodes,
                    episode_seeds=curve_seeds,
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
                    "has_teacher": float(self.state.has_teacher),
                    "capo_selected_n": float(
                        int(self.state.teacher_n)
                        if float(self.state.has_teacher) > 0.5
                        else self.last_capo_info.get("capo_selected_n", 0.0)
                    ),
                    "backend": "jax",
                    "teacher_active": float(self.state.has_teacher),
                    "teacher_state": teacher_state,
                    "active_teacher_nstar": (
                        int(self.state.teacher_n) if teacher_state == "active" else None
                    ),
                    "active_teacher_tau": (
                        float(self.state.teacher_tau) if teacher_state == "active" else None
                    ),
                    "lambda_T": float(cfg.lambda_T),
                    "capo_period": int(cfg.capo_period),
                    "replace_cert_margin": float(cfg.replace_cert_margin),
                    "stale_incumbent_action": cfg.stale_incumbent_action,
                    "teacher_bc_mode": cfg.teacher_bc_mode,
                    "gate_action": self.last_gate_action,
                    **self.gate_counts,
                }
                if "d4rl_score" in student_eval:
                    row["student_d4rl_score"] = student_eval["d4rl_score"]
                    row["student_score"] = student_eval["d4rl_score"]
                    row["d4rl_score"] = student_eval["d4rl_score"]
                    row["base_d4rl_score"] = student_eval["d4rl_score"]
                    self.best_base_score = max(self.best_base_score, student_eval["d4rl_score"])

                teacher_eval = None
                if cfg.eval_teacher_actor and float(self.state.has_teacher) > 0.5:
                    teacher_eval = eval_actor(
                        self.eval_env,
                        self.actor_apply,
                        self.state.teacher_params,
                        cfg.n_episodes,
                        episode_seeds=curve_seeds,
                    )
                    row["teacher_return_mean"] = teacher_eval["return_mean"]
                    if "d4rl_score" in teacher_eval:
                        row["teacher_d4rl_score"] = teacher_eval["d4rl_score"]
                        row["active_teacher_score"] = teacher_eval["d4rl_score"]
                        row["curve_delta_d4rl"] = (
                            teacher_eval["d4rl_score"] - student_eval.get("d4rl_score", float("nan"))
                        )

                row.setdefault("active_teacher_score", None)
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(row) + "\n")

                score = row.get("student_d4rl_score", row.get("d4rl_score", row["return_mean"]))
                msg = (
                    f"[eval] step={t} student_d4rl={row.get('student_d4rl_score', float('nan')):.2f}"
                    f" teacher={'on' if float(self.state.has_teacher) > 0.5 else 'off'} "
                    f"N*={int(self.state.teacher_n)}"
                )
                if teacher_eval is not None and "d4rl_score" in teacher_eval:
                    msg += f" teacher_d4rl={teacher_eval['d4rl_score']:.2f}"
                print(msg, flush=True)
                last_eval = row

                if cfg.save_best and score > self.best_score:
                    self.best_score = score
                    _save_ckpt(
                        self.run_dir / "best.pkl",
                        {
                            "actor": self.state.actor_params,
                            "teacher": self.state.teacher_params,
                            "critics": self.state.critic_params,
                            "vf": self.state.vf_params,
                            "has_teacher": bool(float(self.state.has_teacher) > 0.5),
                            "teacher_n": int(self.state.teacher_n),
                            "step": t,
                            "score": score,
                            "config": asdict(cfg),
                            "backend": "jax",
                        },
                    )

            if cfg.heartbeat_freq > 0 and t % cfg.heartbeat_freq == 0:
                self._write_heartbeat(t)

            if self._on_ckpt_schedule(t) and t != cfg.max_timesteps:
                checkpoint_score = float(
                    last_eval.get(
                        "student_d4rl_score",
                        last_eval.get(
                            "d4rl_score", last_eval.get("return_mean", float("nan"))
                        ),
                    )
                    if last_eval
                    else float("nan")
                )
                checkpoint_path = self.run_dir / f"checkpoint_{t}.pkl"
                self._save_training_checkpoint(t, checkpoint_score)

            if self._termination_signal:
                signal_score = float(
                    last_eval.get(
                        "student_d4rl_score",
                        last_eval.get("d4rl_score", last_eval.get("return_mean", float("nan"))),
                    ) if last_eval else float("nan")
                )
                self._save_training_checkpoint(t, signal_score)
                self._write_heartbeat(t, status="terminated")
                signum = int(self._termination_signal)
                print(f"[signal] safe checkpoint complete at step={t}", flush=True)
                raise SystemExit(128 + signum)

        final_score = (
            last_eval.get(
                "student_d4rl_score", last_eval.get("d4rl_score", float("nan"))
            )
            if last_eval
            else float("nan")
        )
        final_payload = self._checkpoint_payload(cfg.max_timesteps, final_score)
        # Final files retain critics for existing downstream consumers.
        final_payload.update(
            {
                "critics": jax.device_get(self.state.critic_params),
                "vf": jax.device_get(self.state.vf_params),
            }
        )
        _save_ckpt(self.run_dir / "final.pkl", final_payload)
        _save_ckpt(self.run_dir / f"checkpoint_{int(cfg.max_timesteps)}.pkl", final_payload)
        _save_ckpt(self.run_dir / "latest.pkl", final_payload)
        print(
            f"[ckpt] saved final weights → final.pkl, checkpoint_{int(cfg.max_timesteps)}.pkl",
            flush=True,
        )
        summary = {
            "algorithm": cfg.algorithm,
            "env": cfg.env,
            "seed": cfg.seed,
            "capo_mode": "teacher",
            "backend": "jax",
            "status": "complete",
            "best_student_score": self.best_score,
            "best_learn_score": self.best_score,
            "best_base_score": self.best_base_score,
            "final_eval": last_eval,
            "run_dir": str(self.run_dir),
            "elapsed_sec": time.time() - t0,
            "compile_and_first_update_sec": self.compile_and_first_update_sec,
            "first_capo_refresh_wall_sec": self.first_capo_refresh_wall_sec,
            "note": (
                "JAX CAPO: student_d4rl_score is θL under teacher guidance after CAPO starts; "
                "use --no_capo for pure baseline. Checkpoints are .pkl (not PyTorch .pt)."
            ),
        }
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        self._write_heartbeat(cfg.max_timesteps, status="complete")
        print(f"[done] {summary}", flush=True)
        return summary
