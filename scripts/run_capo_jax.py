#!/usr/bin/env python3
"""Train CAPO (JAX) on D4RL with TD3+BC / IQL / CQL bases."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# These must be set before importing D4RL/MuJoCo or JAX.
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_mujoco_bin = Path.home() / ".mujoco" / "mujoco210" / "bin"
if _mujoco_bin.is_dir():
    _ld_paths = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    if str(_mujoco_bin) not in _ld_paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join([str(_mujoco_bin), *_ld_paths])
        # The dynamic loader snapshots LD_LIBRARY_PATH at process startup.
        os.execv(sys.executable, [sys.executable, *sys.argv])

from capo_jax.trainer import CAPOTrainer, TrainConfig  # noqa: E402

ENV_ALIASES = {
    "hopper": "hopper-medium-v2",
    "halfcheetah": "halfcheetah-medium-v2",
    "walker2d": "walker2d-medium-v2",
    "umaze": "antmaze-umaze-v2",
    "antmaze-umaze": "antmaze-umaze-v2",
    "antmaze": "antmaze-umaze-v2",
}

DATASET_ALIASES = {
    "medium": "medium-v2",
    "expert": "expert-v2",
    "replay": "medium-replay-v2",
    "medium-replay": "medium-replay-v2",
    "medium_replay": "medium-replay-v2",
    "medium-expert": "medium-expert-v2",
    "medium_expert": "medium-expert-v2",
}


def parse_args():
    p = argparse.ArgumentParser(description="Run CAPO (JAX) on D4RL")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--algorithm", type=str, default=None, choices=["td3_bc", "iql", "cql", "td3bc"])
    p.add_argument("--env", type=str, default=None, help="Full D4RL id or alias")
    p.add_argument("--env_base", type=str, default=None, choices=["hopper", "halfcheetah", "walker2d"])
    p.add_argument("--dataset", type=str, default=None, help="medium | expert | replay")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--rng_seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max_timesteps", type=int, default=None)
    p.add_argument("--eval_freq", type=int, default=None)
    p.add_argument("--n_episodes", type=int, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--run_tag", type=str, default=None, help="Folder tag, e.g. capo or baseline")
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--sweep_name", type=str, default=None)
    p.add_argument("--resume_run_dir", type=str, default=None)
    p.add_argument("--heartbeat_freq", type=int, default=None)
    p.add_argument("--no_capo", action="store_true")
    p.add_argument("--n_max", type=int, default=None)
    p.add_argument("--capo_period", type=int, default=None)
    p.add_argument("--capo_start_step", type=int, default=None)
    p.add_argument("--lambda_D", type=float, default=None)
    p.add_argument("--lambda_T", type=float, default=None)
    p.add_argument("--tau_pilot_initial", type=float, default=None)
    p.add_argument("--target_action_mse", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument(
        "--jit_update_chunk", type=int, default=None,
        help="Number of updates fused into one XLA dispatch (default: 32)",
    )
    p.add_argument(
        "--save_ckpt_freq", type=int, default=None,
        help="Periodic .pkl actor snapshots for BC distillation; 0 disables",
    )
    p.add_argument("--beta_uncertainty", type=float, default=None)
    p.add_argument("--max_action_mse", type=float, default=None)
    p.add_argument("--replace_cert_margin", type=float, default=None)
    p.add_argument(
        "--stale_incumbent_action",
        type=str,
        default=None,
        choices=["disable", "quarantine", "keep_old", "replace_new", "disable_teacher"],
    )
    p.add_argument(
        "--nstar_zero_action", choices=["legacy_hold", "revalidate_current"], default=None
    )
    p.add_argument("--save_refresh_actors", action="store_true")
    p.add_argument(
        "--td3_actor_objective", choices=["capo_student", "td3bc_legacy"], default=None
    )
    p.add_argument(
        "--teacher_bc_mode",
        choices=["uniform", "statewise_lcb_mask"],
        default=None,
    )
    p.add_argument(
        "--actor_type",
        choices=["deterministic", "gaussian"],
        default=None,
        help="Actor parameterization (gaussian → Wasserstein distance by default)",
    )
    p.add_argument(
        "--distance_metric",
        choices=["amse", "wasserstein", "auto"],
        default=None,
        help="CAPO movement metric; auto = wasserstein iff actor_type=gaussian",
    )
    return p.parse_args()


def resolve_env(args) -> str | None:
    if args.env_base and args.dataset:
        ds = DATASET_ALIASES.get(args.dataset.lower(), args.dataset)
        if not ds.endswith("-v2") and not ds.endswith("_v2"):
            ds = DATASET_ALIASES.get(args.dataset.lower(), f"{args.dataset}-v2")
        return f"{args.env_base}-{ds}"
    if args.env:
        return ENV_ALIASES.get(args.env.lower(), args.env)
    return None


def load_config(args) -> TrainConfig:
    cfg_dict = {}
    if args.config:
        with open(args.config) as f:
            cfg_dict = yaml.safe_load(f) or {}

    known = set(TrainConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in cfg_dict.items() if k in known}
    cfg = TrainConfig(**filtered)

    env = resolve_env(args)
    if env:
        cfg.env = env
    if args.algorithm is not None:
        cfg.algorithm = args.algorithm.lower().replace("-", "_").replace("+", "")
        if cfg.algorithm in ("td3bc", "td3"):
            cfg.algorithm = "td3_bc"
    if args.seed is not None:
        cfg.seed = args.seed
    if args.rng_seed is not None:
        cfg.rng_seed = args.rng_seed
    if args.device is not None:
        cfg.device = args.device
    if args.max_timesteps is not None:
        cfg.max_timesteps = args.max_timesteps
    if args.eval_freq is not None:
        cfg.eval_freq = args.eval_freq
    if args.n_episodes is not None:
        cfg.n_episodes = args.n_episodes
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.run_tag is not None:
        cfg.run_tag = args.run_tag
    if args.run_id is not None:
        cfg.run_id = args.run_id
    if args.sweep_name is not None:
        cfg.sweep_name = args.sweep_name
    if args.resume_run_dir is not None:
        cfg.resume_run_dir = args.resume_run_dir
    if args.heartbeat_freq is not None:
        cfg.heartbeat_freq = max(0, args.heartbeat_freq)
    if args.no_capo:
        cfg.use_capo = False
        if not cfg.run_tag:
            cfg.run_tag = "baseline"
    if args.n_max is not None:
        cfg.n_max = args.n_max
    if args.capo_period is not None:
        cfg.capo_period = args.capo_period
    if args.capo_start_step is not None:
        cfg.capo_start_step = args.capo_start_step
    if args.lambda_D is not None:
        cfg.lambda_D = args.lambda_D
    if args.lambda_T is not None:
        cfg.lambda_T = args.lambda_T
        if float(cfg.lambda_T) <= 0.0:
            cfg.use_capo = False
            cfg.eval_teacher_actor = False
    if args.tau_pilot_initial is not None:
        cfg.tau_pilot_initial = args.tau_pilot_initial
        cfg.initial_tau = args.tau_pilot_initial
    if args.target_action_mse is not None:
        cfg.target_action_mse = args.target_action_mse
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.jit_update_chunk is not None:
        cfg.jit_update_chunk = max(1, args.jit_update_chunk)
    if args.save_ckpt_freq is not None:
        cfg.save_ckpt_freq = max(0, args.save_ckpt_freq)
    if args.beta_uncertainty is not None:
        cfg.beta_uncertainty = args.beta_uncertainty
    if args.max_action_mse is not None:
        cfg.max_action_mse = args.max_action_mse
    if args.replace_cert_margin is not None:
        cfg.replace_cert_margin = args.replace_cert_margin
    if args.stale_incumbent_action is not None:
        cfg.stale_incumbent_action = (
            "disable"
            if args.stale_incumbent_action == "disable_teacher"
            else args.stale_incumbent_action
        )
    if args.nstar_zero_action is not None:
        cfg.nstar_zero_action = args.nstar_zero_action
    if args.save_refresh_actors:
        cfg.save_refresh_actors = True
    if args.td3_actor_objective is not None:
        cfg.td3_actor_objective = args.td3_actor_objective
    if args.teacher_bc_mode is not None:
        cfg.teacher_bc_mode = args.teacher_bc_mode
    if args.actor_type is not None:
        cfg.actor_type = args.actor_type
    if args.distance_metric is not None:
        cfg.distance_metric = args.distance_metric

    if cfg.algorithm == "iql" and "tau" not in cfg_dict:
        cfg.tau = 0.001
    if cfg.algorithm == "cql":
        if "actor_lr" not in cfg_dict and "cql_policy_lr" not in cfg_dict:
            cfg.cql_policy_lr = 3e-5
        if "bc_coef" not in cfg_dict:
            cfg.bc_coef = 0.1

    return cfg


def main():
    args = parse_args()
    cfg = load_config(args)
    trainer = CAPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
