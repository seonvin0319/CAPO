"""Deterministic CAPO stability sweep definition and manifest generation."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import yaml

SWEEP_NAME = "capo_stability_seed0"
ENVIRONMENTS = (
    "hopper-medium-v2",
    "hopper-medium-expert-v2",
    "hopper-medium-replay-v2",
    "halfcheetah-medium-v2",
    "halfcheetah-medium-expert-v2",
    "halfcheetah-medium-replay-v2",
    "walker2d-medium-v2",
    "walker2d-medium-expert-v2",
    "walker2d-medium-replay-v2",
)
FACTORS = {
    "lambda_T": (0.0, 0.5, 1.0),
    "capo_period": (50_000, 100_000),
    "replace_cert_margin": (0.0, 0.001),
    "stale_incumbent_action": ("disable", "quarantine", "replace_new"),
}
FIXED_CONFIG: Dict[str, Any] = {
    "algorithm": "td3_bc",
    "device": "cuda",
    "max_timesteps": 1_000_000,
    "eval_freq": 5_000,
    "n_episodes": 10,
    "batch_size": 256,
    "jit_update_chunk": 32,
    "normalize": True,
    "normalize_reward": True,
    "n_critics": 4,
    "discount": 0.99,
    "use_capo": True,
    "n_max": 2,
    "beta_uncertainty": 0.75,
    "shift_penalty_coef": 0.25,
    "data_penalty_coef": 0.25,
    "tau_min": 0.001,
    "tau_max": 0.05,
    "target_action_mse": 0.0025,
    "max_action_mse": 0.2,
    "tau_pilot_initial": 0.01,
    "initial_tau": 0.01,
    "tau_duplicate_log_tolerance": 1e-6,
    "tau_controller": "pilot_adaptive",
    "normalize_delta_q": True,
    "split_critics_for_certification": True,
    "refine_steps": 2,
    "capo_start_step": 100_000,
    "lambda_D": 0.2,
    "bc_reduction": "element_mean",
    "teacher_bc_mode": "uniform",
    "teacher_hold": True,
    "hold_teacher_on_nstar_zero": True,
    "use_replace_gate": True,
    "eval_base_actor": True,
    "eval_teacher_actor": True,
    "paired_eval_episodes": 40,
    "paired_eval_seed0": 10_000,
    "save_ckpt_freq": 50_000,
    "heartbeat_freq": 10_000,
    "run_tag": "capo_stability",
}
ENV_SHORT = {
    "hopper-medium-v2": "hm",
    "hopper-medium-expert-v2": "hmexp",
    "hopper-medium-replay-v2": "hmr",
    "halfcheetah-medium-v2": "cm",
    "halfcheetah-medium-expert-v2": "cmexp",
    "halfcheetah-medium-replay-v2": "cmr",
    "walker2d-medium-v2": "wm",
    "walker2d-medium-expert-v2": "wmexp",
    "walker2d-medium-replay-v2": "wmr",
}


def _float_id(value: float) -> str:
    if value == 0.0:
        return "0"
    if value == 0.5:
        return "0p5"
    if value == 1.0:
        return "1"
    if value == 0.001:
        return "1e3"
    return format(value, ".8g").replace(".", "p").replace("-", "m")


def configuration_id(config: Dict[str, Any]) -> str:
    return (
        f"lt{_float_id(float(config['lambda_T']))}_"
        f"p{int(config['capo_period']) // 1000}k_"
        f"m{_float_id(float(config['replace_cert_margin']))}_"
        f"{config['stale_incumbent_action']}"
    )


def run_id(env: str, seed: int, config: Dict[str, Any]) -> str:
    return f"{ENV_SHORT[env]}_s{seed}_{configuration_id(config)}"


def deterministic_rng_seed(env: str, seed: int, config_id: str) -> int:
    digest = hashlib.sha256(f"{env}|{seed}|{config_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


def sweep_configurations() -> List[Dict[str, Any]]:
    keys = tuple(FACTORS)
    configs = []
    for values in itertools.product(*(FACTORS[key] for key in keys)):
        configs.append(dict(zip(keys, values)))
    assert len(configs) == 36
    return configs


def generate_manifest(
    *,
    seeds: Sequence[int] = (0,),
    environments: Sequence[str] = ENVIRONMENTS,
    sweep_name: str = SWEEP_NAME,
    results_root: str = "results_jax_sweeps",
    selected_configurations: Iterable[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    configs = list(selected_configurations or sweep_configurations())
    rows: List[Dict[str, Any]] = []
    for env in environments:
        if env not in ENV_SHORT:
            raise ValueError(f"unsupported environment: {env}")
        for seed in seeds:
            for factors in configs:
                config_id = configuration_id(factors)
                identifier = run_id(env, int(seed), factors)
                resolved = {
                    **FIXED_CONFIG,
                    **factors,
                    "env": env,
                    "seed": int(seed),
                    "rng_seed": deterministic_rng_seed(env, int(seed), config_id),
                    "run_id": identifier,
                    "sweep_name": sweep_name,
                    "out_dir": results_root,
                }
                rows.append(
                    {
                        "run_id": identifier,
                        "config_id": config_id,
                        "environment": env,
                        "seed": int(seed),
                        "rng_seed": resolved["rng_seed"],
                        "output_dir": str(Path(results_root) / sweep_name / identifier),
                        "factors": dict(factors),
                        "config": resolved,
                    }
                )
    ids = [row["run_id"] for row in rows]
    if len(ids) != len(set(ids)):
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        raise RuntimeError(f"duplicate run ids: {duplicates}")
    return rows


def write_manifest(
    rows: Sequence[Dict[str, Any]], manifest_path: Path, sweep_path: Path
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    sweep_doc = {
        "name": SWEEP_NAME,
        "environments": list(ENVIRONMENTS),
        "seeds": sorted({row["seed"] for row in rows}),
        "factors": {key: list(value) for key, value in FACTORS.items()},
        "fixed": FIXED_CONFIG,
        "num_configurations": len({row["config_id"] for row in rows}),
        "num_runs": len(rows),
    }
    with open(sweep_path, "w") as stream:
        yaml.safe_dump(sweep_doc, stream, sort_keys=False)


def dry_run_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ids = [row["run_id"] for row in rows]
    duplicate_count = len(ids) - len(set(ids))
    summary = {
        "configurations": len({row["config_id"] for row in rows}),
        "environments": len({row["environment"] for row in rows}),
        "seeds": len({row["seed"] for row in rows}),
        "total_runs": len(rows),
        "duplicates": duplicate_count,
        "first_run_ids": ids[:5],
        "last_run_ids": ids[-5:],
        "output_roots": sorted({str(Path(row["output_dir"]).parent) for row in rows}),
    }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", default=SWEEP_NAME)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--results_root", default="results_jax_sweeps")
    parser.add_argument(
        "--manifest", default="manifests/capo_stability_seed0.jsonl"
    )
    parser.add_argument(
        "--sweep_config", default="configs/sweeps/capo_stability_seed0.yaml"
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = generate_manifest(
        seeds=args.seeds, sweep_name=args.sweep, results_root=args.results_root
    )
    write_manifest(rows, Path(args.manifest), Path(args.sweep_config))
    summary = dry_run_summary(rows)
    print(json.dumps(summary, indent=2))
    if args.dry_run and summary["duplicates"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
