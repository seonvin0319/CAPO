"""Revised broad, legacy-v8, and matched JAX baseline manifests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import yaml

from .stability_sweep import (
    ENVIRONMENTS,
    ENV_SHORT,
    FIXED_CONFIG,
    configuration_id,
    deterministic_rng_seed,
    sweep_configurations,
)

RESULTS_ROOT = "results_jax_sweeps"
BROAD_NAME = "capo_stability_seed0_fast"
LEGACY_NAME = "capo_v8_legacy_seed0"
BASELINE_NAME = "td3bc_4critic_jax_seed0"


def _row(
    *, env: str, seed: int, sweep_name: str, run_id: str,
    config_id: str, config: Dict[str, Any], factors: Dict[str, Any],
    results_root: str,
) -> Dict[str, Any]:
    resolved = {
        **config,
        "env": env,
        "seed": int(seed),
        "rng_seed": deterministic_rng_seed(env, int(seed), f"{sweep_name}|{config_id}"),
        "run_id": run_id,
        "sweep_name": sweep_name,
        "out_dir": results_root,
    }
    return {
        "run_id": run_id,
        "config_id": config_id,
        "environment": env,
        "seed": int(seed),
        "rng_seed": resolved["rng_seed"],
        "output_dir": str(Path(results_root) / sweep_name / run_id),
        "factors": factors,
        "config": resolved,
    }


def broad_manifest(
    *, seeds: Sequence[int] = (0,), results_root: str = RESULTS_ROOT
) -> List[Dict[str, Any]]:
    fixed = {
        **FIXED_CONFIG,
        "paired_eval_episodes": 0,
        "save_refresh_actors": True,
        "nstar_zero_action": "revalidate_current",
        "td3_actor_objective": "capo_student",
    }
    rows = []
    for env in ENVIRONMENTS:
        for seed in seeds:
            for factors in sweep_configurations():
                config_id = configuration_id(factors)
                identifier = f"fast_{ENV_SHORT[env]}_s{seed}_{config_id}"
                rows.append(_row(
                    env=env, seed=int(seed), sweep_name=BROAD_NAME,
                    run_id=identifier, config_id=config_id,
                    config={**fixed, **factors}, factors=dict(factors),
                    results_root=results_root,
                ))
    _validate(rows, expected=36 * len(ENVIRONMENTS) * len(seeds))
    return rows


def legacy_manifest(*, seed: int = 0, results_root: str = RESULTS_ROOT) -> List[Dict[str, Any]]:
    factors = {
        "lambda_T": 1.0,
        "capo_period": 50_000,
        "replace_cert_margin": 0.0,
        "stale_incumbent_action": "replace_new",
    }
    config = {
        **FIXED_CONFIG,
        **factors,
        "lambda_D": 0.2,
        "nstar_zero_action": "legacy_hold",
        "teacher_hold": True,
        "hold_teacher_on_nstar_zero": True,
        "teacher_bc_mode": "uniform",
        "paired_eval_episodes": 0,
        "save_refresh_actors": True,
        "td3_actor_objective": "capo_student",
    }
    rows = [
        _row(
            env=env, seed=seed, sweep_name=LEGACY_NAME,
            run_id=f"v8legacy_{ENV_SHORT[env]}_s{seed}",
            config_id="v8legacy", config=config, factors=factors,
            results_root=results_root,
        )
        for env in ENVIRONMENTS
    ]
    _validate(rows, expected=9)
    return rows


def baseline_manifest(*, seed: int = 0, results_root: str = RESULTS_ROOT) -> List[Dict[str, Any]]:
    config = {
        **FIXED_CONFIG,
        "use_capo": False,
        "n_critics": 4,
        "normalize": True,
        "normalize_reward": True,
        "max_timesteps": 1_000_000,
        "eval_freq": 5_000,
        "n_episodes": 10,
        "alpha": 2.5,
        "td3_actor_objective": "td3bc_legacy",
        "paired_eval_episodes": 0,
        "save_refresh_actors": False,
        "eval_teacher_actor": False,
        "nstar_zero_action": "revalidate_current",
        "run_tag": "td3bc_4critic_jax_baseline",
    }
    rows = [
        _row(
            env=env, seed=seed, sweep_name=BASELINE_NAME,
            run_id=f"td3bc4j_{ENV_SHORT[env]}_s{seed}",
            config_id="td3bc4j", config=config,
            factors={"method": "td3bc_4critic_jax"}, results_root=results_root,
        )
        for env in ENVIRONMENTS
    ]
    _validate(rows, expected=9)
    return rows


def _validate(rows: Sequence[Dict[str, Any]], *, expected: int) -> None:
    ids = [row["run_id"] for row in rows]
    if len(rows) != expected or len(ids) != len(set(ids)):
        raise ValueError(f"invalid manifest: rows={len(rows)} unique={len(set(ids))} expected={expected}")


def write_manifest(rows: Sequence[Dict[str, Any]], manifest: Path, config: Path) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    config.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    config.write_text(yaml.safe_dump({
        "name": rows[0]["config"]["sweep_name"],
        "environments": list(ENVIRONMENTS),
        "seeds": sorted({row["seed"] for row in rows}),
        "num_configurations": len({row["config_id"] for row in rows}),
        "num_runs": len(rows),
        "resolved_example": rows[0]["config"],
    }, sort_keys=False))


def dry_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ids = [row["run_id"] for row in rows]
    return {
        "sweep_name": rows[0]["config"]["sweep_name"],
        "configurations": len({row["config_id"] for row in rows}),
        "environments": len({row["environment"] for row in rows}),
        "seeds": len({row["seed"] for row in rows}),
        "total_runs": len(rows),
        "duplicates": len(ids) - len(set(ids)),
        "first_run_ids": ids[:3],
        "last_run_ids": ids[-3:],
        "output_root": str(Path(rows[0]["output_dir"]).parent),
    }
