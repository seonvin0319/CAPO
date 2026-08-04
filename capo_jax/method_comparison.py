"""Compatibility-checked decomposition across baseline, controls, and gates."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REQUIRED_MATCH_KEYS = (
    "normalize", "normalize_reward", "max_timesteps", "env", "seed",
    "eval_freq", "n_episodes",
)


def assert_compatible(capo: Mapping[str, Any], baseline: Mapping[str, Any]) -> None:
    mismatches = [
        f"{key}: {capo.get(key)!r} != {baseline.get(key)!r}"
        for key in REQUIRED_MATCH_KEYS
        if capo.get(key) != baseline.get(key)
    ]
    if baseline.get("td3_actor_objective") != "td3bc_legacy":
        mismatches.append(
            "baseline actor objective must be td3bc_legacy, got "
            f"{baseline.get('td3_actor_objective')!r}"
        )
    if int(baseline.get("n_critics", -1)) != 4:
        mismatches.append(f"baseline n_critics must be 4, got {baseline.get('n_critics')!r}")
    if mismatches:
        raise ValueError("incompatible baseline comparison: " + "; ".join(mismatches))


def load_completed(root: Optional[Path]) -> List[Dict[str, Any]]:
    if root is None or not Path(root).exists():
        return []
    output = []
    for config_path in Path(root).rglob("config.json"):
        summary_path = config_path.parent / "summary.json"
        metrics_path = config_path.parent / "metrics.jsonl"
        if not summary_path.exists() or not metrics_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "complete":
            continue
        metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        config = json.loads(config_path.read_text())
        scores = [float(row.get("student_score", row.get("d4rl_score", float("nan")))) for row in metrics]
        late = [score for score, row in zip(scores, metrics) if 700_000 <= int(row["step"]) <= 1_000_000 and math.isfinite(score)]
        output.append({
            "run_dir": str(config_path.parent), "config": config,
            "final": scores[-1] if scores else float("nan"),
            "late_mean": sum(late) / len(late) if late else float("nan"),
        })
    return output


def decomposition_rows(
    stability: Sequence[Dict[str, Any]], baseline: Sequence[Dict[str, Any]],
    legacy: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    base_map = {(row["config"]["env"], int(row["config"]["seed"])): row for row in baseline}
    legacy_map = {(row["config"]["env"], int(row["config"]["seed"])): row for row in legacy}
    controls = {}
    for row in stability:
        cfg = row["config"]
        if float(cfg.get("lambda_T", -1)) == 0.0:
            key = (
                cfg["env"], int(cfg["seed"]), int(cfg["capo_period"]),
                float(cfg["replace_cert_margin"]), cfg["stale_incumbent_action"],
            )
            controls[key] = row
    output = []
    for row in stability:
        cfg = row["config"]
        base = base_map.get((cfg["env"], int(cfg["seed"])))
        legacy_row = legacy_map.get((cfg["env"], int(cfg["seed"])))
        key = (
            cfg["env"], int(cfg["seed"]), int(cfg["capo_period"]),
            float(cfg["replace_cert_margin"]), cfg["stale_incumbent_action"],
        )
        control = controls.get(key)
        if base:
            assert_compatible(cfg, base["config"])
        output.append({
            "run_id": cfg["run_id"], "environment": cfg["env"], "seed": cfg["seed"],
            "lambda_T": cfg.get("lambda_T"), "capo_period": cfg.get("capo_period"),
            "replace_cert_margin": cfg.get("replace_cert_margin"),
            "stale_incumbent_action": cfg.get("stale_incumbent_action"),
            "capo_minus_jax_td3bc_final": row["final"] - base["final"] if base else float("nan"),
            "capo_minus_jax_td3bc_late": row["late_mean"] - base["late_mean"] if base and math.isfinite(base["late_mean"]) else float("nan"),
            "lambda_t0_minus_jax_td3bc_final": row["final"] - base["final"] if base and float(cfg.get("lambda_T", -1)) == 0 else float("nan"),
            "teacher_guidance_minus_matched_lambda_t0_final": row["final"] - control["final"] if control and float(cfg.get("lambda_T", 0)) > 0 else float("nan"),
            "stability_gate_minus_legacy_v8_final": row["final"] - legacy_row["final"] if legacy_row else float("nan"),
            "baseline_actor_objective": base["config"].get("td3_actor_objective") if base else None,
        })
    return output
