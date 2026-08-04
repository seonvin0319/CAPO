"""Deterministic multi-seed summaries for CAPO stability analysis."""
from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


def _finite(values: Iterable[Any]) -> List[float]:
    output = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            output.append(number)
    return output


def _summary(values: Sequence[float], *, rng_seed: int) -> Dict[str, float]:
    if not values:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "minimum": float("nan"),
            "bootstrap_ci_low": float("nan"),
            "bootstrap_ci_high": float("nan"),
        }
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(rng_seed)
    draws = rng.choice(array, size=(2000, len(array)), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "bootstrap_ci_low": float(np.percentile(draws, 2.5)),
        "bootstrap_ci_high": float(np.percentile(draws, 97.5)),
    }


def multi_seed_rows(run_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return environment-level and cross-environment multi-seed summaries.

    Bootstrap sampling is deterministic. Missing baseline deltas remain missing
    rather than being replaced with absolute scores.
    """
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    by_config: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        groups[(row["config_id"], row["environment"])].append(row)
        by_config[row["config_id"]].append(row)

    output: List[Dict[str, Any]] = []
    for group_index, ((config_id, environment), rows) in enumerate(sorted(groups.items())):
        scores = _finite(row.get("final_score") for row in rows)
        deltas = _finite(row.get("delta_final_vs_baseline") for row in rows)
        late_deltas = _finite(row.get("delta_late_mean_vs_baseline") for row in rows)
        score_stats = _summary(scores, rng_seed=10_000 + group_index)
        delta_stats = _summary(deltas, rng_seed=20_000 + group_index)
        late_stats = _summary(late_deltas, rng_seed=30_000 + group_index)
        worst = min(
            rows,
            key=lambda row: float(row.get("delta_final_vs_baseline", row["final_score"]))
            if math.isfinite(float(row.get("delta_final_vs_baseline", float("nan"))))
            else float(row["final_score"]),
        )
        output.append(
            {
                "scope": "environment",
                "config_id": config_id,
                "environment": environment,
                "seed_count": len({int(row["seed"]) for row in rows}),
                "mean_final_score": score_stats["mean"],
                "std_final_score": score_stats["std"],
                "median_final_score": score_stats["median"],
                "worst_seed": int(worst["seed"]),
                "mean_delta_final": delta_stats["mean"],
                "std_delta_final": delta_stats["std"],
                "median_delta_final": delta_stats["median"],
                "bootstrap_delta_final_ci_low": delta_stats["bootstrap_ci_low"],
                "bootstrap_delta_final_ci_high": delta_stats["bootstrap_ci_high"],
                "mean_delta_late": late_stats["mean"],
                "std_delta_late": late_stats["std"],
                "bootstrap_delta_late_ci_low": late_stats["bootstrap_ci_low"],
                "bootstrap_delta_late_ci_high": late_stats["bootstrap_ci_high"],
                "seed_win_rate_over_baseline": mean(value > 0 for value in deltas)
                if deltas else float("nan"),
                "environment_win_rate_over_baseline": float("nan"),
            }
        )

    for config_index, (config_id, rows) in enumerate(sorted(by_config.items())):
        scores = _finite(row.get("final_score") for row in rows)
        deltas = _finite(row.get("delta_final_vs_baseline") for row in rows)
        late_deltas = _finite(row.get("delta_late_mean_vs_baseline") for row in rows)
        score_stats = _summary(scores, rng_seed=40_000 + config_index)
        delta_stats = _summary(deltas, rng_seed=50_000 + config_index)
        late_stats = _summary(late_deltas, rng_seed=60_000 + config_index)
        environment_means = []
        for environment in sorted({row["environment"] for row in rows}):
            environment_values = _finite(
                row.get("delta_final_vs_baseline")
                for row in rows
                if row["environment"] == environment
            )
            if environment_values:
                environment_means.append(mean(environment_values))
        worst = min(
            rows,
            key=lambda row: float(row.get("delta_final_vs_baseline", row["final_score"]))
            if math.isfinite(float(row.get("delta_final_vs_baseline", float("nan"))))
            else float(row["final_score"]),
        )
        output.append(
            {
                "scope": "all_environments",
                "config_id": config_id,
                "environment": "ALL",
                "seed_count": len({int(row["seed"]) for row in rows}),
                "mean_final_score": score_stats["mean"],
                "std_final_score": score_stats["std"],
                "median_final_score": score_stats["median"],
                "worst_seed": int(worst["seed"]),
                "mean_delta_final": delta_stats["mean"],
                "std_delta_final": delta_stats["std"],
                "median_delta_final": delta_stats["median"],
                "bootstrap_delta_final_ci_low": delta_stats["bootstrap_ci_low"],
                "bootstrap_delta_final_ci_high": delta_stats["bootstrap_ci_high"],
                "mean_delta_late": late_stats["mean"],
                "std_delta_late": late_stats["std"],
                "bootstrap_delta_late_ci_low": late_stats["bootstrap_ci_low"],
                "bootstrap_delta_late_ci_high": late_stats["bootstrap_ci_high"],
                "seed_win_rate_over_baseline": mean(value > 0 for value in deltas)
                if deltas else float("nan"),
                "environment_win_rate_over_baseline": mean(
                    value > 0 for value in environment_means
                ) if environment_means else float("nan"),
            }
        )
    return output
