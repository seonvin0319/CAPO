"""Offline stability, baseline, Pareto, and factor analysis for CAPO sweeps."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .method_comparison import decomposition_rows, load_completed
from .multiseed_analysis import multi_seed_rows
from .stability_sweep import ENVIRONMENTS, generate_manifest


def _finite(values: Iterable[Any]) -> List[float]:
    out = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def _mean_or_nan(values: Iterable[Any]) -> float:
    finite = _finite(values)
    return mean(finite) if finite else float("nan")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _score(row: Dict[str, Any]) -> float:
    for key in ("student_score", "student_d4rl_score", "d4rl_score", "return_mean"):
        if key in row and row[key] is not None:
            return float(row[key])
    return float("nan")


def largest_drop_within(points: Sequence[Tuple[int, float]], window: int) -> float:
    worst = 0.0
    for i, (step_i, value_i) in enumerate(points):
        for step_j, value_j in points[i + 1 :]:
            if step_j - step_i > window:
                break
            worst = max(worst, value_i - value_j)
    return worst


def compute_curve_metrics(eval_rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    points = sorted(
        (int(row["step"]), _score(row))
        for row in eval_rows
        if "step" in row and math.isfinite(_score(row))
    )
    if not points:
        return {
            key: float("nan")
            for key in (
                "final_score", "best_score", "late_mean_700k_1M",
                "late_std_700k_1M", "late_min_700k_1M", "post_100k_peak",
                "max_peak_to_later_drawdown", "largest_consecutive_eval_drop",
                "largest_drop_within_50k", "largest_drop_within_100k",
            )
        }
    scores = [value for _, value in points]
    late = [value for step, value in points if 700_000 <= step <= 1_000_000]
    post = [(step, value) for step, value in points if step >= 100_000]
    peak = -float("inf")
    drawdown = 0.0
    for _, value in post:
        peak = max(peak, value)
        drawdown = max(drawdown, peak - value)
    consecutive = max(
        [points[i][1] - points[i + 1][1] for i in range(len(points) - 1)] or [0.0]
    )
    return {
        "final_score": scores[-1],
        "best_score": max(scores),
        "late_mean_700k_1M": mean(late) if late else float("nan"),
        "late_std_700k_1M": float(np.std(late)) if late else float("nan"),
        "late_min_700k_1M": min(late) if late else float("nan"),
        "post_100k_peak": max([value for _, value in post], default=float("nan")),
        "max_peak_to_later_drawdown": drawdown,
        "largest_consecutive_eval_drop": consecutive,
        "largest_drop_within_50k": largest_drop_within(points, 50_000),
        "largest_drop_within_100k": largest_drop_within(points, 100_000),
    }


def _event_counts(refresh_rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    actions = [row.get("gate_action", row.get("replacement_decision", "")) for row in refresh_rows]
    states = [row.get("next_teacher_state", row.get("teacher_state", "disabled")) for row in refresh_rows]
    total = max(len(states), 1)
    return {
        "number_of_replace_new_events": sum(
            action in ("replace_new", "replace_quarantined_with_new") for action in actions
        ),
        "number_of_stale_events": sum(action.startswith("stale_") for action in actions),
        "number_of_disable_events": sum(
            action == "stale_disable"
            or (
                row.get("previous_teacher_state") == "active"
                and row.get("next_teacher_state") == "disabled"
            )
            for action, row in zip(actions, refresh_rows)
        ),
        "number_of_quarantine_events": actions.count("stale_quarantine"),
        "number_of_reactivation_events": actions.count("reactivate_quarantined"),
        "active_teacher_fraction": states.count("active") / total,
        "quarantined_fraction": states.count("quarantined") / total,
        "disabled_fraction": states.count("disabled") / total,
    }


def replacement_event_diagnostics(
    eval_rows: Sequence[Dict[str, Any]], refresh_rows: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    points = sorted((int(row["step"]), _score(row)) for row in eval_rows)
    out = []
    for refresh in refresh_rows:
        action = refresh.get("gate_action", refresh.get("replacement_decision"))
        if action not in ("replace_new", "replace_quarantined_with_new"):
            continue
        step = int(refresh["refresh_step"])
        before = [value for point_step, value in points if point_step <= step]
        base = before[-1] if before else float("nan")
        row = {"refresh_step": step, "gate_action": action}
        for label, target in (("next_eval", step + 1), ("plus_50k", step + 50_000), ("plus_100k", step + 100_000)):
            candidates = [(s, v) for s, v in points if s >= target]
            value = candidates[0][1] if candidates else float("nan")
            row[f"delta_{label}"] = value - base if math.isfinite(base) and math.isfinite(value) else float("nan")
        out.append(row)
    return out


def load_baselines(
    baseline_root: Optional[Path], baseline_csv: Optional[Path]
) -> Dict[Tuple[str, int], Dict[str, float]]:
    baselines: Dict[Tuple[str, int], Dict[str, float]] = {}
    if baseline_csv:
        with open(baseline_csv, newline="") as stream:
            for row in csv.DictReader(stream):
                env = row.get("environment", row.get("env"))
                seed = int(row.get("seed", 0))
                baselines[(env, seed)] = {
                    "final": float(row["final_score"]),
                    "late_mean": float(row["late_mean"])
                    if row.get("late_mean") not in (None, "")
                    else float("nan"),
                }
    if baseline_root:
        for path in baseline_root.rglob("metrics.jsonl"):
            rows = read_jsonl(path)
            if not rows:
                continue
            config_path = path.parent / "config.json"
            config = json.loads(config_path.read_text()) if config_path.exists() else {}
            # A baseline directory may also contain CAPO runs. Only pure
            # no-CAPO runs are valid references when that flag is available.
            if config.get("use_capo") is True:
                continue
            env = config.get("env")
            seed = int(config.get("seed", 0))
            if not env:
                continue
            metrics = compute_curve_metrics(rows)
            baselines[(env, seed)] = {
                "final": metrics["final_score"],
                "late_mean": metrics["late_mean_700k_1M"],
            }
    return baselines


def _config_id_from_run_id(identifier: str) -> str:
    parts = identifier.split("_")
    for index, part in enumerate(parts):
        if part.startswith("s") and part[1:].isdigit():
            return "_".join(parts[index + 1:])
    return identifier


def analyze_run(run_dir: Path, baselines: Dict[Tuple[str, int], Dict[str, float]]) -> Dict[str, Any]:
    config = json.loads((run_dir / "config.json").read_text())
    eval_rows = read_jsonl(run_dir / "metrics.jsonl")
    refresh_rows = read_jsonl(run_dir / "capo_refresh.jsonl")
    metrics: Dict[str, Any] = {
        "run_id": config.get("run_id", run_dir.name),
        "config_id": _config_id_from_run_id(config.get("run_id", run_dir.name)),
        "environment": config["env"],
        "seed": int(config["seed"]),
        "lambda_T": float(config["lambda_T"]),
        "capo_period": int(config["capo_period"]),
        "replace_cert_margin": float(config["replace_cert_margin"]),
        "stale_incumbent_action": config["stale_incumbent_action"],
        **compute_curve_metrics(eval_rows),
        **_event_counts(refresh_rows),
    }
    baseline = baselines.get((metrics["environment"], metrics["seed"]))
    metrics["delta_final_vs_baseline"] = (
        metrics["final_score"] - baseline["final"] if baseline else float("nan")
    )
    metrics["delta_late_mean_vs_baseline"] = (
        metrics["late_mean_700k_1M"] - baseline["late_mean"]
        if baseline and math.isfinite(baseline["late_mean"])
        else float("nan")
    )
    metrics["replacement_event_diagnostics"] = replacement_event_diagnostics(
        eval_rows, refresh_rows
    )
    return metrics


def aggregate_configurations(run_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        groups[row["config_id"]].append(row)
    output = []
    for config_id, rows in groups.items():
        first = rows[0]
        values = lambda key: _finite(row.get(key) for row in rows)
        delta_final = values("delta_final_vs_baseline")
        delta_late = values("delta_late_mean_vs_baseline")
        drawdowns = values("max_peak_to_later_drawdown")
        output.append(
            {
                "config_id": config_id,
                "lambda_T": first["lambda_T"],
                "capo_period": first["capo_period"],
                "replace_cert_margin": first["replace_cert_margin"],
                "stale_incumbent_action": first["stale_incumbent_action"],
                "environment_count": len({row["environment"] for row in rows}),
                "mean_final_score": _mean_or_nan(values("final_score")),
                "median_final_score": median(values("final_score")) if values("final_score") else float("nan"),
                "mean_delta_final": mean(delta_final) if delta_final else float("nan"),
                "median_delta_final": median(delta_final) if delta_final else float("nan"),
                "minimum_delta_final": min(delta_final) if delta_final else float("nan"),
                "p10_delta_final": float(np.percentile(delta_final, 10)) if delta_final else float("nan"),
                "mean_late_mean": _mean_or_nan(values("late_mean_700k_1M")),
                "mean_delta_late_mean": mean(delta_late) if delta_late else float("nan"),
                "minimum_delta_late_mean": min(delta_late) if delta_late else float("nan"),
                "mean_maximum_drawdown": _mean_or_nan(drawdowns),
                "worst_maximum_drawdown": max(drawdowns) if drawdowns else float("nan"),
                "environments_delta_final_lt_0": sum(value < 0 for value in delta_final),
                "environments_delta_final_lt_m5": sum(value < -5 for value in delta_final),
                "environments_delta_final_lt_m10": sum(value < -10 for value in delta_final),
                "environments_drawdown_gt_20": sum(value > 20 for value in drawdowns),
                "environments_drawdown_gt_30": sum(value > 30 for value in drawdowns),
            }
        )
    return output


def pareto_frontier(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    objectives = (
        ("mean_delta_late_mean", 1),
        ("minimum_delta_late_mean", 1),
        ("worst_maximum_drawdown", -1),
        ("environments_delta_final_lt_m10", -1),
    )
    finite_rows = [row for row in rows if all(math.isfinite(float(row[key])) for key, _ in objectives)]
    frontier = []
    for candidate in finite_rows:
        dominated = False
        for other in finite_rows:
            if other is candidate:
                continue
            weak = all(direction * other[key] >= direction * candidate[key] for key, direction in objectives)
            strict = any(direction * other[key] > direction * candidate[key] for key, direction in objectives)
            if weak and strict:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier


def rankings(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    safe = sorted(
        rows,
        key=lambda r: (
            r["environments_delta_final_lt_m10"],
            r["environments_drawdown_gt_30"],
            -r["minimum_delta_late_mean"],
            -r["mean_delta_late_mean"],
            -r["mean_delta_final"],
        ),
    )
    return {
        "safety_first": safe,
        "gain_first": sorted(rows, key=lambda r: (-r["mean_delta_late_mean"], -r["mean_delta_final"])),
        "final_performance": sorted(rows, key=lambda r: -r["mean_final_score"]),
        "late_stability": sorted(rows, key=lambda r: (r["worst_maximum_drawdown"], -r["minimum_delta_late_mean"])),
    }


def grouped_factor_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    specs = [
        ("lambda_T",), ("capo_period",), ("replace_cert_margin",),
        ("stale_incumbent_action",),
        ("lambda_T", "stale_incumbent_action"),
        ("lambda_T", "replace_cert_margin"),
        ("capo_period", "stale_incumbent_action"),
        ("capo_period", "replace_cert_margin"),
    ]
    output = []
    for keys in specs:
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[key] for key in keys)].append(row)
        for values, group in groups.items():
            finals = _finite(row["delta_final_vs_baseline"] for row in group)
            late = _finite(row["delta_late_mean_vs_baseline"] for row in group)
            drawdowns = _finite(row["max_peak_to_later_drawdown"] for row in group)
            output.append(
                {
                    "group": "_x_".join(keys),
                    "levels": json.dumps(dict(zip(keys, values)), sort_keys=True),
                    "n": len(group),
                    "mean_delta_final": mean(finals) if finals else float("nan"),
                    "median_delta_final": median(finals) if finals else float("nan"),
                    "mean_delta_late": mean(late) if late else float("nan"),
                    "median_delta_late": median(late) if late else float("nan"),
                    "mean_drawdown": mean(drawdowns) if drawdowns else float("nan"),
                }
            )
    return output


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys and key != "replacement_event_diagnostics":
                keys.append(key)
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _selected_factors(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "lambda_T": row["lambda_T"],
        "capo_period": row["capo_period"],
        "replace_cert_margin": row["replace_cert_margin"],
        "stale_incumbent_action": row["stale_incumbent_action"],
    }


def create_plots(aggregate: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    x = [row["worst_maximum_drawdown"] for row in aggregate]
    y = [row["mean_delta_late_mean"] for row in aggregate]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y)
    for row, x_value, y_value in zip(aggregate, x, y):
        ax.annotate(row["config_id"], (x_value, y_value), fontsize=6)
    ax.set_xlabel("Worst maximum drawdown")
    ax.set_ylabel("Mean late-mean delta vs baseline")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "pareto_stability_scatter.png", dpi=180)
    plt.close(fig)


def analyze(
    results_root: Path,
    output_dir: Path,
    baseline_root: Optional[Path] = None,
    baseline_csv: Optional[Path] = None,
    legacy_root: Optional[Path] = None,
) -> Dict[str, Any]:
    baselines = load_baselines(baseline_root, baseline_csv)
    run_rows = []
    for config_path in results_root.rglob("config.json"):
        run_dir = config_path.parent
        if not (run_dir / "summary.json").exists():
            continue
        config = json.loads(config_path.read_text())
        if config.get("sweep_name") not in (
            "capo_stability_seed0_fast", "capo_stability_stage2"
        ):
            continue
        run_rows.append(analyze_run(run_dir, baselines))
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_configurations(run_rows) if run_rows else []
    frontier = pareto_frontier(aggregate) if aggregate else []
    ranked = rankings(aggregate) if aggregate and baselines else {}
    factors = grouped_factor_rows(run_rows)
    multi_seed = multi_seed_rows(run_rows)
    completed = load_completed(results_root)
    stability_completed = [
        row for row in completed
        if row["config"].get("sweep_name") in (
            "capo_stability_seed0_fast", "capo_stability_stage2"
        )
    ]
    baseline_completed = load_completed(baseline_root)
    legacy_completed = load_completed(legacy_root)
    method_rows = decomposition_rows(
        stability_completed, baseline_completed, legacy_completed
    ) if stability_completed else []
    write_csv(output_dir / "per_run_metrics.csv", run_rows)
    write_csv(output_dir / "configuration_aggregate.csv", aggregate)
    write_csv(output_dir / "pareto_frontier.csv", frontier)
    write_csv(output_dir / "factor_effects_and_interactions.csv", factors)
    write_csv(output_dir / "multi_seed_summary.csv", multi_seed)
    write_csv(output_dir / "method_decomposition.csv", method_rows)
    for name, rows in ranked.items():
        write_csv(output_dir / f"ranking_{name}.csv", rows)

    selected: List[Dict[str, Any]] = []
    if ranked:
        candidate_rows = [*frontier, *ranked["safety_first"][:2], *ranked["gain_first"][:2]]
        seen = set()
        for row in candidate_rows:
            if row["config_id"] not in seen and len(selected) < 6:
                seen.add(row["config_id"])
                selected.append(_selected_factors(row))
        stage2 = generate_manifest(
            seeds=(1, 2, 3, 4),
            environments=ENVIRONMENTS,
            sweep_name="capo_stability_stage2",
            results_root=str(results_root),
            selected_configurations=selected,
        )
        with open(output_dir / "capo_stability_stage2.jsonl", "w") as stream:
            for row in stage2:
                stream.write(json.dumps(row, sort_keys=True) + "\n")

        acceptable = any(
            row["environments_delta_final_lt_m10"] == 0
            and row["environments_drawdown_gt_30"] == 0
            and row["mean_delta_late_mean"] > 0
            for row in aggregate
        )
        if not acceptable:
            best = ranked["safety_first"][0]
            mask_configs = [
                {
                    **_selected_factors(best),
                    "lambda_T": lambda_t,
                    "teacher_bc_mode": "statewise_lcb_mask",
                }
                for lambda_t in (0.5, 1.0)
            ]
            mask_rows = generate_manifest(
                seeds=(0,),
                environments=ENVIRONMENTS,
                sweep_name="capo_stability_statewise_followup",
                results_root=str(results_root),
                selected_configurations=mask_configs,
            )
            with open(output_dir / "statewise_mask_followup.jsonl", "w") as stream:
                for row in mask_rows:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")

    create_plots(aggregate, output_dir)
    summary = {
        "completed_runs": len(run_rows),
        "configurations": len(aggregate),
        "baseline_entries": len(baselines),
        "multi_seed_groups": len(multi_seed),
        "method_comparison_rows": len(method_rows),
        "pareto_configurations": [row["config_id"] for row in frontier],
        "stage2_selected_configurations": selected,
        "delta_late_mean_available": any(
            math.isfinite(row.get("delta_late_mean_vs_baseline", float("nan")))
            for row in run_rows
        ),
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--baseline_root", default=None)
    parser.add_argument("--baseline_csv", default=None)
    parser.add_argument("--legacy_root", default=None)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = analyze(
        Path(args.results_root),
        Path(args.output_dir),
        Path(args.baseline_root) if args.baseline_root else None,
        Path(args.baseline_csv) if args.baseline_csv else None,
        Path(args.legacy_root) if args.legacy_root else None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
