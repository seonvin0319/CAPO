"""Post-hoc paired evaluation for actor-only CAPO refresh bundles."""
from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import jax
import numpy as np

from .networks import Actor
from .refresh_snapshots import actor_params_for_role, load_refresh_actor_bundle

ROLE_COMPARISONS = (
    ("student_before_refresh", "newly_generated_challenger"),
    ("student_before_refresh", "active_incumbent_before_gate"),
    ("active_incumbent_before_gate", "newly_generated_challenger"),
    ("student_before_refresh", "quarantined_incumbent_before_gate"),
    ("student_before_refresh", "active_teacher_after_gate"),
)
EVENT_FILTERS = {
    "all_refreshes", "replace_new", "stale_disable", "stale_quarantine",
    "reactivate_quarantined", "replace_quarantined_with_new",
    "remain_inactive", "collapse_candidate_events",
}


def common_episode_seeds(episodes: int, seed0: int) -> List[int]:
    return [int(seed0) + i for i in range(int(episodes))]


def _eval_actor(env, actor_apply, params, episode_seeds):
    returns = []
    for seed in episode_seeds:
        env.seed(int(seed))
        env.action_space.seed(int(seed))
        state = env.reset()
        done = False
        total = 0.0
        while not done:
            action = np.asarray(actor_apply(params, np.asarray(state, dtype=np.float32)[None]))[0]
            state, reward, done, _ = env.step(action)
            total += float(reward)
        returns.append(total)
    values = np.asarray(returns, dtype=np.float64)
    return {"return_mean": float(values.mean()), "returns": values}


def _score(env, return_mean: float) -> float:
    try:
        return float(env.get_normalized_score(return_mean) * 100.0)
    except Exception:
        return float(return_mean)


def _collapse_steps(run_dir: Path) -> set[int]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    points = [(int(row["step"]), float(row.get("student_score", row.get("d4rl_score", np.nan)))) for row in rows]
    refresh_path = run_dir / "capo_refresh.jsonl"
    refreshes = [json.loads(line) for line in refresh_path.read_text().splitlines() if line.strip()] if refresh_path.exists() else []
    selected = set()
    for event in refreshes:
        step = int(event["refresh_step"])
        before = [value for s, value in points if s <= step and np.isfinite(value)]
        after = [value for s, value in points if step < s <= step + 100_000 and np.isfinite(value)]
        if before and after and before[-1] - min(after) > 20.0:
            selected.add(step)
    return selected


def discover_run_dirs(
    *, run_dir: Optional[Path], results_root: Optional[Path], manifest: Optional[Path]
) -> List[Path]:
    if run_dir:
        return [Path(run_dir)]
    if manifest:
        return [Path(json.loads(line)["output_dir"]) for line in Path(manifest).read_text().splitlines() if line.strip()]
    if results_root:
        return sorted({path.parent.parent for path in Path(results_root).rglob("refresh_actors/step_*")})
    raise ValueError("one of --run_dir, --results_root, or --manifest is required")


def evaluate_bundles(
    run_dirs: Sequence[Path], *, output_dir: Path, steps: Optional[set[int]] = None,
    event_filter: str = "all_refreshes", roles: Optional[Sequence[str]] = None,
    episodes: int = 40, seed0: int = 10_000, resume: bool = False,
) -> List[Dict[str, Any]]:
    if event_filter not in EVENT_FILTERS:
        raise ValueError(f"unsupported event filter: {event_filter}")
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "paired_refresh_eval.jsonl"
    existing: List[Dict[str, Any]] = []
    done = set()
    if resume and jsonl_path.exists():
        existing = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
        done = {(row["run_id"], row["refresh_step"], row["role_a"], row["role_b"]) for row in existing}
    requested = set(roles or [])
    comparisons = [pair for pair in ROLE_COMPARISONS if not requested or pair[0] in requested or pair[1] in requested]
    seeds = common_episode_seeds(episodes, seed0)
    output = list(existing)
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        collapse_steps = _collapse_steps(run_dir) if event_filter == "collapse_candidate_events" else set()
        for path in sorted(run_dir.glob("refresh_actors/step_*/actors.pkl")):
            bundle = load_refresh_actor_bundle(path)
            step = int(bundle["refresh_step"])
            if steps and step not in steps:
                continue
            if event_filter == "collapse_candidate_events":
                if step not in collapse_steps:
                    continue
            elif event_filter != "all_refreshes" and bundle.get("gate_action") != event_filter:
                continue
            meta = bundle["actor_architecture"]
            actor = Actor(
                action_dim=int(meta["action_dim"]), max_action=float(meta["max_action"]),
                hidden=int(meta["hidden"]),
            )
            apply_fn = jax.jit(lambda params, obs: actor.apply({"params": params}, obs))
            from .buffer import NormalizeObsWrapper, make_env
            env = make_env(bundle["environment"], seed=seed0)
            norm = bundle["observation_normalization"]
            if norm["enabled"]:
                env = NormalizeObsWrapper(env, np.asarray(norm["mean"]), np.asarray(norm["std"]))
            for role_a, role_b in comparisons:
                key = (bundle["run_id"], step, role_a, role_b)
                if key in done or bundle["roles"].get(role_a) is None or bundle["roles"].get(role_b) is None:
                    continue
                params_a = actor_params_for_role(bundle, role_a)
                params_b = actor_params_for_role(bundle, role_b)
                eval_a = _eval_actor(env, apply_fn, params_a, seeds)
                eval_b = _eval_actor(env, apply_fn, params_b, seeds)
                differences = np.asarray(eval_b["returns"]) - np.asarray(eval_a["returns"])
                row = {
                    "run_id": bundle["run_id"], "environment": bundle["environment"],
                    "seed": bundle["seed"], "refresh_step": step,
                    "role_a": role_a, "role_b": role_b,
                    "score_a": _score(env, eval_a["return_mean"]),
                    "score_b": _score(env, eval_b["return_mean"]),
                    "paired_difference": _score(env, eval_b["return_mean"]) - _score(env, eval_a["return_mean"]),
                    "paired_win_rate": float(np.mean(differences > 0)),
                    "episode_seeds": seeds,
                    "episode_level_differences": differences.tolist(),
                    "gate_certificates": bundle["gate_certificates"],
                    "gate_action": bundle["gate_action"],
                    "actor_snapshot_path": str(path),
                    "role_a_checksum": bundle["roles"][role_a],
                    "role_b_checksum": bundle["roles"][role_b],
                }
                output.append(row)
                with open(jsonl_path, "a") as stream:
                    stream.write(json.dumps(row) + "\n")
            env.close()
    csv_path = output_dir / "paired_refresh_eval.csv"
    if output:
        scalar_keys = [key for key, value in output[0].items() if not isinstance(value, (list, dict))]
        with open(csv_path, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=scalar_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(output)
    return output
