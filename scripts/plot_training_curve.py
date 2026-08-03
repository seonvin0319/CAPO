#!/usr/bin/env python3
"""Plot CAPO/IQL training curves from a run directory's metrics.jsonl.

Writes:
  <run_dir>/training_curve.png
  <run_dir>/training_curve_losses.png  (if loss keys present)

Usage:
  python scripts/plot_training_curve.py results/iql/.../0803_...
  python scripts/plot_training_curve.py --all-done results
  python scripts/plot_training_curve.py --watch results --poll 30
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCORE_KEYS = ("student_d4rl_score", "d4rl_score", "base_d4rl_score")
TEACHER_KEY = "teacher_d4rl_score"
LOSS_KEYS = ("critic_loss", "actor_loss", "value_loss")


def load_metrics(run_dir: Path) -> List[Dict]:
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty metrics: {path}")
    return rows


def _series(rows: List[Dict], key: str):
    xs, ys = [], []
    for r in rows:
        if key in r and r[key] is not None and "step" in r:
            try:
                xs.append(int(r["step"]))
                ys.append(float(r[key]))
            except (TypeError, ValueError):
                continue
    return xs, ys


def plot_run(run_dir: Path, force: bool = False) -> List[Path]:
    run_dir = Path(run_dir).resolve()
    out_score = run_dir / "training_curve.png"
    out_loss = run_dir / "training_curve_losses.png"
    if out_score.is_file() and not force:
        # Refresh if metrics are newer than the plot.
        metrics = run_dir / "metrics.jsonl"
        if metrics.stat().st_mtime <= out_score.stat().st_mtime:
            return [out_score] + ([out_loss] if out_loss.is_file() else [])

    rows = load_metrics(run_dir)
    cfg = {}
    cj = run_dir / "config.json"
    if cj.is_file():
        cfg = json.loads(cj.read_text())
    title = (
        f"{cfg.get('algorithm', '?')} | {cfg.get('env', run_dir.parent.parent.name)} | "
        f"tag={cfg.get('run_tag', '') or '?'} | seed={cfg.get('seed', '?')}"
    )

    written: List[Path] = []

    # --- D4RL scores ---
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=140)
    plotted = False
    for key, label, style in (
        ("student_d4rl_score", "student D4RL", {"color": "#1f77b4", "lw": 1.8}),
        ("teacher_d4rl_score", "teacher D4RL", {"color": "#d62728", "lw": 1.4, "ls": "--"}),
        ("base_d4rl_score", "base D4RL", {"color": "#2ca02c", "lw": 1.0, "alpha": 0.7}),
    ):
        xs, ys = _series(rows, key)
        if not xs:
            continue
        # Skip base if identical to student (common).
        if key == "base_d4rl_score":
            sx, sy = _series(rows, "student_d4rl_score")
            if sx and len(sx) == len(xs) and all(abs(a - b) < 1e-9 for a, b in zip(sy, ys)):
                continue
        ax.plot(xs, ys, label=label, **style)
        plotted = True
    if not plotted:
        xs, ys = _series(rows, "d4rl_score")
        if xs:
            ax.plot(xs, ys, label="d4rl_score", color="#1f77b4", lw=1.8)
            plotted = True
    if not plotted:
        xs, ys = _series(rows, "return_mean")
        ax.plot(xs, ys, label="return_mean", color="#1f77b4", lw=1.8)

    # Mark CAPO refresh steps if present in capo_refresh.jsonl
    refresh = run_dir / "capo_refresh.jsonl"
    if refresh.is_file():
        for line in refresh.read_text().splitlines():
            if not line.strip():
                continue
            try:
                step = int(json.loads(line).get("refresh_step"))
            except Exception:
                continue
            ax.axvline(step, color="#aaaaaa", lw=0.6, alpha=0.5)

    ax.set_xlabel("gradient step")
    ax.set_ylabel("D4RL score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_score)
    plt.close(fig)
    written.append(out_score)

    # --- losses ---
    fig, ax = plt.subplots(figsize=(9, 4.0), dpi=140)
    any_loss = False
    for key, color in zip(LOSS_KEYS, ("#9467bd", "#ff7f0e", "#8c564b")):
        xs, ys = _series(rows, key)
        if xs:
            ax.plot(xs, ys, label=key, color=color, lw=1.2)
            any_loss = True
    if any_loss:
        ax.set_xlabel("gradient step")
        ax.set_ylabel("loss")
        ax.set_title(title + " — losses")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_loss)
        written.append(out_loss)
    plt.close(fig)

    # Tiny sidecar json for watchers / boards
    meta = {
        "run_dir": str(run_dir),
        "n_points": len(rows),
        "final_step": rows[-1].get("step"),
        "final_student": rows[-1].get("student_d4rl_score", rows[-1].get("d4rl_score")),
        "final_teacher": rows[-1].get("teacher_d4rl_score"),
        "plots": [str(p) for p in written],
    }
    (run_dir / "training_curve_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return written


def iter_done_runs(root: Path):
    for summary in sorted(root.glob("**/summary.json")):
        if any(part.startswith("_") for part in summary.parts):
            continue
        yield summary.parent


def watch(root: Path, poll: float, force: bool = False) -> None:
    seen = set()
    print(f"[plot-watch] root={root} poll={poll}s", flush=True)
    while True:
        for run_dir in iter_done_runs(root):
            key = str(run_dir.resolve())
            summary = run_dir / "summary.json"
            curve = run_dir / "training_curve.png"
            # Plot when summary exists and curve missing/outdated.
            need = force or (not curve.is_file()) or (
                curve.is_file() and summary.stat().st_mtime > curve.stat().st_mtime
            )
            if need:
                try:
                    paths = plot_run(run_dir, force=True)
                    print(f"[plot-watch] wrote {paths[0]}", flush=True)
                    seen.add(key)
                except Exception as e:
                    print(f"[plot-watch] skip {run_dir}: {e}", flush=True)
        time.sleep(poll)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="?", default=None, help="Single run directory")
    p.add_argument("--all-done", type=str, default=None, help="Scan root for **/summary.json")
    p.add_argument("--watch", type=str, default=None, help="Watch root and plot on completion")
    p.add_argument("--poll", type=float, default=30.0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.watch:
        watch(Path(args.watch), args.poll, force=args.force)
        return

    targets: List[Path] = []
    if args.run_dir:
        targets.append(Path(args.run_dir))
    if args.all_done:
        targets.extend(iter_done_runs(Path(args.all_done)))
    if not targets:
        p.error("provide run_dir and/or --all-done ROOT")

    for run_dir in targets:
        try:
            paths = plot_run(run_dir, force=args.force)
            print(f"ok {run_dir.name}: " + ", ".join(str(x.name) for x in paths))
        except Exception as e:
            print(f"fail {run_dir}: {e}")


if __name__ == "__main__":
    main()
