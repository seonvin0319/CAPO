#!/usr/bin/env python3
"""Plot Phase-1 UC-critic training curves (student score, uncertainty, cert)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_jax_sweeps" / "capo_antmaze_uncertainty_critic_seed0"
OUTDIR = ROOT / "host" / "sweeps" / "capo_antmaze_uncertainty_critic_seed0" / "plots"

KEYS = [
    ("student_d4rl_score", "student D4RL"),
    ("critic/uncertainty_mean", "critic uncertainty_mean"),
    ("critic/uncertainty_weight_mean", "uncertainty_weight_mean"),
    ("critic/td_loss_unweighted", "TD loss (unweighted)"),
    ("capo_accepted_cert", "accepted_cert"),
    ("replace_count", "replace_count"),
]


def load_series(run_dir: Path):
    metrics = run_dir / "metrics.jsonl"
    if not metrics.exists():
        return None
    series = {k: [] for k, _ in KEYS}
    steps = []
    for line in metrics.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = row.get("step") or row.get("t")
        if step is None:
            continue
        steps.append(int(step))
        for k, _ in KEYS:
            series[k].append(row.get(k))
    return steps, series


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    runs = sorted(
        d
        for d in RESULTS.iterdir()
        if d.is_dir() and d.name.startswith("ucam_") and "QUARANTINE" not in d.name
    )
    for run_dir in runs:
        loaded = load_series(run_dir)
        if not loaded:
            continue
        steps, series = loaded
        fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
        axes = axes.ravel()
        for ax, (key, title) in zip(axes, KEYS):
            ys = series[key]
            xs = [s for s, y in zip(steps, ys) if y is not None]
            yy = [y for y in ys if y is not None]
            ax.plot(xs, yy, lw=1.2)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        fig.suptitle(run_dir.name, fontsize=10)
        fig.tight_layout()
        fig.savefig(OUTDIR / f"{run_dir.name}.png", dpi=120)
        plt.close(fig)
        print("wrote", run_dir.name)
    print("outdir", OUTDIR)


if __name__ == "__main__":
    main()
