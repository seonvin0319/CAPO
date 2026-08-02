#!/usr/bin/env python3
"""Aggregate matrix run summaries into a CSV table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="results/matrix")
    args = p.parse_args()
    root = Path(args.out_dir)
    rows = []
    seen = set()
    for summary in sorted(root.glob("**/summary.json")):
        # Skip archive/smoke helpers and dedupe legacy symlink duplicates.
        parts = set(summary.parts)
        if parts & {"_archived_margin0.01", "_smoke_iql", "_smoke"}:
            continue
        key = summary.resolve()
        if key in seen:
            continue
        seen.add(key)
        with open(summary) as f:
            s = json.load(f)
        rows.append(
            {
                "algorithm": s.get("algorithm"),
                "env": s.get("env"),
                "seed": s.get("seed"),
                "best_score": s.get("best_student_score", s.get("best_score")),
                "final_d4rl": (s.get("final_eval") or {}).get("d4rl_score"),
                "final_return": (s.get("final_eval") or {}).get("return_mean"),
                "elapsed_sec": s.get("elapsed_sec"),
                "run_dir": s.get("run_dir"),
            }
        )
    out = root / "matrix_summary.csv"
    if not rows:
        print(f"no summaries under {root}")
        return
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    for r in rows:
        print(
            f"  {r['algorithm']:7s} {r['env']:28s} "
            f"best={r['best_score']}"
            f" final={r['final_d4rl']}"
        )


if __name__ == "__main__":
    main()
