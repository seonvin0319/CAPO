#!/usr/bin/env python3
"""Run paired environment diagnostics from saved CAPO refresh actors."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

from capo_jax.posthoc_paired_eval import discover_run_dirs, evaluate_bundles  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir")
    parser.add_argument("--results_root")
    parser.add_argument("--manifest")
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--event_filter", default="all_refreshes")
    parser.add_argument("--roles", nargs="+")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed0", type=int, default=10_000)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_dirs = discover_run_dirs(
        run_dir=Path(args.run_dir) if args.run_dir else None,
        results_root=Path(args.results_root) if args.results_root else None,
        manifest=Path(args.manifest) if args.manifest else None,
    )
    rows = evaluate_bundles(
        run_dirs, output_dir=Path(args.output_dir),
        steps=set(args.steps) if args.steps else None,
        event_filter=args.event_filter, roles=args.roles,
        episodes=args.episodes, seed0=args.seed0, resume=args.resume,
    )
    print(f"paired rows: {len(rows)}")


if __name__ == "__main__":
    main()
