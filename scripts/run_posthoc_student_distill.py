#!/usr/bin/env python3
"""CLI for post-hoc persistent student distillation (pure BC)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capo.posthoc_student_distill import (  # noqa: E402
    PosthocDistillConfig,
    PosthocStudentDistiller,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Post-hoc persistent student distillation (pure BC, no CAPO/Q)"
    )
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--checkpoint_glob", type=str, default="checkpoint_*.pt")
    p.add_argument("--start_step", type=int, default=100_000)
    p.add_argument("--end_step", type=int, default=1_000_000)
    p.add_argument("--checkpoint_interval", type=int, default=50_000)
    p.add_argument("--distill_steps_per_checkpoint", type=int, default=1000)
    p.add_argument("--distill_lr", type=float, default=None)
    p.add_argument("--distill_batch_size", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--eval_seed0", type=int, default=0)
    p.add_argument("--heldout_batch_size", type=int, default=2048)
    p.add_argument("--no_resume", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Two-checkpoint smoke: start_step then start_step+interval only",
    )
    p.add_argument(
        "--smoke_steps",
        type=int,
        nargs=2,
        default=None,
        metavar=("STEP_A", "STEP_B"),
        help="Explicit smoke pair (overrides --smoke arithmetic)",
    )
    return p.parse_args()


def main():
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    os.environ.setdefault("MUJOCO_GL", "egl")
    args = parse_args()

    ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve()
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = str(ckpt_dir / "posthoc_student_distill")

    smoke_steps = None
    if args.smoke_steps is not None:
        smoke_steps = (int(args.smoke_steps[0]), int(args.smoke_steps[1]))
    elif args.smoke:
        smoke_steps = (
            int(args.start_step),
            int(args.start_step) + int(args.checkpoint_interval),
        )

    cfg = PosthocDistillConfig(
        checkpoint_dir=str(ckpt_dir),
        out_dir=out_dir,
        checkpoint_glob=args.checkpoint_glob,
        start_step=int(args.start_step),
        end_step=int(args.end_step),
        checkpoint_interval=int(args.checkpoint_interval),
        distill_steps_per_checkpoint=int(args.distill_steps_per_checkpoint),
        distill_lr=args.distill_lr,
        distill_batch_size=args.distill_batch_size,
        device=args.device,
        seed=int(args.seed),
        n_episodes=int(args.n_episodes),
        eval_seed0=int(args.eval_seed0),
        heldout_batch_size=int(args.heldout_batch_size),
        smoke_steps=smoke_steps,
        resume=not args.no_resume,
    )
    PosthocStudentDistiller(cfg).run()


if __name__ == "__main__":
    main()
