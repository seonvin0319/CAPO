#!/usr/bin/env python3
"""CLI for JAX post-hoc persistent student distillation."""
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
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_mujoco_bin = Path.home() / ".mujoco" / "mujoco210" / "bin"
if _mujoco_bin.is_dir():
    _ld_paths = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    if str(_mujoco_bin) not in _ld_paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join([str(_mujoco_bin), *_ld_paths])
        os.execv(sys.executable, [sys.executable, *sys.argv])

from capo_jax.posthoc_student_distill import (  # noqa: E402
    PosthocDistillConfig,
    PosthocStudentDistiller,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="JAX post-hoc persistent student distillation (pure BC)"
    )
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--checkpoint_glob", default="checkpoint_*.pkl")
    parser.add_argument("--start_step", type=int, default=100_000)
    parser.add_argument("--end_step", type=int, default=1_000_000)
    parser.add_argument("--checkpoint_interval", type=int, default=50_000)
    parser.add_argument("--distill_steps_per_checkpoint", type=int, default=1000)
    parser.add_argument("--distill_lr", type=float, default=None)
    parser.add_argument("--distill_batch_size", type=int, default=None)
    parser.add_argument("--jit_update_chunk", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--eval_seed0", type=int, default=0)
    parser.add_argument("--heldout_batch_size", type=int, default=2048)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke_steps", type=int, nargs=2, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    smoke_steps = None
    if args.smoke_steps is not None:
        smoke_steps = tuple(args.smoke_steps)
    elif args.smoke:
        smoke_steps = (args.start_step, args.start_step + args.checkpoint_interval)
    cfg = PosthocDistillConfig(
        checkpoint_dir=str(checkpoint_dir),
        out_dir=args.out_dir or str(checkpoint_dir / "posthoc_student_distill_jax"),
        checkpoint_glob=args.checkpoint_glob,
        start_step=args.start_step,
        end_step=args.end_step,
        checkpoint_interval=args.checkpoint_interval,
        distill_steps_per_checkpoint=args.distill_steps_per_checkpoint,
        distill_lr=args.distill_lr,
        distill_batch_size=args.distill_batch_size,
        jit_update_chunk=args.jit_update_chunk,
        device=args.device,
        seed=args.seed,
        n_episodes=args.n_episodes,
        eval_seed0=args.eval_seed0,
        heldout_batch_size=args.heldout_batch_size,
        smoke_steps=smoke_steps,
        resume=not args.no_resume,
    )
    PosthocStudentDistiller(cfg).run()


if __name__ == "__main__":
    main()
