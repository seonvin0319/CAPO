#!/usr/bin/env python3
"""Create JSON/Markdown JAX–PyTorch numerical parity reports."""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capo_jax.jax_torch_parity import run_parity, write_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jax_checkpoint", required=True)
    parser.add_argument("--batch_npz", required=True, help="Fixed normalized states/actions NPZ")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    try:
        with open(args.jax_checkpoint, "rb") as stream:
            checkpoint = pickle.load(stream)
    except Exception as exc:
        raise SystemExit(f"unsupported checkpoint/parameter conversion: {exc}") from exc
    batch = np.load(args.batch_npz)
    config = dict(checkpoint.get("config") or {})
    config.setdefault("max_action", float(batch.get("max_action", 1.0)))
    report = run_parity(
        actor_params=checkpoint["actor"], critic_params=checkpoint["critics"],
        teacher_params=checkpoint.get("teacher", checkpoint["actor"]),
        states=np.asarray(batch["states"], dtype=np.float32),
        actions=np.asarray(batch["actions"], dtype=np.float32), config=config,
    )
    output = Path(args.output_dir)
    write_report(report, output / "jax_torch_parity.json", output / "jax_torch_parity.md")
    print(output / "jax_torch_parity.json")


if __name__ == "__main__":
    main()
