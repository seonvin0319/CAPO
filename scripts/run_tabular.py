#!/usr/bin/env python3
"""Run tabular CAPO validation demo."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capo.tabular import run_tabular_demo  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run_tabular_demo(seed=args.seed)


if __name__ == "__main__":
    main()
