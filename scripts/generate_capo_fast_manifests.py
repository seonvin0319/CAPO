#!/usr/bin/env python3
"""Generate revised broad, legacy-v8, and matched JAX baseline manifests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capo_jax.experiment_manifests import (  # noqa: E402
    BASELINE_NAME, BROAD_NAME, LEGACY_NAME,
    baseline_manifest, broad_manifest, dry_summary, legacy_manifest, write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", default="results_jax_sweeps")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    specs = [
        (BROAD_NAME, broad_manifest(results_root=args.results_root)),
        (BASELINE_NAME, baseline_manifest(results_root=args.results_root)),
        (LEGACY_NAME, legacy_manifest(results_root=args.results_root)),
    ]
    summaries = []
    all_ids = []
    for name, rows in specs:
        write_manifest(
            rows,
            ROOT / "manifests" / f"{name}.jsonl",
            ROOT / "configs" / "sweeps" / f"{name}.yaml",
        )
        summaries.append(dry_summary(rows))
        all_ids.extend(row["run_id"] for row in rows)
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("run ID collision across broad/baseline/legacy manifests")
    print(json.dumps({"manifests": summaries, "cross_manifest_duplicates": 0}, indent=2))


if __name__ == "__main__":
    main()
