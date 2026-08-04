#!/usr/bin/env bash
# Refresh host/ext_csh.csv from local queue status (or mark launch), then commit+push.
# Usage:
#   bash scripts/sync_host_board_ext_csh.sh              # sync from queue_status_v8_hold.tsv
#   bash scripts/sync_host_board_ext_csh.sh --mark-launch # first cell running, rest queued
#   bash scripts/sync_host_board_ext_csh.sh --push-only   # commit+push current CSV as-is
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BOARD="host/ext_csh.csv"
STATUS_FILE="${STATUS_FILE:-results/queue_status_v8_hold.tsv}"
PYTHON="${PYTHON:-/home/ext_csh/miniconda3/envs/capo/bin/python}"
MODE="sync"
for a in "$@"; do
  case "$a" in
    --mark-launch) MODE="launch" ;;
    --push-only) MODE="push" ;;
    --sync) MODE="sync" ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

if [[ "$MODE" != "push" ]]; then
  MODE="$MODE" STATUS_FILE="$STATUS_FILE" "$PYTHON" - <<'PY'
from __future__ import annotations
import csv
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

board = Path("host/ext_csh.csv")
status_file = Path(os.environ.get("STATUS_FILE", "results/queue_status_v8_hold.tsv"))
mode = os.environ["MODE"]
now = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")

ENVS = ["hopper", "halfcheetah", "walker2d"]
# Paper queue datasets. Leftover expert rows (if any) are preserved via `by`.
DATASETS = ["medium", "medium-expert", "replay"]

with board.open(newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
header = list(rows[0].keys())
by = {(r["variant"], r["algo"], r["env_base"], r["dataset"], r["seed"]): r for r in rows}


def ensure(variant, algo, envb, ds, seed="0"):
    key = (variant, algo, envb, ds, seed)
    if key not in by:
        by[key] = {
            "variant": variant,
            "algo": algo,
            "env_base": envb,
            "dataset": ds,
            "seed": seed,
            "status": "planned",
            "config": "configs/v8_hold.yaml",
            "run_tag": "v8_hold",
            "run_dir": "",
            "started": "",
            "finished": "",
            "eta": "",
            "notes": "",
            "updated_at": now,
        }
    return by[key]


q = {}
if status_file.is_file():
    with status_file.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            parts = line.rstrip("\n").split("\t")
            if i == 0 or len(parts) < 5:
                continue
            variant, algo, envb, ds, status = parts[:5]
            run_dir = parts[5] if len(parts) > 5 else ""
            started = parts[6] if len(parts) > 6 else ""
            finished = parts[7] if len(parts) > 7 else ""
            q[(variant, algo, envb, ds)] = (status, run_dir, started, finished)

if mode == "launch":
    first = True
    for envb in ENVS:
        for ds in DATASETS:
            r = ensure("v8_hold", "td3_bc", envb, ds)
            if first:
                r["status"] = "running"
                r["started"] = r["started"] or now
                r["notes"] = "matrix launch; scripts/run_matrix_v8_hold.sh"
                first = False
            elif r["status"] not in ("done", "failed", "cancelled"):
                r["status"] = "queued"
                r["notes"] = "behind live v8_hold cell"
            r["updated_at"] = now
else:
    for envb in ENVS:
        for ds in DATASETS:
            r = ensure("v8_hold", "td3_bc", envb, ds)
            key = ("v8_hold", "td3_bc", envb, ds)
            if key in q:
                st, run_dir, started, finished = q[key]
                r["status"] = st
                if run_dir and run_dir != "pending":
                    r["run_dir"] = run_dir
                if started:
                    r["started"] = started
                if finished:
                    r["finished"] = finished
                r["updated_at"] = now

ordered = []
for envb in ENVS:
    for ds in DATASETS:
        ordered.append(by[("v8_hold", "td3_bc", envb, ds, "0")])
seen = {("v8_hold", "td3_bc", e, d, "0") for e in ENVS for d in DATASETS}
for k, r in by.items():
    if k not in seen:
        ordered.append(r)

with board.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=header)
    w.writeheader()
    w.writerows(ordered)
print(f"updated {board} mode={mode} rows={len(ordered)}")
PY
fi

export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-ext_csh}"
export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-ext_csh@ext_csh-box}"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"
export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"

git pull --rebase --autostash origin main || true
git add "$BOARD" scripts/sync_host_board_ext_csh.sh 2>/dev/null || git add "$BOARD"
if git diff --cached --quiet; then
  echo "no board changes to commit"
  exit 0
fi
git commit -m "host(ext_csh): refresh v8_hold board (${MODE})"
git push origin HEAD
echo "pushed host/ext_csh.csv"
