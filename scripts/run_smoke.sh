#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"

PYTHON="${PYTHON:-/home/choi/miniconda3/envs/offrl_backup/bin/python}"

echo "[1/2] tabular demo"
"$PYTHON" scripts/run_tabular.py --seed 0

echo "[2/2] D4RL smoke (hopper-medium-v2, 2k steps)"
"$PYTHON" scripts/run_capo.py --config configs/smoke.yaml "$@"
echo "smoke ok"
