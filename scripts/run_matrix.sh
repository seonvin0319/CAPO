#!/usr/bin/env bash
# CaPO matrix: 3 algos × 3 env bases × 3 datasets = 27 runs (seed 0, sequential)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"

PYTHON="${PYTHON:-/home/choi/miniconda3/envs/offrl_backup/bin/python}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
OUT_DIR="${OUT_DIR:-results}"
CONFIG="${CONFIG:-configs/defaults.yaml}"

ENVS=(hopper halfcheetah walker2d)
DATASETS=(medium expert replay)
ALGOS=(td3_bc iql cql)

STATUS_FILE="$OUT_DIR/queue_status.tsv"
mkdir -p "$OUT_DIR"
if [[ ! -f "$STATUS_FILE" ]]; then
  echo -e "algo\tenv_base\tdataset\tstatus\trun_dir\tstarted\tfinished" > "$STATUS_FILE"
fi

is_done() {
  local algo="$1" envb="$2" ds="$3"
  grep -q $'\t'"${algo}"$'\t'"${envb}"$'\t'"${ds}"$'\t'done$'\t' "$STATUS_FILE" 2>/dev/null
}

mark() {
  local algo="$1" envb="$2" ds="$3" status="$4" run_dir="$5" started="$6" finished="${7:-}"
  local tmp
  tmp="$(mktemp)"
  awk -F'\t' -v a="$algo" -v e="$envb" -v d="$ds" '
    NR==1 {print; next}
    !($1==a && $2==e && $3==d) {print}
  ' "$STATUS_FILE" > "$tmp"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$algo" "$envb" "$ds" "$status" "$run_dir" "$started" "$finished" >> "$tmp"
  mv "$tmp" "$STATUS_FILE"
}

latest_run_dir() {
  local env_id="$1"
  local seed="$2"
  local algo="$3"
  local base="$OUT_DIR/$env_id/s${seed}"
  [[ -d "$base" ]] || return 1
  ls -dt "$base"/[0-9][0-9][0-9][0-9]_*_"${algo}"_"${env_id}"_s"${seed}" 2>/dev/null | head -1
}

echo "[CaPO matrix] 27 jobs → $OUT_DIR (seed=$SEED device=$DEVICE config=$CONFIG)"

for algo in "${ALGOS[@]}"; do
  for envb in "${ENVS[@]}"; do
    for ds in "${DATASETS[@]}"; do
      if is_done "$algo" "$envb" "$ds"; then
        echo "[skip] $algo $envb $ds (already done)"
        continue
      fi
      case "$ds" in
        medium) env_id="${envb}-medium-v2" ;;
        expert) env_id="${envb}-expert-v2" ;;
        replay) env_id="${envb}-medium-replay-v2" ;;
      esac
      started="$(date '+%F %T')"
      mark "$algo" "$envb" "$ds" "running" "pending" "$started" ""
      echo "[run] $algo $envb $ds → $env_id"
      set +e
      "$PYTHON" scripts/run_capo.py \
        --config "$CONFIG" \
        --algorithm "$algo" \
        --env_base "$envb" \
        --dataset "$ds" \
        --seed "$SEED" \
        --device "$DEVICE" \
        --out_dir "$OUT_DIR"
      rc=$?
      set -e
      finished="$(date '+%F %T')"
      run_dir="$(latest_run_dir "$env_id" "$SEED" "$algo" || echo unknown)"
      if [[ $rc -eq 0 ]]; then
        mark "$algo" "$envb" "$ds" "done" "$run_dir" "$started" "$finished"
        echo "[ok] $algo $envb $ds → $run_dir"
      else
        mark "$algo" "$envb" "$ds" "failed" "$run_dir" "$started" "$finished"
        echo "[fail] $algo $envb $ds rc=$rc (see $run_dir/train.log)"
      fi
    done
  done
done

echo "[CaPO matrix] finished. status=$STATUS_FILE"
"$PYTHON" scripts/summarize_matrix.py --out_dir "$OUT_DIR" || true
