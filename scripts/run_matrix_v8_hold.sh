#!/usr/bin/env bash
# v8 hold success config × 9 locomotion cells (seed 0, sequential).
# Config: configs/v8_hold.yaml  (λ_D=0.2, λ_T=1.0, period=50k, margin=0.0, …)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"

PYTHON="${PYTHON:-/home/choi/miniconda3/envs/offrl_backup/bin/python}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
OUT_DIR="${OUT_DIR:-results}"
CONFIG="${CONFIG:-configs/v8_hold.yaml}"
VARIANT="v8_hold"
RUN_TAG="v8_hold"

ENVS=(hopper halfcheetah walker2d)
DATASETS=(medium medium-expert replay)

STATUS_FILE="$OUT_DIR/queue_status_v8_hold.tsv"
mkdir -p "$OUT_DIR"
if [[ ! -f "$STATUS_FILE" ]]; then
  echo -e "variant\talgo\tenv_base\tdataset\tstatus\trun_dir\tstarted\tfinished" > "$STATUS_FILE"
fi

is_done() {
  local envb="$1" ds="$2"
  grep -q '^'"${VARIANT}"$'\ttd3_bc\t'"${envb}"$'\t'"${ds}"$'\t'done$'\t' "$STATUS_FILE" 2>/dev/null
}

mark() {
  local envb="$1" ds="$2" status="$3" run_dir="$4" started="$5" finished="${6:-}"
  local tmp
  tmp="$(mktemp)"
  awk -F'\t' -v v="$VARIANT" -v e="$envb" -v d="$ds" '
    NR==1 {print; next}
    !($1==v && $2=="td3_bc" && $3==e && $4==d) {print}
  ' "$STATUS_FILE" > "$tmp"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$VARIANT" "td3_bc" "$envb" "$ds" "$status" "$run_dir" "$started" "$finished" >> "$tmp"
  mv "$tmp" "$STATUS_FILE"
}

latest_run_dir() {
  local env_id="$1" seed="$2"
  local algo="td3_bc"
  local pat="[0-9][0-9][0-9][0-9]_*_${RUN_TAG}_${algo}_${env_id}_s${seed}"
  local base_new="$OUT_DIR/$algo/$env_id/s${seed}"
  local base_old="$OUT_DIR/$env_id/s${seed}"
  local hit=""
  if [[ -d "$base_new" ]]; then
    hit="$(ls -dt "$base_new"/$pat 2>/dev/null | head -1 || true)"
  fi
  if [[ -z "$hit" && -d "$base_old" ]]; then
    hit="$(ls -dt "$base_old"/$pat 2>/dev/null | head -1 || true)"
  fi
  [[ -n "$hit" ]] || return 1
  if [[ -L "$hit" ]]; then
    hit="$(readlink -f "$hit")"
  fi
  echo "$hit"
}

echo "[CAPO v8_hold] 9 jobs → $OUT_DIR (config=$CONFIG seed=$SEED device=$DEVICE)"

for envb in "${ENVS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    if is_done "$envb" "$ds"; then
      echo "[skip] $VARIANT td3_bc $envb $ds (already done)"
      continue
    fi
    case "$ds" in
      medium) env_id="${envb}-medium-v2" ;;
      expert) env_id="${envb}-expert-v2" ;;
      medium-expert|medium_expert) env_id="${envb}-medium-expert-v2" ;;
      replay|medium-replay|medium_replay) env_id="${envb}-medium-replay-v2" ;;
      *) env_id="${envb}-${ds}-v2" ;;
    esac
    started="$(date '+%F %T')"
    mark "$envb" "$ds" "running" "pending" "$started" ""
    echo "[run] $VARIANT td3_bc $envb $ds → $env_id"
    set +e
    "$PYTHON" scripts/run_capo.py \
      --config "$CONFIG" \
      --algorithm td3_bc \
      --env_base "$envb" \
      --dataset "$ds" \
      --seed "$SEED" \
      --device "$DEVICE" \
      --out_dir "$OUT_DIR" \
      --run_tag "$RUN_TAG"
    rc=$?
    set -e
    finished="$(date '+%F %T')"
    run_dir="$(latest_run_dir "$env_id" "$SEED" || echo unknown)"
    if [[ $rc -eq 0 ]]; then
      mark "$envb" "$ds" "done" "$run_dir" "$started" "$finished"
      echo "[ok] $VARIANT $envb $ds → $run_dir"
      if [[ -d "$run_dir" && -f "$run_dir/metrics.jsonl" ]]; then
        "$PYTHON" scripts/plot_training_curve.py "$run_dir" || true
      fi
    else
      mark "$envb" "$ds" "failed" "$run_dir" "$started" "$finished"
      echo "[fail] $VARIANT $envb $ds rc=$rc"
    fi
  done
done

echo "[CAPO v8_hold] finished. status=$STATUS_FILE"
"$PYTHON" scripts/summarize_matrix.py --out_dir "$OUT_DIR" || true
