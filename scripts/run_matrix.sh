#!/usr/bin/env bash
# CAPO defaults × 9 locomotion cells (seed 0, sequential).
#   capo → configs/defaults.yaml (use_capo: true, n_critics=4)
# Baseline and v8_hold are separate launchers / pipeline stages.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"

PYTHON="${PYTHON:-/home/choi/miniconda3/envs/offrl_backup/bin/python}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
OUT_DIR="${OUT_DIR:-results}"

ENVS=(hopper halfcheetah walker2d)
# paper cells: medium / medium-expert / medium-replay (alias: replay)
DATASETS=(medium medium-expert replay)
# status key | config | run_tag
VARIANTS=(
  "capo|configs/defaults.yaml|capo"
)

STATUS_FILE="$OUT_DIR/queue_status.tsv"
mkdir -p "$OUT_DIR"
if [[ ! -f "$STATUS_FILE" ]]; then
  echo -e "variant\talgo\tenv_base\tdataset\tstatus\trun_dir\tstarted\tfinished" > "$STATUS_FILE"
fi

is_done() {
  local variant="$1" envb="$2" ds="$3"
  grep -q $'\t'"${variant}"$'\ttd3_bc\t'"${envb}"$'\t'"${ds}"$'\t'done$'\t' "$STATUS_FILE" 2>/dev/null \
    || grep -q '^'"${variant}"$'\ttd3_bc\t'"${envb}"$'\t'"${ds}"$'\t'done$'\t' "$STATUS_FILE" 2>/dev/null
}

mark() {
  local variant="$1" envb="$2" ds="$3" status="$4" run_dir="$5" started="$6" finished="${7:-}"
  local tmp
  tmp="$(mktemp)"
  awk -F'\t' -v v="$variant" -v e="$envb" -v d="$ds" '
    NR==1 {print; next}
    !($1==v && $2=="td3_bc" && $3==e && $4==d) {print}
  ' "$STATUS_FILE" > "$tmp"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$variant" "td3_bc" "$envb" "$ds" "$status" "$run_dir" "$started" "$finished" >> "$tmp"
  mv "$tmp" "$STATUS_FILE"
}

latest_run_dir() {
  local env_id="$1" seed="$2" tag="$3"
  local algo="td3_bc"
  local pat="[0-9][0-9][0-9][0-9]_*_${tag}_${algo}_${env_id}_s${seed}"
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

echo "[CAPO matrix] 9 jobs (capo td3_bc × 9 envs) → $OUT_DIR (seed=$SEED device=$DEVICE)"

for spec in "${VARIANTS[@]}"; do
  IFS='|' read -r variant config run_tag <<<"$spec"
  for envb in "${ENVS[@]}"; do
    for ds in "${DATASETS[@]}"; do
      if is_done "$variant" "$envb" "$ds"; then
        echo "[skip] $variant td3_bc $envb $ds (already done)"
        continue
      fi
      case "$ds" in
        medium) env_id="${envb}-medium-v2" ;;
        medium-expert|medium_expert) env_id="${envb}-medium-expert-v2" ;;
        expert) env_id="${envb}-expert-v2" ;;
        replay|medium-replay|medium_replay) env_id="${envb}-medium-replay-v2" ;;
        *) echo "[fail] unknown dataset=$ds"; exit 2 ;;
      esac
      started="$(date '+%F %T')"
      mark "$variant" "$envb" "$ds" "running" "pending" "$started" ""
      echo "[run] $variant td3_bc $envb $ds → $env_id (config=$config)"
      set +e
      "$PYTHON" scripts/run_capo.py \
        --config "$config" \
        --algorithm td3_bc \
        --env_base "$envb" \
        --dataset "$ds" \
        --seed "$SEED" \
        --device "$DEVICE" \
        --out_dir "$OUT_DIR" \
        --run_tag "$run_tag"
      rc=$?
      set -e
      finished="$(date '+%F %T')"
      run_dir="$(latest_run_dir "$env_id" "$SEED" "$run_tag" || echo unknown)"
      if [[ $rc -eq 0 ]]; then
        mark "$variant" "$envb" "$ds" "done" "$run_dir" "$started" "$finished"
        echo "[ok] $variant $envb $ds → $run_dir"
      else
        mark "$variant" "$envb" "$ds" "failed" "$run_dir" "$started" "$finished"
        echo "[fail] $variant $envb $ds rc=$rc (see $run_dir/train.log)"
      fi
    done
  done
done

echo "[CAPO matrix] finished. status=$STATUS_FILE"
"$PYTHON" scripts/summarize_matrix.py --out_dir "$OUT_DIR" || true
