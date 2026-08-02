#!/usr/bin/env bash
# CAPO vs CQL baseline on locomotion (matched to run_matrix.sh / defaults).
# 2 variants × 3 envs × 3 datasets = 18 runs (seed 0, sequential).
#   capo      → configs/defaults.yaml      (use_capo: true,  --algorithm cql)
#   baseline  → configs/baseline_cql.yaml  (use_capo: false, --algorithm cql)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"

PYTHON="${PYTHON:-/home/svcho/anaconda3/envs/off_rl_gpu/bin/python}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
OUT_DIR="${OUT_DIR:-results}"
ALGO="cql"

ENVS=(hopper halfcheetah walker2d)
DATASETS=(medium expert replay)
# status key | config | run_tag
VARIANTS=(
  "capo|configs/defaults.yaml|capo"
  "baseline|configs/baseline_cql.yaml|baseline"
)

STATUS_FILE="$OUT_DIR/queue_status.tsv"
mkdir -p "$OUT_DIR"
if [[ ! -f "$STATUS_FILE" ]]; then
  echo -e "variant\talgo\tenv_base\tdataset\tstatus\trun_dir\tstarted\tfinished" > "$STATUS_FILE"
fi

is_done() {
  local variant="$1" envb="$2" ds="$3"
  grep -q $'\t'"${variant}"$'\t'"${ALGO}"$'\t'"${envb}"$'\t'"${ds}"$'\t'done$'\t' "$STATUS_FILE" 2>/dev/null \
    || grep -q '^'"${variant}"$'\t'"${ALGO}"$'\t'"${envb}"$'\t'"${ds}"$'\t'done$'\t' "$STATUS_FILE" 2>/dev/null
}

mark() {
  local variant="$1" envb="$2" ds="$3" status="$4" run_dir="$5" started="$6" finished="${7:-}"
  local tmp
  tmp="$(mktemp)"
  awk -F'\t' -v v="$variant" -v a="$ALGO" -v e="$envb" -v d="$ds" '
    NR==1 {print; next}
    !($1==v && $2==a && $3==e && $4==d) {print}
  ' "$STATUS_FILE" > "$tmp"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$variant" "$ALGO" "$envb" "$ds" "$status" "$run_dir" "$started" "$finished" >> "$tmp"
  mv "$tmp" "$STATUS_FILE"
}

latest_run_dir() {
  local env_id="$1" seed="$2" tag="$3"
  local base="$OUT_DIR/$env_id/s${seed}"
  [[ -d "$base" ]] || return 1
  ls -dt "$base"/[0-9][0-9][0-9][0-9]_*_"${tag}"_"${ALGO}"_"${env_id}"_s"${seed}" 2>/dev/null | head -1
}

echo "[CAPO CQL matrix] 18 jobs (capo + baseline cql × 9 envs) → $OUT_DIR (seed=$SEED device=$DEVICE python=$PYTHON)"

for spec in "${VARIANTS[@]}"; do
  IFS='|' read -r variant config run_tag <<<"$spec"
  for envb in "${ENVS[@]}"; do
    for ds in "${DATASETS[@]}"; do
      if is_done "$variant" "$envb" "$ds"; then
        echo "[skip] $variant $ALGO $envb $ds (already done)"
        continue
      fi
      case "$ds" in
        medium) env_id="${envb}-medium-v2" ;;
        expert) env_id="${envb}-expert-v2" ;;
        replay) env_id="${envb}-medium-replay-v2" ;;
      esac
      started="$(date '+%F %T')"
      mark "$variant" "$envb" "$ds" "running" "pending" "$started" ""
      echo "[run] $variant $ALGO $envb $ds → $env_id (config=$config)"
      set +e
      "$PYTHON" scripts/run_capo.py \
        --config "$config" \
        --algorithm "$ALGO" \
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

echo "[CAPO CQL matrix] finished. status=$STATUS_FILE"
"$PYTHON" scripts/summarize_matrix.py --out_dir "$OUT_DIR" || true
