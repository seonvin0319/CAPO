#!/usr/bin/env bash
# v8 hold × 9 locomotion cells (seed 0), packed across GPUs.
# Default: 3 jobs/GPU × GPUs 0,1 → max 6 concurrent.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-/home/ext_csh/.mujoco/mujoco210}"
_MUJOCO_BIN="${MUJOCO_PY_MUJOCO_PATH}/bin"
_CONDA_LIB=/home/ext_csh/miniconda3/envs/capo/lib
export LD_LIBRARY_PATH="${_MUJOCO_BIN}:${_CONDA_LIB}:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
unset MUJOCO_PY_FORCE_CPU

PYTHON="${PYTHON:-/home/ext_csh/miniconda3/envs/capo/bin/python}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
OUT_DIR="${OUT_DIR:-results}"
CONFIG="${CONFIG:-configs/v8_hold.yaml}"
VARIANT="v8_hold"
RUN_TAG="v8_hold"
GPUS_CSV="${GPUS:-0,1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
POLL_SEC="${POLL_SEC:-20}"

IFS=',' read -r -a GPUS <<<"$GPUS_CSV"
MAX_PARALLEL=$(( ${#GPUS[@]} * JOBS_PER_GPU ))

ENVS=(hopper halfcheetah walker2d)
# Paper cells: medium / medium-expert / replay (not expert).
DATASETS=(medium medium-expert replay)

env_id_for() {
  local envb="$1" ds="$2"
  case "$ds" in
    medium) echo "${envb}-medium-v2" ;;
    expert) echo "${envb}-expert-v2" ;;
    medium-expert|medium_expert) echo "${envb}-medium-expert-v2" ;;
    replay|medium-replay|medium_replay) echo "${envb}-medium-replay-v2" ;;
    *) echo "${envb}-${ds}-v2" ;;
  esac
}

STATUS_FILE="$OUT_DIR/queue_status_v8_hold.tsv"
mkdir -p "$OUT_DIR" "$OUT_DIR/parallel_logs"
if [[ ! -f "$STATUS_FILE" ]]; then
  echo -e "variant\talgo\tenv_base\tdataset\tstatus\trun_dir\tstarted\tfinished" > "$STATUS_FILE"
fi

is_done() {
  local envb="$1" ds="$2"
  grep -q '^'"${VARIANT}"$'\ttd3_bc\t'"${envb}"$'\t'"${ds}"$'\t'done$'\t' "$STATUS_FILE" 2>/dev/null
}

is_failed() {
  local envb="$1" ds="$2"
  grep -q '^'"${VARIANT}"$'\ttd3_bc\t'"${envb}"$'\t'"${ds}"$'\t'failed$'\t' "$STATUS_FILE" 2>/dev/null
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

cell_live() {
  local envb="$1" ds="$2"
  pgrep -f "${PYTHON} scripts/run_capo.py .*--env_base ${envb} .*--dataset ${ds} .*--run_tag ${RUN_TAG}" >/dev/null 2>&1 \
    || pgrep -f "scripts/run_capo.py .*--env_base ${envb} .*--dataset ${ds} .*--run_tag ${RUN_TAG}" >/dev/null 2>&1
}

declare -A JOB_PID=()
declare -A JOB_GPU=()
declare -A JOB_START=()
declare -A JOB_ENV=()
declare -A SLOT_COUNT=()
for g in "${GPUS[@]}"; do
  SLOT_COUNT[$g]=0
done

# Count externally launched live jobs toward GPU slots (best-effort via CUDA_VISIBLE_DEVICES).
adopt_external() {
  local envb="$1" ds="$2"
  local pid gpu key="${envb}|${ds}"
  if [[ -n "${JOB_PID[$key]+x}" ]] && kill -0 "${JOB_PID[$key]}" 2>/dev/null; then
    return 0
  fi
  pid="$(pgrep -f "scripts/run_capo.py .*--env_base ${envb} .*--dataset ${ds} .*--run_tag ${RUN_TAG}" | head -1 || true)"
  [[ -n "$pid" ]] || return 1
  gpu="$(tr '\0' '\n' <"/proc/$pid/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)"
  gpu="${gpu%%,*}"
  [[ -n "$gpu" ]] || gpu="${GPUS[0]}"
  JOB_PID[$key]="$pid"
  JOB_GPU[$key]="$gpu"
  JOB_START[$key]="$(date '+%F %T')"
  JOB_ENV[$key]="$envb $ds"
  if [[ -n "${SLOT_COUNT[$gpu]+x}" ]]; then
    SLOT_COUNT[$gpu]=$(( SLOT_COUNT[$gpu] + 1 ))
  fi
  mark "$envb" "$ds" "running" "pending" "${JOB_START[$key]}" ""
  echo "[adopt] $VARIANT td3_bc $envb $ds pid=$pid gpu=$gpu"
  return 0
}

pick_gpu() {
  local g best="" best_n=999
  for g in "${GPUS[@]}"; do
    local n="${SLOT_COUNT[$g]}"
    if (( n < JOBS_PER_GPU && n < best_n )); then
      best="$g"
      best_n=$n
    fi
  done
  [[ -n "$best" ]] || return 1
  echo "$best"
}

total_running() {
  local k n=0
  for k in "${!JOB_PID[@]}"; do
    if kill -0 "${JOB_PID[$k]}" 2>/dev/null; then
      n=$((n + 1))
    fi
  done
  echo "$n"
}

reap() {
  local key envb ds pid gpu started finished run_dir env_id rc
  for key in "${!JOB_PID[@]}"; do
    pid="${JOB_PID[$key]}"
    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi
    envb="${key%%|*}"
    ds="${key##*|}"
    gpu="${JOB_GPU[$key]}"
    started="${JOB_START[$key]}"
    finished="$(date '+%F %T')"
    env_id="$(env_id_for "$envb" "$ds")"
    run_dir="$(latest_run_dir "$env_id" "$SEED" || echo unknown)"
    rc=1
    # wait only works for children of this shell; adopted PIDs use summary.json.
    if wait "$pid" 2>/dev/null; then
      rc=0
    else
      wrc=$?
      if [[ $wrc -eq 127 ]]; then
        # not our child
        if [[ -f "$run_dir/summary.json" ]]; then rc=0; else rc=1; fi
      else
        rc=$wrc
      fi
    fi
    if [[ $rc -eq 0 ]] || [[ -f "$run_dir/summary.json" ]]; then
      mark "$envb" "$ds" "done" "$run_dir" "$started" "$finished"
      echo "[ok] $VARIANT $envb $ds → $run_dir"
      if [[ -d "$run_dir" && -f "$run_dir/metrics.jsonl" ]]; then
        "$PYTHON" scripts/plot_training_curve.py "$run_dir" \
          >>"$OUT_DIR/parallel_logs/plot_${envb}_${ds}.log" 2>&1 \
          && echo "[plot] $run_dir/training_curve.png" \
          || echo "[plot] failed for $run_dir (see parallel_logs/plot_${envb}_${ds}.log)"
      fi
    else
      mark "$envb" "$ds" "failed" "$run_dir" "$started" "$finished"
      echo "[fail] $VARIANT $envb $ds pid=$pid (rc=$rc)"
    fi
    if [[ -n "${SLOT_COUNT[$gpu]+x}" ]] && (( SLOT_COUNT[$gpu] > 0 )); then
      SLOT_COUNT[$gpu]=$(( SLOT_COUNT[$gpu] - 1 ))
    fi
    unset "JOB_PID[$key]" "JOB_GPU[$key]" "JOB_START[$key]" "JOB_ENV[$key]"
  done
}

launch_one() {
  local envb="$1" ds="$2" gpu env_id started key log
  gpu="$(pick_gpu)" || return 1
  env_id="$(env_id_for "$envb" "$ds")"
  started="$(date '+%F %T')"
  key="${envb}|${ds}"
  log="$OUT_DIR/parallel_logs/${VARIANT}_${envb}_${ds}_s${SEED}_gpu${gpu}.log"
  mark "$envb" "$ds" "running" "pending" "$started" ""
  echo "[run] $VARIANT td3_bc $envb $ds → $env_id gpu=$gpu (slots gpu$gpu=$((SLOT_COUNT[$gpu]+1))/$JOBS_PER_GPU total<=$MAX_PARALLEL)"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHON
    exec "$PYTHON" scripts/run_capo.py \
      --config "$CONFIG" \
      --algorithm td3_bc \
      --env_base "$envb" \
      --dataset "$ds" \
      --seed "$SEED" \
      --device "$DEVICE" \
      --out_dir "$OUT_DIR" \
      --run_tag "$RUN_TAG"
  ) >"$log" 2>&1 &
  local pid=$!
  JOB_PID[$key]="$pid"
  JOB_GPU[$key]="$gpu"
  JOB_START[$key]="$started"
  JOB_ENV[$key]="$envb $ds"
  SLOT_COUNT[$gpu]=$(( SLOT_COUNT[$gpu] + 1 ))
}

echo "[CAPO v8_hold parallel] max=$MAX_PARALLEL (${JOBS_PER_GPU}/gpu × gpus=${GPUS_CSV}) datasets=${DATASETS[*]} → $OUT_DIR"

# Adopt every live v8_hold trainer (incl. leftover expert) for GPU slot accounting.
while read -r pid envb ds; do
  [[ -n "${pid:-}" && -n "${envb:-}" && -n "${ds:-}" ]] || continue
  key="${envb}|${ds}"
  [[ -n "${JOB_PID[$key]+x}" ]] && continue
  adopt_external "$envb" "$ds" || true
done < <(
  pgrep -af "scripts/run_capo.py .*--run_tag ${RUN_TAG}" 2>/dev/null \
    | sed -n 's/^\([0-9][0-9]*\).*--env_base \([^ ]*\) .*--dataset \([^ ]*\).*/\1 \2 \3/p' \
    || true
)

# Build paper work list.
PENDING=()
for envb in "${ENVS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    if is_done "$envb" "$ds"; then
      echo "[skip] $VARIANT td3_bc $envb $ds (already done)"
      continue
    fi
    if cell_live "$envb" "$ds"; then
      adopt_external "$envb" "$ds" || true
      continue
    fi
    # Leave failed cells retryable unless SKIP_FAILED=1
    if [[ "${SKIP_FAILED:-0}" == "1" ]] && is_failed "$envb" "$ds"; then
      echo "[skip] $VARIANT td3_bc $envb $ds (failed)"
      continue
    fi
    PENDING+=("${envb}|${ds}")
  done
done
echo "[queue] pending=${#PENDING[@]}: ${PENDING[*]:-none}"

idx=0
while (( idx < ${#PENDING[@]} )) || (( $(total_running) > 0 )); do
  reap
  while (( idx < ${#PENDING[@]} )) && (( $(total_running) < MAX_PARALLEL )); do
    if ! pick_gpu >/dev/null; then
      break
    fi
    key="${PENDING[$idx]}"
    idx=$((idx + 1))
    envb="${key%%|*}"
    ds="${key##*|}"
    if cell_live "$envb" "$ds"; then
      adopt_external "$envb" "$ds" || true
      continue
    fi
    launch_one "$envb" "$ds" || break
  done
  if (( idx >= ${#PENDING[@]} )) && (( $(total_running) == 0 )); then
    break
  fi
  sleep "$POLL_SEC"
done

reap
echo "[CAPO v8_hold parallel] finished. status=$STATUS_FILE"
"$PYTHON" scripts/summarize_matrix.py --out_dir "$OUT_DIR" || true
