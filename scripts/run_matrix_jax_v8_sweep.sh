#!/usr/bin/env bash
# JAX sweep on v8_hold baseline (pilot_adaptive fixed).
# Priority: collapse-prone medium-expert first, then medium, then lighter replay.
# Packing: 3 jobs/GPU × GPUs 0,1 → max 6 concurrent.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-/home/ext_csh/.mujoco/mujoco210}"
_MUJOCO_BIN="${MUJOCO_PY_MUJOCO_PATH}/bin"
case ":${LD_LIBRARY_PATH:-}:" in
  *":${_MUJOCO_BIN}:"*) ;;
  *) export LD_LIBRARY_PATH="${_MUJOCO_BIN}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

PYTHON="${PYTHON:-/home/ext_csh/miniconda3/envs/offrl/bin/python}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
OUT_DIR="${OUT_DIR:-results}"
CONFIG="${CONFIG:-configs/v8_hold.yaml}"
VARIANT="v8j_sweep"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-v8j}"
GPUS_CSV="${GPUS:-0,1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
POLL_SEC="${POLL_SEC:-20}"

IFS=',' read -r -a GPUS <<<"$GPUS_CSV"
MAX_PARALLEL=$(( ${#GPUS[@]} * JOBS_PER_GPU ))

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

tag_for() {
  local mams="$1" rcm="$2" lt="$3"
  local mams_s rcm_s lt_s
  if [[ "$mams" == "0.15" ]]; then mams_s="015"; elif [[ "$mams" == "0.2" ]]; then mams_s="02"; else mams_s="$(printf '%s' "$mams" | tr -d '.')"; fi
  if [[ "$rcm" == "0.0" || "$rcm" == "0" ]]; then rcm_s="0"; else rcm_s="001"; fi
  if [[ "$lt" == "0.5" ]]; then lt_s="05"; else lt_s="10"; fi
  echo "mams${mams_s}_rcm${rcm_s}_lt${lt_s}"
}

FULL8=()
for mams in 0.2 0.15; do
  for rcm in 0.0 0.01; do
    for lt in 1.0 0.5; do
      FULL8+=("$(tag_for "$mams" "$rcm" "$lt")|$mams|$rcm|$lt")
    done
  done
done

REPLAY4=(
  "$(tag_for 0.2 0.0 1.0)|0.2|0.0|1.0"
  "$(tag_for 0.15 0.0 1.0)|0.15|0.0|1.0"
  "$(tag_for 0.2 0.01 1.0)|0.2|0.01|1.0"
  "$(tag_for 0.15 0.01 0.5)|0.15|0.01|0.5"
)

PENDING=()
append_wave() {
  local envb="$1" ds="$2"
  shift 2
  local spec tag mams rcm lt
  for spec in "$@"; do
    IFS='|' read -r tag mams rcm lt <<<"$spec"
    PENDING+=("${envb}|${ds}|${tag}|${mams}|${rcm}|${lt}")
  done
}

# Wave A — recommended min matrix
append_wave hopper medium-expert "${FULL8[@]}"
append_wave hopper medium "${FULL8[@]}"
# Wave B — other medium-expert
append_wave halfcheetah medium-expert "${FULL8[@]}"
append_wave walker2d medium-expert "${FULL8[@]}"
# Wave C — remaining medium
append_wave halfcheetah medium "${FULL8[@]}"
append_wave walker2d medium "${FULL8[@]}"
# Wave D — replay lighter
append_wave hopper replay "${REPLAY4[@]}"
append_wave halfcheetah replay "${REPLAY4[@]}"
append_wave walker2d replay "${REPLAY4[@]}"

STATUS_FILE="$OUT_DIR/queue_status_${VARIANT}.tsv"
mkdir -p "$OUT_DIR" "$OUT_DIR/parallel_logs"
if [[ ! -f "$STATUS_FILE" ]]; then
  echo -e "variant\talgo\tenv_base\tdataset\ttag\tstatus\trun_dir\tstarted\tfinished" > "$STATUS_FILE"
fi

is_done() {
  local envb="$1" ds="$2" tag="$3"
  grep -q '^'"${VARIANT}"$'\ttd3_bc\t'"${envb}"$'\t'"${ds}"$'\t'"${tag}"$'\t'done$'\t' "$STATUS_FILE" 2>/dev/null
}

is_failed() {
  local envb="$1" ds="$2" tag="$3"
  grep -q '^'"${VARIANT}"$'\ttd3_bc\t'"${envb}"$'\t'"${ds}"$'\t'"${tag}"$'\t'failed$'\t' "$STATUS_FILE" 2>/dev/null
}

mark() {
  local envb="$1" ds="$2" tag="$3" status="$4" run_dir="$5" started="$6" finished="${7:-}"
  local tmp
  tmp="$(mktemp)"
  awk -F'\t' -v v="$VARIANT" -v e="$envb" -v d="$ds" -v t="$tag" '
    NR==1 {print; next}
    !($1==v && $2=="td3_bc" && $3==e && $4==d && $5==t) {print}
  ' "$STATUS_FILE" > "$tmp"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$VARIANT" "td3_bc" "$envb" "$ds" "$tag" "$status" "$run_dir" "$started" "$finished" >> "$tmp"
  mv "$tmp" "$STATUS_FILE"
}

latest_run_dir() {
  local env_id="$1" seed="$2" run_tag="$3"
  local algo="td3_bc"
  local pat="[0-9][0-9][0-9][0-9]_*_${run_tag}_${algo}_${env_id}_s${seed}"
  local base="$OUT_DIR/$algo/$env_id/s${seed}"
  local hit=""
  if [[ -d "$base" ]]; then
    hit="$(ls -dt "$base"/$pat 2>/dev/null | head -1 || true)"
  fi
  [[ -n "$hit" ]] || return 1
  if [[ -L "$hit" ]]; then
    hit="$(readlink -f "$hit")"
  fi
  echo "$hit"
}

cell_live() {
  local envb="$1" ds="$2" tag="$3"
  local run_tag="${RUN_TAG_PREFIX}_${tag}"
  pgrep -f "scripts/run_capo_jax.py .*--env_base ${envb} .*--dataset ${ds} .*--run_tag ${run_tag}" >/dev/null 2>&1
}

declare -A JOB_PID=()
declare -A JOB_GPU=()
declare -A JOB_START=()
declare -A JOB_META=()
declare -A SLOT_COUNT=()
for g in "${GPUS[@]}"; do
  SLOT_COUNT[$g]=0
done

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
  local key envb ds tag pid gpu started finished run_dir env_id rc run_tag meta wrc
  for key in "${!JOB_PID[@]}"; do
    pid="${JOB_PID[$key]}"
    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi
    meta="${JOB_META[$key]}"
    IFS='|' read -r envb ds tag <<<"$meta"
    gpu="${JOB_GPU[$key]}"
    started="${JOB_START[$key]}"
    finished="$(date '+%F %T')"
    run_tag="${RUN_TAG_PREFIX}_${tag}"
    env_id="$(env_id_for "$envb" "$ds")"
    run_dir="$(latest_run_dir "$env_id" "$SEED" "$run_tag" || echo unknown)"
    rc=1
    if wait "$pid" 2>/dev/null; then
      rc=0
    else
      wrc=$?
      if [[ $wrc -eq 127 ]]; then
        if [[ -f "$run_dir/summary.json" ]]; then rc=0; else rc=1; fi
      else
        rc=$wrc
      fi
    fi
    if [[ $rc -eq 0 ]] || [[ -f "$run_dir/summary.json" ]]; then
      mark "$envb" "$ds" "$tag" "done" "$run_dir" "$started" "$finished"
      echo "[ok] $VARIANT $envb $ds $tag → $run_dir"
      if [[ -d "$run_dir" && -f "$run_dir/metrics.jsonl" ]]; then
        /home/ext_csh/miniconda3/envs/capo/bin/python scripts/plot_training_curve.py "$run_dir" \
          >>"$OUT_DIR/parallel_logs/plot_${VARIANT}_${envb}_${ds}_${tag}.log" 2>&1 \
          && echo "[plot] $run_dir/training_curve.png" \
          || echo "[plot] failed for $run_dir"
      fi
    else
      mark "$envb" "$ds" "$tag" "failed" "$run_dir" "$started" "$finished"
      echo "[fail] $VARIANT $envb $ds $tag pid=$pid (rc=$rc)"
    fi
    if [[ -n "${SLOT_COUNT[$gpu]+x}" ]] && (( SLOT_COUNT[$gpu] > 0 )); then
      SLOT_COUNT[$gpu]=$(( SLOT_COUNT[$gpu] - 1 ))
    fi
    unset "JOB_PID[$key]" "JOB_GPU[$key]" "JOB_START[$key]" "JOB_META[$key]"
  done
}

launch_one() {
  local envb="$1" ds="$2" tag="$3" mams="$4" rcm="$5" lt="$6"
  local gpu env_id started key log run_tag
  gpu="$(pick_gpu)" || return 1
  env_id="$(env_id_for "$envb" "$ds")"
  started="$(date '+%F %T')"
  key="${envb}|${ds}|${tag}"
  run_tag="${RUN_TAG_PREFIX}_${tag}"
  log="$OUT_DIR/parallel_logs/${VARIANT}_${envb}_${ds}_${tag}_s${SEED}_gpu${gpu}.log"
  mark "$envb" "$ds" "$tag" "running" "pending" "$started" ""
  echo "[run] $VARIANT $envb $ds $tag mams=$mams rcm=$rcm lt=$lt → $env_id gpu=$gpu"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHON
    export JAX_PLATFORMS=cuda
    exec "$PYTHON" scripts/run_capo_jax.py \
      --config "$CONFIG" \
      --algorithm td3_bc \
      --env_base "$envb" \
      --dataset "$ds" \
      --seed "$SEED" \
      --device "$DEVICE" \
      --out_dir "$OUT_DIR" \
      --run_tag "$run_tag" \
      --max_action_mse "$mams" \
      --replace_cert_margin "$rcm" \
      --lambda_T "$lt"
  ) >"$log" 2>&1 &
  local pid=$!
  JOB_PID[$key]="$pid"
  JOB_GPU[$key]="$gpu"
  JOB_START[$key]="$started"
  JOB_META[$key]="${envb}|${ds}|${tag}"
  SLOT_COUNT[$gpu]=$(( SLOT_COUNT[$gpu] + 1 ))
}

echo "[CAPO ${VARIANT}] max=$MAX_PARALLEL (${JOBS_PER_GPU}/gpu × gpus=${GPUS_CSV}) cells=${#PENDING[@]} → $OUT_DIR"
echo "[CAPO ${VARIANT}] python=$PYTHON config=$CONFIG"

QUEUE=()
for cell in "${PENDING[@]}"; do
  IFS='|' read -r envb ds tag mams rcm lt <<<"$cell"
  if is_done "$envb" "$ds" "$tag"; then
    echo "[skip] $envb $ds $tag (done)"
    continue
  fi
  if [[ "${SKIP_FAILED:-0}" == "1" ]] && is_failed "$envb" "$ds" "$tag"; then
    echo "[skip] $envb $ds $tag (failed)"
    continue
  fi
  if cell_live "$envb" "$ds" "$tag"; then
    echo "[live] $envb $ds $tag — leave running"
    continue
  fi
  QUEUE+=("$cell")
done
echo "[queue] pending=${#QUEUE[@]}"

idx=0
while (( idx < ${#QUEUE[@]} )) || (( $(total_running) > 0 )); do
  reap
  while (( idx < ${#QUEUE[@]} )) && (( $(total_running) < MAX_PARALLEL )); do
    if ! pick_gpu >/dev/null; then
      break
    fi
    cell="${QUEUE[$idx]}"
    idx=$((idx + 1))
    IFS='|' read -r envb ds tag mams rcm lt <<<"$cell"
    if cell_live "$envb" "$ds" "$tag"; then
      continue
    fi
    launch_one "$envb" "$ds" "$tag" "$mams" "$rcm" "$lt" || break
  done
  if (( idx >= ${#QUEUE[@]} )) && (( $(total_running) == 0 )); then
    break
  fi
  sleep "$POLL_SEC"
done

reap
echo "[CAPO ${VARIANT}] finished. status=$STATUS_FILE"
