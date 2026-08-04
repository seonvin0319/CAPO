#!/usr/bin/env bash
# Fill free GPU slots (max 6 = 3/gpu × 2) with JAX v8 sweep cells.
# Does NOT kill torch; adopts live torch/jax toward the budget.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-/home/ext_csh/.mujoco/mujoco210}"
_MUJOCO_BIN="${MUJOCO_PY_MUJOCO_PATH}/bin"
_OFFRL_LIB=/home/ext_csh/miniconda3/envs/offrl/lib
_CAPO_LIB=/home/ext_csh/miniconda3/envs/capo/lib
export LD_LIBRARY_PATH="${_MUJOCO_BIN}:${_OFFRL_LIB}:${_CAPO_LIB}:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
unset MUJOCO_PY_FORCE_CPU

PYTHON="/home/ext_csh/miniconda3/envs/offrl/bin/python"
SEED=0
DEVICE=cuda
OUT_DIR=results
CONFIG=configs/v8_hold.yaml
VARIANT=v8j_sweep
RUN_TAG_PREFIX=v8j
GPUS=(0 1)
JOBS_PER_GPU=3
MAX_PARALLEL=$(( ${#GPUS[@]} * JOBS_PER_GPU ))
POLL_SEC="${POLL_SEC:-15}"

env_id_for() {
  case "$2" in
    medium) echo "$1-medium-v2" ;;
    medium-expert) echo "$1-medium-expert-v2" ;;
    replay) echo "$1-medium-replay-v2" ;;
    *) echo "$1-$2-v2" ;;
  esac
}
tag_for() {
  local mams="$1" rcm="$2" lt="$3" ms rs ls
  [[ "$mams" == "0.15" ]] && ms=015 || ms=02
  [[ "$rcm" == "0.0" || "$rcm" == "0" ]] && rs=0 || rs=001
  [[ "$lt" == "0.5" ]] && ls=05 || ls=10
  echo "mams${ms}_rcm${rs}_lt${ls}"
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
  local envb="$1" ds="$2"; shift 2
  local spec tag mams rcm lt
  for spec in "$@"; do
    IFS='|' read -r tag mams rcm lt <<<"$spec"
    PENDING+=("${envb}|${ds}|${tag}|${mams}|${rcm}|${lt}")
  done
}
append_wave hopper medium-expert "${FULL8[@]}"
append_wave hopper medium "${FULL8[@]}"
append_wave halfcheetah medium-expert "${FULL8[@]}"
append_wave walker2d medium-expert "${FULL8[@]}"
append_wave halfcheetah medium "${FULL8[@]}"
append_wave walker2d medium "${FULL8[@]}"
append_wave hopper replay "${REPLAY4[@]}"
append_wave halfcheetah replay "${REPLAY4[@]}"
append_wave walker2d replay "${REPLAY4[@]}"

STATUS_FILE="$OUT_DIR/queue_status_${VARIANT}.tsv"
mkdir -p "$OUT_DIR" "$OUT_DIR/parallel_logs"
[[ -f "$STATUS_FILE" ]] || echo -e "variant\talgo\tenv_base\tdataset\ttag\tstatus\trun_dir\tstarted\tfinished" > "$STATUS_FILE"

is_done() {
  grep -q $'^'"${VARIANT}"$'\ttd3_bc\t'"$1"$'\t'"$2"$'\t'"$3"$'\t'done$'\t' "$STATUS_FILE" 2>/dev/null
}
mark() {
  local envb="$1" ds="$2" tag="$3" status="$4" run_dir="$5" started="$6" finished="${7:-}"
  local tmp; tmp="$(mktemp)"
  awk -F'\t' -v v="$VARIANT" -v e="$envb" -v d="$ds" -v t="$tag" '
    NR==1{print;next} !($1==v && $3==e && $4==d && $5==t){print}
  ' "$STATUS_FILE" >"$tmp"
  printf "%s\ttd3_bc\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$VARIANT" "$envb" "$ds" "$tag" "$status" "$run_dir" "$started" "$finished" >>"$tmp"
  mv "$tmp" "$STATUS_FILE"
}

count_live_trainers() {
  # torch + jax trainers (any run_tag)
  ps -eo pid,cmd | awk '/scripts\/run_capo(_jax)?\.py/ && !/awk/ {c++} END{print c+0}'
}

count_live_on_gpu() {
  local gpu="$1" n=0 pid
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    local vis
    vis="$(tr '\0' '\n' </proc/$pid/environ 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)"
    vis="${vis%%,*}"
    [[ "$vis" == "$gpu" ]] && n=$((n+1))
  done < <(ps -eo pid,cmd | awk '/scripts\/run_capo(_jax)?\.py/ && !/awk/ {print $1}')
  echo "$n"
}

pick_gpu() {
  local g best="" bn=999 n
  for g in "${GPUS[@]}"; do
    n="$(count_live_on_gpu "$g")"
    if (( n < JOBS_PER_GPU && n < bn )); then best=$g; bn=$n; fi
  done
  [[ -n "$best" ]] || return 1
  echo "$best"
}

declare -A JOB_PID JOB_GPU JOB_META JOB_START

reap_jax() {
  local key pid meta envb ds tag gpu started finished run_tag env_id run_dir rc
  for key in "${!JOB_PID[@]}"; do
    pid=${JOB_PID[$key]}
    if kill -0 "$pid" 2>/dev/null; then continue; fi
    meta=${JOB_META[$key]}
    IFS='|' read -r envb ds tag <<<"$meta"
    gpu=${JOB_GPU[$key]}
    started=${JOB_START[$key]}
    finished=$(date '+%F %T')
    run_tag="${RUN_TAG_PREFIX}_${tag}"
    env_id=$(env_id_for "$envb" "$ds")
    run_dir="$(ls -dt "$OUT_DIR/td3_bc/$env_id/s${SEED}"/*_${run_tag}_* 2>/dev/null | head -1 || echo unknown)"
    rc=1
    if wait "$pid" 2>/dev/null; then rc=0
    else
      wrc=$?
      [[ $wrc -eq 127 && -f "$run_dir/summary.json" ]] && rc=0 || rc=$wrc
      [[ -f "$run_dir/summary.json" ]] && rc=0
    fi
    if [[ $rc -eq 0 ]]; then
      mark "$envb" "$ds" "$tag" done "$run_dir" "$started" "$finished"
      echo "[ok] jax $envb $ds $tag → $run_dir"
      [[ -f "$run_dir/metrics.jsonl" ]] && \
        /home/ext_csh/miniconda3/envs/capo/bin/python scripts/plot_training_curve.py "$run_dir" >/dev/null 2>&1 || true
    else
      mark "$envb" "$ds" "$tag" failed "$run_dir" "$started" "$finished"
      echo "[fail] jax $envb $ds $tag rc=$rc"
    fi
    unset "JOB_PID[$key]" "JOB_GPU[$key]" "JOB_META[$key]" "JOB_START[$key]"
  done
}

launch_jax() {
  local envb="$1" ds="$2" tag="$3" mams="$4" rcm="$5" lt="$6"
  local gpu; gpu=$(pick_gpu) || return 1
  local started; started=$(date '+%F %T')
  local key="${envb}|${ds}|${tag}"
  local run_tag="${RUN_TAG_PREFIX}_${tag}"
  local log="$OUT_DIR/parallel_logs/${VARIANT}_${envb}_${ds}_${tag}_gpu${gpu}.log"
  mark "$envb" "$ds" "$tag" running pending "$started" ""
  echo "[run] jax $envb $ds $tag mams=$mams rcm=$rcm lt=$lt gpu=$gpu (live=$(count_live_trainers)/$MAX_PARALLEL)"
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    export JAX_PLATFORMS=cuda
    exec "$PYTHON" scripts/run_capo_jax.py \
      --config "$CONFIG" --algorithm td3_bc \
      --env_base "$envb" --dataset "$ds" --seed "$SEED" \
      --device "$DEVICE" --out_dir "$OUT_DIR" --run_tag "$run_tag" \
      --max_action_mse "$mams" --replace_cert_margin "$rcm" --lambda_T "$lt"
  ) >"$log" 2>&1 &
  JOB_PID[$key]=$!
  JOB_GPU[$key]=$gpu
  JOB_META[$key]="${envb}|${ds}|${tag}"
  JOB_START[$key]=$started
}

# Filter queue
QUEUE=()
for cell in "${PENDING[@]}"; do
  IFS='|' read -r envb ds tag mams rcm lt <<<"$cell"
  is_done "$envb" "$ds" "$tag" && continue
  QUEUE+=("$cell")
done
echo "[fill-jax] pending=${#QUEUE[@]} max=$MAX_PARALLEL python=$PYTHON"
echo "[fill-jax] will use free slots as torch finishes (no new torch launches)"

idx=0
while (( idx < ${#QUEUE[@]} )) || (( ${#JOB_PID[@]} > 0 )); do
  reap_jax
  while (( idx < ${#QUEUE[@]} )) && (( $(count_live_trainers) < MAX_PARALLEL )); do
    if ! pick_gpu >/dev/null; then break; fi
    cell=${QUEUE[$idx]}; idx=$((idx+1))
    IFS='|' read -r envb ds tag mams rcm lt <<<"$cell"
    launch_jax "$envb" "$ds" "$tag" "$mams" "$rcm" "$lt" || break
    # brief stagger so mujoco/jax init don't stampede
    sleep 2
  done
  if (( idx >= ${#QUEUE[@]} )) && (( ${#JOB_PID[@]} == 0 )); then
    # still wait if torch alive? no — our jax queue done; exit even if torch remains
    break
  fi
  sleep "$POLL_SEC"
done
reap_jax
echo "[fill-jax] finished status=$STATUS_FILE"
