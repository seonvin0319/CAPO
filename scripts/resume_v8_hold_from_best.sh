#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-/home/ext_csh/.mujoco/mujoco210}"
_MUJOCO_BIN="${MUJOCO_PY_MUJOCO_PATH}/bin"
_CONDA_LIB="$(dirname "${PYTHON:-/home/ext_csh/miniconda3/envs/capo/bin/python}")/../lib"
_CONDA_LIB="$(cd "$_CONDA_LIB" 2>/dev/null && pwd || echo /home/ext_csh/miniconda3/envs/capo/lib)"
export LD_LIBRARY_PATH="${_MUJOCO_BIN}:${_CONDA_LIB}:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export C_INCLUDE_PATH="${_CONDA_LIB%/lib}/include${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
unset MUJOCO_PY_FORCE_CPU
PYTHON="${PYTHON:-/home/ext_csh/miniconda3/envs/capo/bin/python}"
CONFIG="${CONFIG:-configs/v8_hold.yaml}"
OUT_DIR="${OUT_DIR:-results}"
GPUS=(0 1)
JOBS_PER_GPU=3
POLL_SEC=20
STATUS="$OUT_DIR/queue_status_v8_hold.tsv"

mark() {
  local envb="$1" ds="$2" status="$3" run_dir="$4" started="$5" finished="${6:-}"
  local tmp; tmp="$(mktemp)"
  awk -F'\t' -v e="$envb" -v d="$ds" 'NR==1{print;next} !($1=="v8_hold" && $3==e && $4==d){print}' "$STATUS" >"$tmp"
  printf "v8_hold\ttd3_bc\t%s\t%s\t%s\t%s\t%s\t%s\n" "$envb" "$ds" "$status" "$run_dir" "$started" "$finished" >>"$tmp"
  mv "$tmp" "$STATUS"
}

# env|dataset|run_dir_or_EMPTY for fresh
CELLS=(
  "hopper|medium-expert|results/td3_bc/hopper-medium-expert-v2/s0/0803_0136_v8_hold_td3_bc_hopper-medium-expert-v2_s0"
  "hopper|replay|results/td3_bc/hopper-medium-replay-v2/s0/0803_0035_v8_hold_td3_bc_hopper-medium-replay-v2_s0"
  "halfcheetah|medium|results/td3_bc/halfcheetah-medium-v2/s0/0803_0036_v8_hold_td3_bc_halfcheetah-medium-v2_s0"
  "halfcheetah|medium-expert|results/td3_bc/halfcheetah-medium-expert-v2/s0/0803_0227_v8_hold_td3_bc_halfcheetah-medium-expert-v2_s0"
  "halfcheetah|replay|results/td3_bc/halfcheetah-medium-replay-v2/s0/0803_0035_v8_hold_td3_bc_halfcheetah-medium-replay-v2_s0"
  "walker2d|medium|"
  "walker2d|medium-expert|"
  "walker2d|replay|"
)

declare -A JOB_PID JOB_GPU JOB_META SLOT
for g in "${GPUS[@]}"; do SLOT[$g]=0; done

pick_gpu() {
  local g best="" bn=999
  for g in "${GPUS[@]}"; do
    local n=${SLOT[$g]}
    if (( n < JOBS_PER_GPU && n < bn )); then best=$g; bn=$n; fi
  done
  [[ -n $best ]] || return 1
  echo "$best"
}

total() {
  local k n=0
  for k in "${!JOB_PID[@]}"; do kill -0 "${JOB_PID[$k]}" 2>/dev/null && n=$((n+1)) || true; done
  echo $n
}

reap() {
  local key pid meta envb ds rd gpu started finished rc
  for key in "${!JOB_PID[@]}"; do
    pid=${JOB_PID[$key]}
    kill -0 "$pid" 2>/dev/null && continue
    meta=${JOB_META[$key]}
    IFS='|' read -r envb ds rd <<<"$meta"
    gpu=${JOB_GPU[$key]}
    started=$(date '+%F %T')
    finished=$(date '+%F %T')
    if wait "$pid" 2>/dev/null; then rc=0; else rc=$?; [[ $rc -eq 127 ]] && rc=1; fi
    if [[ -n "$rd" && -f "$rd/summary.json" ]]; then rc=0; fi
    if [[ $rc -eq 0 ]]; then
      mark "$envb" "$ds" done "${rd:-pending}" "$started" "$finished"
      echo "[ok] $envb $ds"
      [[ -n "$rd" && -f "$rd/metrics.jsonl" ]] && $PYTHON scripts/plot_training_curve.py "$rd" >/dev/null 2>&1 || true
    else
      mark "$envb" "$ds" failed "${rd:-pending}" "$started" "$finished"
      echo "[fail] $envb $ds rc=$rc"
    fi
    SLOT[$gpu]=$(( SLOT[$gpu] > 0 ? SLOT[$gpu]-1 : 0 ))
    unset "JOB_PID[$key]" "JOB_GPU[$key]" "JOB_META[$key]"
  done
}

launch() {
  local envb="$1" ds="$2" rd="$3"
  local gpu; gpu=$(pick_gpu) || return 1
  local started; started=$(date '+%F %T')
  local key="${envb}|${ds}"
  local log="$OUT_DIR/parallel_logs/resume_v8_hold_${envb}_${ds}_gpu${gpu}.log"
  mark "$envb" "$ds" running "${rd:-pending}" "$started" ""
  echo "[run] resume $envb $ds rd=${rd:-FRESH} gpu=$gpu"
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    if [[ -n "$rd" ]]; then
      exec "$PYTHON" scripts/run_capo.py \
        --config "$CONFIG" --algorithm td3_bc \
        --env_base "$envb" --dataset "$ds" --seed 0 \
        --device cuda --out_dir "$OUT_DIR" --run_tag v8_hold \
        --resume_run_dir "$rd"
    else
      exec "$PYTHON" scripts/run_capo.py \
        --config "$CONFIG" --algorithm td3_bc \
        --env_base "$envb" --dataset "$ds" --seed 0 \
        --device cuda --out_dir "$OUT_DIR" --run_tag v8_hold
    fi
  ) >"$log" 2>&1 &
  JOB_PID[$key]=$!
  JOB_GPU[$key]=$gpu
  JOB_META[$key]="$envb|$ds|$rd"
  SLOT[$gpu]=$(( SLOT[$gpu]+1 ))
}

is_done() {
  grep -q $'^v8_hold\ttd3_bc\t'"$1"$'\t'"$2"$'\t'done$'\t' "$STATUS" 2>/dev/null
}

cell_live_pid() {
  local envb="$1" ds="$2"
  pgrep -f "scripts/run_capo.py .*--env_base ${envb} .*--dataset ${ds} .*--run_tag v8_hold" | head -1 || true
}

adopt() {
  local envb="$1" ds="$2" rd="$3"
  local pid gpu key="${envb}|${ds}"
  pid="$(cell_live_pid "$envb" "$ds")"
  [[ -n "$pid" ]] || return 1
  gpu="$(tr '\0' '\n' </proc/$pid/environ 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)"
  gpu="${gpu%%,*}"; [[ -n "$gpu" ]] || gpu=0
  JOB_PID[$key]="$pid"
  JOB_GPU[$key]="$gpu"
  JOB_META[$key]="$envb|$ds|$rd"
  if [[ -n "${SLOT[$gpu]+x}" ]]; then SLOT[$gpu]=$(( SLOT[$gpu] + 1 )); fi
  mark "$envb" "$ds" running "${rd:-pending}" "$(date '+%F %T')" ""
  echo "[adopt] $envb $ds pid=$pid gpu=$gpu"
}

QUEUE=()
for cell in "${CELLS[@]}"; do
  IFS='|' read -r envb ds rd <<<"$cell"
  if is_done "$envb" "$ds"; then
    echo "[skip] $envb $ds (done)"
    continue
  fi
  if [[ -n "$(cell_live_pid "$envb" "$ds")" ]]; then
    adopt "$envb" "$ds" "$rd" || true
    continue
  fi
  QUEUE+=("$cell")
done

echo "[resume v8_hold] pending=${#QUEUE[@]} live=$(total) max=$(( ${#GPUS[@]} * JOBS_PER_GPU ))"
idx=0
while (( idx < ${#QUEUE[@]} )) || (( $(total) > 0 )); do
  reap
  while (( idx < ${#QUEUE[@]} )) && (( $(total) < ${#GPUS[@]} * JOBS_PER_GPU )); do
    pick_gpu >/dev/null || break
    cell=${QUEUE[$idx]}; idx=$((idx+1))
    IFS='|' read -r envb ds rd <<<"$cell"
    if [[ -n "$(cell_live_pid "$envb" "$ds")" ]]; then
      adopt "$envb" "$ds" "$rd" || true
      continue
    fi
    launch "$envb" "$ds" "$rd" || break
  done
  if (( idx >= ${#QUEUE[@]} )) && (( $(total) == 0 )); then break; fi
  sleep "$POLL_SEC"
done
reap
echo "[resume v8_hold] finished"
