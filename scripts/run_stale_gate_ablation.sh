#!/usr/bin/env bash
# Stale-gate / period ablation on the two collapsed hopper cells.
# 1) stale→disable, period 50k
# 2) stale→keep_old, period 50k
# 3) stale→replace_new (baseline), period 100k
# 4) stale→keep_old, period 100k
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-/home/ext_csh/.mujoco/mujoco210}"
_MUJOCO_BIN="${MUJOCO_PY_MUJOCO_PATH}/bin"
_CONDA_LIB=/home/ext_csh/miniconda3/envs/capo/lib
export LD_LIBRARY_PATH="${_MUJOCO_BIN}:${_CONDA_LIB}:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset MUJOCO_PY_FORCE_CPU

PYTHON="${PYTHON:-/home/ext_csh/miniconda3/envs/capo/bin/python}"
SEED="${SEED:-0}"
CONFIG="${CONFIG:-configs/v8_hold.yaml}"
OUT_DIR="${OUT_DIR:-results}"
GPUS_CSV="${GPUS:-0,1}"
IFS=',' read -r -a GPUS <<<"$GPUS_CSV"

LOG_DIR="$OUT_DIR/parallel_logs/stale_gate_ablation"
mkdir -p "$LOG_DIR" "$OUT_DIR"
STATUS="$OUT_DIR/queue_status_stale_gate_ablation.tsv"
if [[ ! -f "$STATUS" ]]; then
  echo -e "variant\tenv_base\tdataset\tstale_action\tcapo_period\tstatus\trun_dir\tpid\tgpu\tstarted" > "$STATUS"
fi

# variant_tag | stale_action | capo_period
VARIANTS=(
  "v8_stale_disable|disable_teacher|50000"
  "v8_stale_keep|keep_old|50000"
  "v8_period100k|replace_new|100000"
  "v8_stale_keep_p100k|keep_old|100000"
)
DATASETS=(medium-replay medium-expert)
ENV_BASE=hopper

idx=0
for vspec in "${VARIANTS[@]}"; do
  IFS='|' read -r TAG STALE PERIOD <<<"$vspec"
  for DS in "${DATASETS[@]}"; do
    GPU="${GPUS[$((idx % ${#GPUS[@]}))]}"
    LOG="$LOG_DIR/${TAG}_${ENV_BASE}_${DS}_s${SEED}.log"
    STARTED="$(date -Iseconds)"
    echo "[launch] tag=$TAG ds=$DS stale=$STALE period=$PERIOD gpu=$GPU log=$LOG"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" scripts/run_capo.py \
      --config "$CONFIG" \
      --algorithm td3_bc \
      --env_base "$ENV_BASE" \
      --dataset "$DS" \
      --seed "$SEED" \
      --device cuda \
      --out_dir "$OUT_DIR" \
      --run_tag "$TAG" \
      --stale_incumbent_action "$STALE" \
      --capo_period "$PERIOD" \
      >"$LOG" 2>&1 &
    PID=$!
    # resolve run_dir after a short wait (created at startup)
    sleep 2
    RUN_DIR=""
    ENV_ID="${ENV_BASE}-${DS}-v2"
    # medium-replay alias → hopper-medium-replay-v2
    case "$DS" in
      medium-replay|replay) ENV_ID="${ENV_BASE}-medium-replay-v2" ;;
      medium-expert) ENV_ID="${ENV_BASE}-medium-expert-v2" ;;
    esac
    BASE="$OUT_DIR/td3_bc/${ENV_ID}/s${SEED}"
    if [[ -d "$BASE" ]]; then
      RUN_DIR="$(ls -dt "$BASE"/*_"${TAG}"_td3_bc_"${ENV_ID}"_s${SEED} 2>/dev/null | head -1 || true)"
    fi
    printf "%s\t%s\t%s\t%s\t%s\trunning\t%s\t%s\t%s\t%s\n" \
      "$TAG" "$ENV_BASE" "$DS" "$STALE" "$PERIOD" "${RUN_DIR:-}" "$PID" "$GPU" "$STARTED" >> "$STATUS"
    idx=$((idx + 1))
  done
done

echo "Launched $idx jobs. Status: $STATUS"
echo "Logs: $LOG_DIR"
ps -ef | grep 'scripts/run_capo.py' | grep -E 'v8_stale_|v8_period100k' | grep -v grep || true
