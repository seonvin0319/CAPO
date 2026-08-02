#!/usr/bin/env bash
# choi pipeline (sequential, one GPU):
#   1) wait until current CAPO matrix finishes (9 cells in results/queue_status.tsv)
#      — live cell is walker2d-replay; do not kill it from outside this script
#   2) stop the live matrix master so it does not continue with stale in-memory plan
#   3) CAPO medium-expert × 3 (replaces cancelled expert plan)
#   4) baseline td3bc × 9 (medium / medium-expert / replay)
# v8_hold is owned by ext_csh (not launched here).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

POLL_SEC="${POLL_SEC:-15}"
LOG="${LOG:-results/queue_choi_pipeline.log}"
CAPO_STATUS="${CAPO_STATUS:-results/queue_status.tsv}"
MASTER_PID_FILE="${MASTER_PID_FILE:-results/queue.pid}"
PYTHON_BIN="${PYTHON_BIN:-/home/choi/miniconda3/envs/offrl_backup/bin/python}"

capo_done_count() {
  [[ -f "$CAPO_STATUS" ]] || { echo 0; return; }
  # Count original medium/expert/replay capo cells (the live matrix).
  awk -F'\t' '
    $1=="capo" && $2=="td3_bc" && $5=="done" &&
    ($4=="medium" || $4=="expert" || $4=="replay") {c++}
    END{print c+0}
  ' "$CAPO_STATUS"
}

stop_live_matrix() {
  local master=""
  if [[ -f "$MASTER_PID_FILE" ]]; then
    master="$(cat "$MASTER_PID_FILE" || true)"
  fi
  if [[ -n "${master}" ]] && kill -0 "$master" 2>/dev/null; then
    echo "[$(date '+%F %T')] stopping matrix master pid=$master"
    kill "$master" 2>/dev/null || true
    sleep 2
    kill -9 "$master" 2>/dev/null || true
  fi
  for pid in $(pgrep -f "${PYTHON_BIN} scripts/run_capo.py" || true); do
    echo "[$(date '+%F %T')] stopping run_capo pid=$pid"
    kill "$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in $(pgrep -f "${PYTHON_BIN} scripts/run_capo.py" || true); do
    kill -9 "$pid" 2>/dev/null || true
  done
}

wait_pid_file() {
  local pid_file="$1" name="$2"
  [[ -f "$pid_file" ]] || return 0
  local pid
  pid="$(cat "$pid_file" || true)"
  [[ -n "${pid}" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[$(date '+%F %T')] waiting for $name pid=$pid …"
    while kill -0 "$pid" 2>/dev/null; do
      sleep "$POLL_SEC"
    done
    echo "[$(date '+%F %T')] $name pid=$pid exited"
  fi
}

{
  echo "[$(date '+%F %T')] choi pipeline start: finish capo9 → medium-expert×3 → baseline9 (expert cancelled)"

  echo "[$(date '+%F %T')] phase1: wait for 9 capo cells done in $CAPO_STATUS"
  while true; do
    n="$(capo_done_count)"
    echo "[$(date '+%F %T')] capo done=$n/9"
    if [[ "$n" -ge 9 ]]; then
      break
    fi
    sleep "$POLL_SEC"
  done

  stop_live_matrix

  echo "[$(date '+%F %T')] skip v8_hold (owned by ext_csh)"

  echo "[$(date '+%F %T')] phase2: CAPO medium-expert × 3"
  nohup bash "$ROOT/scripts/run_matrix_capo_medium_expert.sh" \
    > "$ROOT/results/queue_capo_medium_expert.log" 2>&1 &
  echo $! > "$ROOT/results/queue_capo_medium_expert.pid"
  wait_pid_file "$ROOT/results/queue_capo_medium_expert.pid" "capo_medium_expert"

  echo "[$(date '+%F %T')] phase3: launch baseline td3bc × 9 (medium / medium-expert / replay)"
  nohup bash "$ROOT/scripts/run_matrix_baseline_td3bc.sh" > "$ROOT/results/queue_baseline.log" 2>&1 &
  echo $! > "$ROOT/results/queue_baseline.pid"
  wait_pid_file "$ROOT/results/queue_baseline.pid" "baseline"

  echo "[$(date '+%F %T')] choi pipeline finished"
} >>"$LOG" 2>&1
