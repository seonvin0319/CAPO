#!/usr/bin/env bash
# Wait for the live IQL matrix (defaults+baseline), then launch v8_hold IQL × 9.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WAIT_PID_FILE="${WAIT_PID_FILE:-results/queue_iql.pid}"
POLL_SEC="${POLL_SEC:-60}"
LOG="${LOG:-results/queue_v8_hold_iql_waiter.log}"
PYTHON_HINT="${PYTHON_HINT:-/home/offrl/miniconda3/envs/offrl/bin/python}"

master=""
if [[ -f "$WAIT_PID_FILE" ]]; then
  master="$(cat "$WAIT_PID_FILE" || true)"
fi

{
  echo "[$(date '+%F %T')] waiter start; wait_pid_file=$WAIT_PID_FILE master=${master:-none}"
  if [[ -n "${master}" ]] && kill -0 "$master" 2>/dev/null; then
    echo "[$(date '+%F %T')] waiting for IQL matrix master pid=$master to exit…"
    while kill -0 "$master" 2>/dev/null; do
      sleep "$POLL_SEC"
    done
    echo "[$(date '+%F %T')] master $master exited"
  else
    echo "[$(date '+%F %T')] no live master in $WAIT_PID_FILE; checking for stray run_capo…"
  fi

  # Wait until no offrl IQL trainer remains.
  while pgrep -f "${PYTHON_HINT} scripts/run_capo.py" >/dev/null 2>&1 \
     || pgrep -f 'bash scripts/run_matrix_iql.sh' >/dev/null 2>&1; do
    echo "[$(date '+%F %T')] IQL trainer/matrix still alive; sleep ${POLL_SEC}s"
    sleep "$POLL_SEC"
  done

  echo "[$(date '+%F %T')] launching scripts/run_matrix_v8_hold_iql.sh"
  nohup bash "$ROOT/scripts/run_matrix_v8_hold_iql.sh" > "$ROOT/results/queue_v8_hold_iql.log" 2>&1 &
  echo $! > "$ROOT/results/queue_v8_hold_iql.pid"
  # Prefer the real matrix bash pid if wrapper differs
  sleep 1
  real="$(pgrep -f 'bash scripts/run_matrix_v8_hold_iql.sh' | head -1 || true)"
  if [[ -n "$real" ]]; then
    echo "$real" > "$ROOT/results/queue_v8_hold_iql.pid"
  fi
  echo "[$(date '+%F %T')] v8_hold IQL matrix pid=$(cat "$ROOT/results/queue_v8_hold_iql.pid") log=results/queue_v8_hold_iql.log"
} >>"$LOG" 2>&1
