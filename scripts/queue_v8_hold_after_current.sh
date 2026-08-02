#!/usr/bin/env bash
# Wait for the current CAPO matrix master, then launch v8_hold 9-cell matrix.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WAIT_PID_FILE="${WAIT_PID_FILE:-results/queue.pid}"
POLL_SEC="${POLL_SEC:-60}"
LOG="${LOG:-results/queue_v8_hold_waiter.log}"

master=""
if [[ -f "$WAIT_PID_FILE" ]]; then
  master="$(cat "$WAIT_PID_FILE" || true)"
fi

{
  echo "[$(date '+%F %T')] waiter start; wait_pid_file=$WAIT_PID_FILE master=${master:-none}"
  if [[ -n "${master}" ]] && kill -0 "$master" 2>/dev/null; then
    echo "[$(date '+%F %T')] waiting for matrix master pid=$master to exit…"
    while kill -0 "$master" 2>/dev/null; do
      sleep "$POLL_SEC"
    done
    echo "[$(date '+%F %T')] master $master exited"
  else
    echo "[$(date '+%F %T')] no live master in $WAIT_PID_FILE; launching v8_hold immediately"
  fi

  # Also wait until no run_capo child remains (brief grace).
  for _ in 1 2 3 4 5; do
    if pgrep -f '/home/choi/miniconda3/envs/offrl_backup/bin/python scripts/run_capo.py' >/dev/null 2>&1; then
      echo "[$(date '+%F %T')] run_capo still alive; sleep ${POLL_SEC}s"
      sleep "$POLL_SEC"
    else
      break
    fi
  done

  echo "[$(date '+%F %T')] launching scripts/run_matrix_v8_hold.sh"
  nohup bash "$ROOT/scripts/run_matrix_v8_hold.sh" > "$ROOT/results/queue_v8_hold.log" 2>&1 &
  echo $! > "$ROOT/results/queue_v8_hold.pid"
  echo "[$(date '+%F %T')] v8_hold matrix pid=$(cat "$ROOT/results/queue_v8_hold.pid") log=results/queue_v8_hold.log"
} >>"$LOG" 2>&1
