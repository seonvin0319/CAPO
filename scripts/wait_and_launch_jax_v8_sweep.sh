#!/usr/bin/env bash
# Wait until no live torch v8_hold trainers / parallel master, then launch JAX sweep.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
POLL_SEC="${POLL_SEC:-60}"
LOG="$ROOT/results/wait_jax_v8_sweep.log"
MASTER_LOG="$ROOT/results/queue_master_v8j_sweep.log"
mkdir -p "$ROOT/results"

torch_busy() {
  pgrep -f "scripts/run_capo.py .*--run_tag v8_hold" >/dev/null 2>&1 \
    || pgrep -f "scripts/run_capo.py .*--run_tag v8_stale_" >/dev/null 2>&1 \
    || pgrep -f "scripts/run_capo.py .*--run_tag v8_period100k" >/dev/null 2>&1 \
    || pgrep -f "scripts/run_matrix_v8_hold_parallel.sh" >/dev/null 2>&1 \
    || pgrep -f "scripts/resume_v8_hold_from_best.sh" >/dev/null 2>&1 \
    || pgrep -f "scripts/run_stale_gate_ablation.sh" >/dev/null 2>&1
}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "waiting for torch v8_hold to finish before JAX sweep"
while torch_busy; do
  n="$(pgrep -af 'scripts/run_capo.py .*--run_tag v8_hold' 2>/dev/null | wc -l || true)"
  log "still busy: torch_trainers≈${n} — sleep ${POLL_SEC}s"
  sleep "$POLL_SEC"
done

if pgrep -f "scripts/run_matrix_jax_v8_sweep.sh" >/dev/null 2>&1; then
  log "JAX sweep master already running — exit"
  exit 0
fi

log "torch clear — launching JAX sweep (6-way)"
nohup bash scripts/run_matrix_jax_v8_sweep.sh >"$MASTER_LOG" 2>&1 &
log "master_pid=$! log=$MASTER_LOG"
