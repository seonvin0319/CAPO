#!/usr/bin/env bash
# After the live halfcheetah-expert cell finishes: kill frozen old matrix,
# rewrite queue_status for medium/medium-expert/replay, relaunch matrix.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MATRIX_PID="${1:-}"
TRAIN_PID="${2:-}"
OUT_DIR="${OUT_DIR:-results}"
STATUS="$OUT_DIR/queue_status.tsv"
LOG="$OUT_DIR/requeue_medium_expert.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "wait trainer=$TRAIN_PID matrix(frozen)=$MATRIX_PID"
if [[ -n "$TRAIN_PID" ]]; then
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 30
  done
fi
log "trainer exited"

# Mark finishing expert cell done (if artifacts present)
EXPERT_DIR="$(ls -dt "$OUT_DIR"/cql/halfcheetah-expert-v2/s0/[0-9]*_capo_cql_halfcheetah-expert-v2_s0 2>/dev/null | head -1 || true)"
FINISHED="$(date '+%F %T')"
STARTED="2026-08-03 07:48:49"

# Kill frozen / old matrix so it cannot continue the expert schedule
if [[ -n "$MATRIX_PID" ]] && kill -0 "$MATRIX_PID" 2>/dev/null; then
  kill -CONT "$MATRIX_PID" 2>/dev/null || true
  kill "$MATRIX_PID" 2>/dev/null || true
  sleep 1
  kill -9 "$MATRIX_PID" 2>/dev/null || true
  log "killed old matrix $MATRIX_PID"
fi
# Also clear any stray matrix
pkill -f 'bash scripts/run_matrix_cql.sh' 2>/dev/null || true
rm -f "$OUT_DIR/queue_cql.pid"

# Rewrite status: keep finished medium/replay (+ finishing expert as cancelled/done record),
# leave medium-expert and remaining cells for the new matrix to run.
{
  printf 'variant\talgo\tenv_base\tdataset\tstatus\trun_dir\tstarted\tfinished\n'
  printf 'capo\tcql\thopper\tmedium\tdone\t%s\t2026-08-03 01:03:47\t2026-08-03 02:44:05\n' \
    'results/cql/hopper-medium-v2/s0/0803_0103_capo_cql_hopper-medium-v2_s0'
  printf 'capo\tcql\thopper\tmedium-replay\tdone\t%s\t2026-08-03 04:27:09\t2026-08-03 06:01:17\n' \
    'results/cql/hopper-medium-replay-v2/s0/0803_0427_capo_cql_hopper-medium-replay-v2_s0'
  printf 'capo\tcql\thalfcheetah\tmedium\tdone\t%s\t2026-08-03 06:01:17\t2026-08-03 07:48:49\n' \
    'results/cql/halfcheetah-medium-v2/s0/0803_0601_capo_cql_halfcheetah-medium-v2_s0'
  # expert cells are NOT in the new DATASETS; record for provenance only
  printf 'capo\tcql\thopper\texpert\tcancelled\t%s\t2026-08-03 02:44:05\t2026-08-03 04:27:09\n' \
    'results/cql/hopper-expert-v2/s0/0803_0244_capo_cql_hopper-expert-v2_s0'
  if [[ -n "${EXPERT_DIR:-}" ]]; then
    printf 'capo\tcql\thalfcheetah\texpert\tcancelled\t%s\t%s\t%s\n' \
      "$EXPERT_DIR" "$STARTED" "$FINISHED"
  fi
} > "$STATUS"
log "rewrote $STATUS"
cat "$STATUS" | tee -a "$LOG"

# Relaunch matrix with medium-expert schedule (skips done medium/replay)
nohup bash scripts/run_matrix_cql.sh >> "$OUT_DIR/queue_cql_master.log" 2>&1 &
echo $! > "$OUT_DIR/queue_cql.pid"
log "relaunched matrix pid=$(cat "$OUT_DIR/queue_cql.pid")"

# Wait briefly for first new cell
for i in $(seq 1 30); do
  if pgrep -f '[s]cripts/run_capo.py' >/dev/null; then
    break
  fi
  sleep 2
done
pgrep -af '[s]cripts/run_matrix_cql|[s]cripts/run_capo.py' | tee -a "$LOG" || true
log "done"
