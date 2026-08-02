#!/usr/bin/env bash
# 1) Stop IQL matrix master now (leave the live run_capo cell alone).
# 2) When that cell finishes, rewrite status and relaunch with
#    DATASETS=(medium medium-expert replay).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG="${LOG:-results/queue_switch_medium_expert.log}"
POLL_SEC="${POLL_SEC:-10}"
MATRIX_PID_FILE="${MATRIX_PID_FILE:-results/queue_iql.pid}"
PYTHON_HINT="${PYTHON_HINT:-/home/offrl/miniconda3/envs/offrl/bin/python}"

{
  echo "[$(date '+%F %T')] switcher start"

  # Detach live trainer from the old matrix so no further expert cells launch.
  if [[ -f "$MATRIX_PID_FILE" ]]; then
    mp="$(cat "$MATRIX_PID_FILE" || true)"
    if [[ -n "${mp}" ]] && kill -0 "$mp" 2>/dev/null; then
      echo "[$(date '+%F %T')] killing IQL matrix master pid=$mp (trainer kept)"
      kill "$mp" 2>/dev/null || true
      sleep 2
      kill -9 "$mp" 2>/dev/null || true
    fi
  fi
  pkill -f 'bash scripts/run_matrix_iql.sh' 2>/dev/null || true
  sleep 1
  if pgrep -f 'bash scripts/run_matrix_iql.sh' >/dev/null 2>&1; then
    pkill -9 -f 'bash scripts/run_matrix_iql.sh' 2>/dev/null || true
  fi
  echo "[$(date '+%F %T')] matrix masters cleared; live run_capo (if any) continues"

  echo "[$(date '+%F %T')] waiting for live run_capo to finish…"
  while pgrep -f "${PYTHON_HINT} scripts/run_capo.py" >/dev/null 2>&1; do
    echo "[$(date '+%F %T')] run_capo still alive; sleep ${POLL_SEC}s"
    sleep "$POLL_SEC"
  done
  echo "[$(date '+%F %T')] run_capo exited"

  python3 - <<'PY'
from pathlib import Path
from datetime import datetime

status = Path("results/queue_status_iql.tsv")
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
lines = status.read_text().splitlines()
header, rows = lines[0], lines[1:]
out = [header]
seen = set()

for line in rows:
    p = line.split("\t")
    while len(p) < 8:
        p.append("")
    key = (p[0], p[2], p[3])
    seen.add(key)
    st = p[4]
    if st == "done":
        out.append("\t".join(p))
        continue
    # Finish baseline hopper expert if summary exists.
    if p[0] == "baseline" and p[2] == "hopper" and p[3] == "expert":
        base = Path("results/iql/hopper-expert-v2/s0")
        cands = sorted(base.glob("*baseline_iql_hopper-expert*"), reverse=True) if base.is_dir() else []
        if cands and (cands[0] / "summary.json").exists():
            p[4] = "done"
            p[5] = str(cands[0])
            if not p[7]:
                p[7] = now
            out.append("\t".join(p))
            continue
        p[4] = "cancelled"
        p[7] = now
        out.append("\t".join(p))
        continue
    p[4] = "cancelled"
    if p[5] == "pending":
        p[5] = ""
    p[7] = now
    out.append("\t".join(p))

# Explicitly cancel remaining baseline expert cells that never started.
for env in ("halfcheetah", "walker2d"):
    key = ("baseline", env, "expert")
    if key not in seen:
        out.append("\t".join(["baseline", "iql", env, "expert", "cancelled", "", "", now]))

status.write_text("\n".join(out) + "\n")
print(status.read_text())
PY

  echo "[$(date '+%F %T')] relaunching run_matrix_iql.sh (medium / medium-expert / replay)"
  # append to master log with a banner
  {
    echo ""
    echo "===== relaunch medium-expert $(date '+%F %T') ====="
  } >> results/queue_iql_master.log
  nohup bash scripts/run_matrix_iql.sh >> results/queue_iql_master.log 2>&1 &
  echo $! > results/queue_iql.pid
  sleep 1
  real="$(pgrep -f 'bash scripts/run_matrix_iql.sh' | head -1 || true)"
  if [[ -n "$real" ]]; then
    echo "$real" > results/queue_iql.pid
  fi
  echo "[$(date '+%F %T')] new matrix pid=$(cat results/queue_iql.pid)"

  # Refresh v8 waiter to follow the new matrix pid
  pkill -f 'queue_v8_hold_iql_after_iql.sh' 2>/dev/null || true
  sleep 1
  nohup bash scripts/queue_v8_hold_iql_after_iql.sh >> results/queue_v8_hold_iql_waiter.log 2>&1 &
  echo $! > results/queue_v8_hold_iql_waiter.pid
  sleep 1
  wr="$(pgrep -f 'queue_v8_hold_iql_after_iql.sh' | head -1 || true)"
  if [[ -n "$wr" ]]; then
    echo "$wr" > results/queue_v8_hold_iql_waiter.pid
  fi
  echo "[$(date '+%F %T')] v8 waiter pid=$(cat results/queue_v8_hold_iql_waiter.pid)"
  echo "[$(date '+%F %T')] switcher done"
} >>"$LOG" 2>&1
