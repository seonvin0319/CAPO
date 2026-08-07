#!/usr/bin/env bash
# Wait for antmaze n8 to finish, then run n16. Does not touch a live n8 runner.
set -uo pipefail

ROOT=/home/ext_csh/CaPO
PY=/home/ext_csh/miniconda3/envs/capo_jax/bin/python
export MUJOCO_GL=egl
export MUJOCO_PY_MUJOCO_PATH=/home/ext_csh/.mujoco/mujoco210
export LD_LIBRARY_PATH=/home/ext_csh/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}

LOG="$ROOT/results_jax_sweeps/capo_antmaze_chain_n8_n16.log"
mkdir -p "$(dirname "$LOG")"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

count_done() {
  local manifest="$1" results_root="$2"
  "$PY" - <<PY
import json
from pathlib import Path
root=Path("$results_root")
rows=[json.loads(l) for l in Path("$manifest").read_text().splitlines() if l.strip()]
done=sum(1 for r in rows if (root/r["run_id"]/"final.pkl").exists() or (root/r["run_id"]/"checkpoint_1000000.pkl").exists())
print(f"{done}/{len(rows)}")
PY
}

wait_manifest() {
  local name="$1" manifest="$2" results_root="$3"
  while true; do
    local prog done total
    prog=$(count_done "$manifest" "$results_root")
    done="${prog%%/*}"; total="${prog##*/}"
    log "$name progress $prog"
    if [[ "$done" == "$total" ]]; then
      log "$name COMPLETE"
      return 0
    fi
    # if n8 runner died early, relaunch with resume (adopt/incomplete)
    if [[ "$name" == "antmaze_n8" ]]; then
      if ! pgrep -f "run_capo_stability_sweep.py.*antmaze_n8" >/dev/null; then
        log "n8 runner missing — relaunch --resume"
        tmux has-session -t capo_fast_antmaze_n8 2>/dev/null && tmux kill-session -t capo_fast_antmaze_n8 2>/dev/null || true
        tmux new-session -d -s capo_fast_antmaze_n8 \
          "$PY $ROOT/scripts/run_capo_stability_sweep.py \
            --manifest $manifest \
            --gpus 0,1 --jobs_per_gpu 6 --max_concurrent_jobs 12 \
            --resume --poll_sec 10 \
            >> $ROOT/results_jax_sweeps/capo_fast_antmaze_n8_runner.log 2>&1"
      fi
    fi
    sleep 120
  done
}

log "=== chain start: wait n8 → run n16 (n2 skipped) ==="

N8_MAN="$ROOT/manifests/capo_stability_seed0_fast_antmaze_n8.jsonl"
N8_ROOT="$ROOT/results_jax_sweeps/capo_stability_seed0_fast_antmaze_n8"
N16_MAN="$ROOT/manifests/capo_stability_seed0_fast_antmaze_n16.jsonl"
N16_ROOT="$ROOT/results_jax_sweeps/capo_stability_seed0_fast_antmaze_n16"

wait_manifest "antmaze_n8" "$N8_MAN" "$N8_ROOT"

log "LAUNCH antmaze_n16"
tmux has-session -t capo_fast_antmaze_n16 2>/dev/null && tmux kill-session -t capo_fast_antmaze_n16 2>/dev/null || true
tmux new-session -d -s capo_fast_antmaze_n16 \
  "$PY $ROOT/scripts/run_capo_stability_sweep.py \
    --manifest $N16_MAN \
    --gpus 0,1 --jobs_per_gpu 6 --max_concurrent_jobs 12 \
    --resume --poll_sec 10 \
    >> $ROOT/results_jax_sweeps/capo_fast_antmaze_n16_runner.log 2>&1"
sleep 5
if ! pgrep -f "run_capo_stability_sweep.py.*antmaze_n16" >/dev/null; then
  log "ERROR failed to start n16 runner"
  exit 1
fi
log "runner up for antmaze_n16"
wait_manifest "antmaze_n16" "$N16_MAN" "$N16_ROOT"
log "=== chain ALL COMPLETE (n8→n16) ==="
