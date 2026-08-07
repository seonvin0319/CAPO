#!/usr/bin/env bash
# Wait for antmaze n4 (42 cells) to finish, then run n8 then n2.
# Does not touch the live n4 runner/trainers.
set -uo pipefail

ROOT=/home/ext_csh/CaPO
PY=/home/ext_csh/miniconda3/envs/capo_jax/bin/python
export MUJOCO_GL=egl
export MUJOCO_PY_MUJOCO_PATH=/home/ext_csh/.mujoco/mujoco210
export LD_LIBRARY_PATH=/home/ext_csh/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}

LOG="$ROOT/results_jax_sweeps/capo_antmaze_chain_n4_n8_n2.log"
mkdir -p "$(dirname "$LOG")"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

count_done() {
  local manifest="$1" results_root="$2"
  "$PY" - <<PY
import json
from pathlib import Path
root=Path("$results_root")
rows=[json.loads(l) for l in Path("$manifest").read_text().splitlines() if l.strip()]
done=0
for r in rows:
    d=root/r["run_id"]
    if (d/"final.pkl").exists() or (d/"checkpoint_1000000.pkl").exists():
        done+=1
print(f"{done}/{len(rows)}")
PY
}

wait_manifest() {
  local name="$1" manifest="$2" results_root="$3"
  while true; do
    local prog
    prog=$(count_done "$manifest" "$results_root")
    local done="${prog%%/*}" total="${prog##*/}"
    log "$name progress $prog"
    if [[ "$done" == "$total" ]]; then
      log "$name COMPLETE"
      return 0
    fi
    # also require no live trainers for this sweep prefix if complete count stuck? no — wait for files
    sleep 120
  done
}

run_sweep() {
  local name="$1" manifest="$2" runner_log="$3" session="$4"
  log "LAUNCH $name session=$session"
  tmux has-session -t "$session" 2>/dev/null && tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" \
    "$PY $ROOT/scripts/run_capo_stability_sweep.py \
      --manifest $manifest \
      --gpus 0,1 --jobs_per_gpu 6 --max_concurrent_jobs 12 \
      --resume --poll_sec 10 \
      >> $runner_log 2>&1"
  sleep 5
  if ! pgrep -f "run_capo_stability_sweep.py.*$(basename "$manifest")" >/dev/null; then
    log "ERROR failed to start runner for $name"
    tail -n 40 "$runner_log" | tee -a "$LOG" || true
    return 1
  fi
  log "runner up for $name"
}

log "=== chain start: wait n4 → n8 → n16 (n2 skipped) ==="

N4_MAN="$ROOT/manifests/capo_stability_seed0_fast_antmaze.jsonl"
N4_ROOT="$ROOT/results_jax_sweeps/capo_stability_seed0_fast_antmaze"
N8_MAN="$ROOT/manifests/capo_stability_seed0_fast_antmaze_n8.jsonl"
N8_ROOT="$ROOT/results_jax_sweeps/capo_stability_seed0_fast_antmaze_n8"
N2_MAN="$ROOT/manifests/capo_stability_seed0_fast_antmaze_n16.jsonl"
N2_ROOT="$ROOT/results_jax_sweeps/capo_stability_seed0_fast_antmaze_n16"

wait_manifest "antmaze_n4" "$N4_MAN" "$N4_ROOT"

# Ensure n4 runner has exited (optional); do not kill live trainers mid-flight.
# If somehow incomplete dirs remain without final, wait_manifest already gated on finals.

run_sweep "antmaze_n8" "$N8_MAN" \
  "$ROOT/results_jax_sweeps/capo_fast_antmaze_n8_runner.log" \
  "capo_fast_antmaze_n8"
wait_manifest "antmaze_n8" "$N8_MAN" "$N8_ROOT"

run_sweep "antmaze_n16" "$N2_MAN" \
  "$ROOT/results_jax_sweeps/capo_fast_antmaze_n16_runner.log" \
  "capo_fast_antmaze_n16"
wait_manifest "antmaze_n16" "$N2_MAN" "$N2_ROOT"

log "=== chain ALL COMPLETE (n4→n8→n2) ==="
