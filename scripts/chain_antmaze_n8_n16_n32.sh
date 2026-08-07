#!/usr/bin/env bash
# n8 → n16 → n32. Does not kill a live n8 runner on startup.
set -uo pipefail
ROOT=/home/ext_csh/CaPO
PY=/home/ext_csh/miniconda3/envs/capo_jax/bin/python
export MUJOCO_GL=egl MUJOCO_PY_MUJOCO_PATH=/home/ext_csh/.mujoco/mujoco210
export LD_LIBRARY_PATH=/home/ext_csh/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}
LOG="$ROOT/results_jax_sweeps/capo_antmaze_chain_n8_n16_n32.log"
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

launch_sweep() {
  local name="$1" manifest="$2" session="$3" runner_log="$4"
  if pgrep -f "run_capo_stability_sweep.py.*$(basename "$manifest")" >/dev/null; then
    log "$name runner already live — leave it"
    return 0
  fi
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
    log "ERROR failed to start $name"; return 1
  fi
  log "runner up for $name"
}

wait_manifest() {
  local name="$1" manifest="$2" results_root="$3" session="$4" runner_log="$5"
  while true; do
    local prog done total
    prog=$(count_done "$manifest" "$results_root")
    done="${prog%%/*}"; total="${prog##*/}"
    log "$name progress $prog"
    if [[ "$done" == "$total" ]]; then
      log "$name COMPLETE"; return 0
    fi
    # heal dead runner without killing live one
    if ! pgrep -f "run_capo_stability_sweep.py.*$(basename "$manifest")" >/dev/null; then
      log "$name runner missing — resume relaunch"
      launch_sweep "$name" "$manifest" "$session" "$runner_log" || true
    fi
    sleep 120
  done
}

log "=== chain start: n8 → n16 → n32 ==="
N8_MAN="$ROOT/manifests/capo_stability_seed0_fast_antmaze_n8.jsonl"
N8_ROOT="$ROOT/results_jax_sweeps/capo_stability_seed0_fast_antmaze_n8"
N16_MAN="$ROOT/manifests/capo_stability_seed0_fast_antmaze_n16.jsonl"
N16_ROOT="$ROOT/results_jax_sweeps/capo_stability_seed0_fast_antmaze_n16"
N32_MAN="$ROOT/manifests/capo_stability_seed0_fast_antmaze_n32.jsonl"
N32_ROOT="$ROOT/results_jax_sweeps/capo_stability_seed0_fast_antmaze_n32"

# Ensure n8 running if incomplete, then wait
launch_sweep antmaze_n8 "$N8_MAN" capo_fast_antmaze_n8 \
  "$ROOT/results_jax_sweeps/capo_fast_antmaze_n8_runner.log"
wait_manifest antmaze_n8 "$N8_MAN" "$N8_ROOT" capo_fast_antmaze_n8 \
  "$ROOT/results_jax_sweeps/capo_fast_antmaze_n8_runner.log"

launch_sweep antmaze_n16 "$N16_MAN" capo_fast_antmaze_n16 \
  "$ROOT/results_jax_sweeps/capo_fast_antmaze_n16_runner.log"
wait_manifest antmaze_n16 "$N16_MAN" "$N16_ROOT" capo_fast_antmaze_n16 \
  "$ROOT/results_jax_sweeps/capo_fast_antmaze_n16_runner.log"

launch_sweep antmaze_n32 "$N32_MAN" capo_fast_antmaze_n32 \
  "$ROOT/results_jax_sweeps/capo_fast_antmaze_n32_runner.log"
wait_manifest antmaze_n32 "$N32_MAN" "$N32_ROOT" capo_fast_antmaze_n32 \
  "$ROOT/results_jax_sweeps/capo_fast_antmaze_n32_runner.log"

log "=== chain ALL COMPLETE (n8→n16→n32) ==="
