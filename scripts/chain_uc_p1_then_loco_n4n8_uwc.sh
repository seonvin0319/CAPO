#!/usr/bin/env bash
# After antmaze UC Phase1 completes, launch loco n4_uwc + n8_uwc (user recipe).
set -uo pipefail
ROOT=/home/ext_csh/CaPO
PY=/home/ext_csh/miniconda3/envs/capo_jax/bin/python
export MUJOCO_GL=egl
export MUJOCO_PY_MUJOCO_PATH=/home/ext_csh/.mujoco/mujoco210
export LD_LIBRARY_PATH=/home/ext_csh/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}
LOG="$ROOT/results_jax_sweeps/capo_uc_p1_then_loco_uwc_chain.log"
mkdir -p "$(dirname "$LOG")"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

P1_MAN="$ROOT/manifests/capo_antmaze_uncertainty_critic_seed0.jsonl"
P1_ROOT="$ROOT/results_jax_sweeps/capo_antmaze_uncertainty_critic_seed0"

count_done() {
  local manifest="$1" results_root="$2"
  "$PY" - <<PY
import json
from pathlib import Path
root=Path("$results_root")
rows=[json.loads(l) for l in Path("$manifest").read_text().splitlines() if l.strip()]
done=sum(1 for r in rows if (root/r["run_id"]/"summary.json").exists() or (root/r["run_id"]/"final.pkl").exists() or (root/r["run_id"]/"checkpoint_1000000.pkl").exists())
print(f"{done}/{len(rows)}")
PY
}

log "=== wait UC Phase1 then launch loco n4/n8 UWC ==="
while true; do
  prog=$(count_done "$P1_MAN" "$P1_ROOT")
  done="${prog%%/*}"; total="${prog##*/}"
  log "uc_phase1 progress $prog"
  if [[ "$done" == "$total" ]]; then
    log "uc_phase1 COMPLETE"
    break
  fi
  # heal Phase1 runner if dead
  if ! pgrep -f 'run_capo_stability_sweep.py.*capo_antmaze_uncertainty_critic_seed0' >/dev/null; then
    log "uc_phase1 runner missing — resume relaunch"
    # quarantine NOCKPT
    "$PY" - <<'PY'
import json
from pathlib import Path
from datetime import datetime,timezone
man=Path("/home/ext_csh/CaPO/manifests/capo_antmaze_uncertainty_critic_seed0.jsonl")
root=Path("/home/ext_csh/CaPO/results_jax_sweeps/capo_antmaze_uncertainty_critic_seed0")
ts=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
for r in [json.loads(l) for l in man.read_text().splitlines() if l.strip()]:
    d=root/r["run_id"]
    if (d/"summary.json").exists() or (d/"final.pkl").exists() or (d/"checkpoint_1000000.pkl").exists():
        continue
    has=(d/"latest.pkl").exists() or bool(list(d.glob("checkpoint_*.pkl")))
    if d.exists() and any(d.iterdir()) and not has:
        d.rename(d.with_name(d.name+f".QUARANTINE_nocckpt_{ts}"))
        print("quarantined", d.name)
PY
    tmux has-session -t capo_uc_critic_p1 2>/dev/null && tmux kill-session -t capo_uc_critic_p1 2>/dev/null || true
    tmux new-session -d -s capo_uc_critic_p1 \
      "$PY $ROOT/scripts/run_capo_stability_sweep.py \
        --manifest $P1_MAN \
        --gpus 0,1 --jobs_per_gpu 6 --max_concurrent_jobs 12 \
        --resume --poll_sec 10 \
        >> $ROOT/results_jax_sweeps/capo_uc_critic_p1_runner.log 2>&1"
  fi
  sleep 120
done

# --- loco UWC (user recipe): both on GPU0, 2 jobs each ---
launch_uwc() {
  local name="$1" man="$2" sweep="$3" ncrit="$4"
  local out="$ROOT/results_jax_sweeps/$sweep"
  mkdir -p "$out"
  if pgrep -f "run_capo_stability_sweep.py.*$(basename "$man")" >/dev/null; then
    log "$name already live — leave it"
    return 0
  fi
  log "LAUNCH $name n_critics=$ncrit"
  nohup "$PY" "$ROOT/scripts/run_capo_stability_sweep.py" \
    --manifest "$man" \
    --gpus 0 --jobs_per_gpu 2 \
    --override "n_critics=$ncrit" \
    --override use_uncertainty_weighted_critic=true \
    --override actor_type=deterministic \
    --override distance_metric=amse \
    --override "sweep_name=$sweep" \
    --override save_best=false \
    --override save_refresh_actors=false \
    --python "$PY" \
    --mujoco_py_source /home/ext_csh/miniconda3/envs/capo/lib/python3.10/site-packages/mujoco_py \
    --resume \
    >> "$out/queue.log" 2>&1 &
  echo $! > "$out/queue.pid"
  sleep 3
  if ! pgrep -f "run_capo_stability_sweep.py.*$(basename "$man")" >/dev/null; then
    log "ERROR failed to start $name"; return 1
  fi
  log "runner up for $name pid=$(cat "$out/queue.pid")"
}

launch_uwc n4_uwc \
  "$ROOT/manifests/capo_stability_seed0_fast_n4_uwc_replace_new_mgrid.jsonl" \
  capo_stability_seed0_fast_n4_uwc 4

launch_uwc n8_uwc \
  "$ROOT/manifests/capo_stability_seed0_fast_n8_uwc_replace_new_mgrid.jsonl" \
  capo_stability_seed0_fast_n8_uwc 8

log "both UWC queues launched (GPU0 ×2 each). Chain done."
