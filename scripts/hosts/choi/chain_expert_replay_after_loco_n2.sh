#!/usr/bin/env bash
# Wait for locomotion capo_stability_seed0_fast_n2 replace_new queue to finish,
# then launch expert + full-replay replace_new mgrid (72 cells) with same settings.
# Replaces the previous antmaze chain on this host.
set -euo pipefail
ROOT=/home/choi/CAPO
PY=/home/choi/miniconda3/envs/offrl_backup/bin/python
LOCO_ROOT="$ROOT/results_jax_sweeps/capo_stability_seed0_fast_n2"
LOCO_MANIFEST="$ROOT/manifests/capo_stability_seed0_fast_n2_replace_new_mgrid.jsonl"
NEXT_ROOT="$ROOT/results_jax_sweeps/capo_stability_seed0_fast_n2_expert_replay"
NEXT_MANIFEST="$ROOT/manifests/capo_stability_seed0_fast_n2_expert_replay_replace_new_mgrid.jsonl"
LOG="$NEXT_ROOT/chain_waiter.log"
mkdir -p "$NEXT_ROOT"
cd "$ROOT"

export MUJOCO_PY_MUJOCO_PATH=/home/choi/.mujoco/mujoco210
export LD_LIBRARY_PATH=/home/choi/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}
export D4RL_SUPPRESS_IMPORT_ERROR=1

echo "[$(date -Is)] expert/full-replay chain waiter started" | tee -a "$LOG"

loco_complete() {
  "$PY" - <<'PY'
import json
from pathlib import Path
root = Path("/home/choi/CAPO/results_jax_sweeps/capo_stability_seed0_fast_n2")
mani = Path("/home/choi/CAPO/manifests/capo_stability_seed0_fast_n2_replace_new_mgrid.jsonl")
ids = [json.loads(l)["run_id"] for l in mani.read_text().splitlines() if l.strip()]
done = sum(1 for rid in ids if (root / rid / "summary.json").exists())
print(f"{done}/{len(ids)}")
raise SystemExit(0 if done >= len(ids) else 1)
PY
}

loco_trainers_live() {
  "$PY" - <<'PY'
from pathlib import Path
live = 0
for ent in Path("/proc").iterdir():
    if not ent.name.isdigit():
        continue
    try:
        cmd = (ent / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
    except Exception:
        continue
    if "run_capo_jax.py" in cmd and "capo_stability_seed0_fast_n2/" in cmd and "_antmaze" not in cmd and "_expert_replay" not in cmd:
        live += 1
print(live)
raise SystemExit(0 if live == 0 else 1)
PY
}

while true; do
  if loco_complete >>"$LOG" 2>&1; then
    if loco_trainers_live >>"$LOG" 2>&1; then
      echo "[$(date -Is)] locomotion complete; launching expert/full-replay" | tee -a "$LOG"
      break
    fi
    echo "[$(date -Is)] summaries done but trainers still draining; wait" | tee -a "$LOG"
  else
    echo "[$(date -Is)] loco progress: $(loco_complete 2>/dev/null || true) waiting..." | tee -a "$LOG"
  fi
  sleep 300
done

MASTER=$(cat "$LOCO_ROOT/queue.pid" 2>/dev/null || true)
if [[ -n "${MASTER}" ]] && kill -0 "$MASTER" 2>/dev/null; then
  echo "[$(date -Is)] loco master $MASTER still up; wait for exit" | tee -a "$LOG"
  for _ in $(seq 1 60); do
    kill -0 "$MASTER" 2>/dev/null || break
    sleep 10
  done
fi

nohup "$PY" scripts/run_capo_stability_sweep.py \
  --manifest "$NEXT_MANIFEST" \
  --gpus 0 --jobs_per_gpu 2 \
  --override n_critics=2 \
  --override sweep_name=capo_stability_seed0_fast_n2_expert_replay \
  --override save_best=false \
  --override save_refresh_actors=false \
  --python "$PY" \
  --mujoco_py_source /tmp/no_such_mujoco_py_for_choi \
  --resume \
  >> "$NEXT_ROOT/queue.log" 2>&1 &
echo $! > "$NEXT_ROOT/queue.pid"
echo "[$(date -Is)] expert_replay master=$(cat "$NEXT_ROOT/queue.pid")" | tee -a "$LOG"
