#!/usr/bin/env bash
# Skip remaining lambda_T=0 cells and relaunch sweep on λT∈{0.5,1.0}.
set -euo pipefail
ROOT=/home/choi/CAPO
SWEEP_ROOT="$ROOT/results_jax_sweeps/capo_stability_seed0_fast_n2"
PY=/home/choi/miniconda3/envs/offrl_backup/bin/python
MANIFEST="$ROOT/manifests/capo_stability_seed0_fast_lt05_lt1.jsonl"
cd "$ROOT"

if [[ ! -f "$MANIFEST" ]]; then
  echo "missing filtered manifest: $MANIFEST" >&2
  exit 1
fi

MASTER=$(cat "$SWEEP_ROOT/queue.pid" 2>/dev/null || true)
if [[ -n "${MASTER}" ]] && kill -0 "$MASTER" 2>/dev/null; then
  echo "kill master $MASTER"
  kill "$MASTER" || true
  sleep 2
  kill -0 "$MASTER" 2>/dev/null && kill -9 "$MASTER" || true
fi

mapfile -t TRAIN_PIDS < <(pgrep -f 'run_capo_jax.py.*capo_stability_seed0_fast_n2/fast_.*_lt0_' || true)
for pid in "${TRAIN_PIDS[@]:-}"; do
  [[ -z "$pid" ]] && continue
  echo "kill trainer $pid"
  kill "$pid" || true
done
sleep 2
mapfile -t TRAIN_PIDS < <(pgrep -f 'run_capo_jax.py.*capo_stability_seed0_fast_n2/fast_.*_lt0_' || true)
for pid in "${TRAIN_PIDS[@]:-}"; do
  [[ -z "$pid" ]] && continue
  echo "force kill $pid"
  kill -9 "$pid" || true
done

"$PY" - <<'PY'
import json, time
from pathlib import Path
ROOT = Path("/home/choi/CAPO/results_jax_sweeps/capo_stability_seed0_fast_n2")
for d in ROOT.glob("fast_*_lt0_*"):
    if (d / "summary.json").exists():
        continue
    if not (d / "latest.pkl").exists() and not (d / "heartbeat.json").exists():
        continue
    skip = {
        "run_id": d.name,
        "status": "skipped",
        "reason": "user requested skip remaining lambda_T=0; proceed to lambda_T in {0.5,1.0}",
        "skipped_unix": time.time(),
    }
    (d / "skip.json").write_text(json.dumps(skip, indent=2))
    print("marked skip", d.name)
PY

cat > "$SWEEP_ROOT/SKIP_LT0.txt" <<'EOF'
2026-08-04: Remaining lambda_T=0 cells skipped by operator request.
Completed lt0 hopper-medium p50k cells kept.
Queue restarted on manifests/capo_stability_seed0_fast_lt05_lt1.jsonl (216 runs: lt 0.5 + 1.0).
EOF

export MUJOCO_PY_MUJOCO_PATH=/home/choi/.mujoco/mujoco210
export LD_LIBRARY_PATH=/home/choi/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}
nohup "$PY" scripts/run_capo_stability_sweep.py \
  --manifest manifests/capo_stability_seed0_fast_lt05_lt1.jsonl \
  --gpus 0 --jobs_per_gpu 2 \
  --override n_critics=2 \
  --override sweep_name=capo_stability_seed0_fast_n2 \
  --python "$PY" \
  --mujoco_py_source /tmp/no_such_mujoco_py_for_choi \
  --resume \
  >> "$SWEEP_ROOT/queue.log" 2>&1 &
echo $! > "$SWEEP_ROOT/queue.pid"
sleep 5
NEW=$(cat "$SWEEP_ROOT/queue.pid")
echo "new_master=$NEW alive=$(kill -0 "$NEW" 2>/dev/null && echo yes || echo no)"
pgrep -af 'run_capo_jax.py|run_capo_stability_sweep' | grep -v pgrep | head -10
tail -n 35 "$SWEEP_ROOT/queue.log"
