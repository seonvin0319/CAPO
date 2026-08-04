#!/usr/bin/env bash
# Sequentially post-hoc distill completed JAX sweep runs, then refresh overview/plots.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
QUEUE="${1:-results_jax_sweeps/distill_queue.txt}"
LOG_DIR=results_jax_sweeps/parallel_logs
mkdir -p "$LOG_DIR"
PY=/home/ext_csh/miniconda3/envs/capo_jax/bin/python
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"

if [[ ! -f "$QUEUE" ]]; then
  echo "[distill-queue] missing $QUEUE" >&2
  exit 1
fi

gpu="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES="$gpu"
echo "[distill-queue] GPU=$CUDA_VISIBLE_DEVICES queue=$QUEUE"

while IFS= read -r rd || [[ -n "$rd" ]]; do
  [[ -z "$rd" || "$rd" =~ ^# ]] && continue
  name="$(basename "$rd")"
  echo "[distill-queue] START $name $(date -Is)"
  if ! "$PY" scripts/run_posthoc_student_distill_jax.py \
      --checkpoint_dir "$rd" \
      --out_dir "$rd/posthoc_student_distill_jax" \
      --start_step 100000 \
      --end_step 1000000 \
      --checkpoint_interval 50000 \
      --distill_steps_per_checkpoint 1000 \
      --n_episodes 10 \
      --device cuda; then
    echo "[distill-queue] FAIL $name" >&2
    continue
  fi
  "$PY" scripts/plot_training_curve.py "$rd" || true
  echo "[distill-queue] DONE $name $(date -Is)"
done < "$QUEUE"

"$PY" scripts/summarize_jax_sweeps.py --root results_jax_sweeps --no_plot || true
echo "[distill-queue] all finished $(date -Is)"
