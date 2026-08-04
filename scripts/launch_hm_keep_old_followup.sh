#!/usr/bin/env bash
set -uo pipefail
cd /home/ext_csh/CaPO
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
# Prefer prebuilt mujoco_py from capo env (same overlay as run_capo_stability_sweep).
export PYTHONPATH="/home/ext_csh/CaPO/results_jax_sweeps/.runtime_overlay${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${HOME}/miniconda3/envs/capo_jax/lib:${HOME}/miniconda3/envs/capo/lib:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
unset MUJOCO_PY_FORCE_CPU
exec /home/ext_csh/miniconda3/envs/capo_jax/bin/python scripts/run_capo_jax.py \
  --config results_jax_sweeps/capo_stability_seed0_fast/fast_hm_s0_lt0_p50k_m1e3_keep_old/resolved_config.yaml
