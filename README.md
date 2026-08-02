# CAPO

**Calibrated Adaptive Policy Optimization** for offline reinforcement learning.

Teacher-guided offline RL on [D4RL](https://github.com/Farama-Foundation/D4RL) with TD3+BC, IQL, and CQL base learners.

Repository: [https://github.com/seonvin0319/CAPO](https://github.com/seonvin0319/CAPO)

---

## What it does

CAPO trains a **student actor** θL with standard offline RL critics while periodically running **CAPO** on a held-out batch: a certificate-driven procedure picks a proximal step size τ (via **pilot-adaptive** search), runs capped multi-step refinement, and optionally installs a **teacher** θR. The student actor is updated with a TD3-style Q term plus dataset BC and **teacher BC** when a teacher is active. Critics always bootstrap from θL only, so teacher refinement does not create a critic feedback loop.

Typical use: locomotion benchmarks (`hopper`, `halfcheetah`, `walker2d` × `medium` / `expert` / `replay`) and smoke tests on a single GPU.

---

## Requirements

**Recommended:** Conda environment `offrl_backup` (PyTorch + CUDA, MuJoCo, D4RL already installed on the authors' machines).

Otherwise install Python dependencies and D4RL manually:

```bash
conda create -n capo python=3.10  # example
conda activate capo
pip install -r requirements.txt
# PyTorch: install a CUDA build matching your driver from https://pytorch.org
pip install git+https://github.com/Farama-Foundation/D4RL.git
```

`requirements.txt` pins `gym==0.23.0`, `numpy<2`, `pyyaml`, `tqdm`, `pyrallis`. **torch** and **d4rl** are expected but not always listed as pip wheels.

Environment variables used by the scripts:

```bash
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl   # or osmesa / glfw as appropriate
```

---

## Project layout

| Path | Description |
|------|-------------|
| `capo/` | Core library: `core.py` (CAPO certificates), `trainer.py` (D4RL loop), `refiner.py`, `networks.py`, `buffer.py`, `tabular.py` |
| `configs/defaults.yaml` | Canonical 1M-step defaults (v8 teacher + replace gate) |
| `configs/baseline_td3bc.yaml` | Matched TD3+BC control (`use_capo: false`, `n_critics: 4`) |
| `configs/smoke.yaml` | Short hopper smoke run (~2k steps) |
| `host/` | Shared per-host run board (`choi.csv`, …); see [`host/README.md`](host/README.md) |
| `scripts/run_capo.py` | Main D4RL training CLI |
| `scripts/run_tabular.py` | Tabular certificate demo (no MuJoCo) |
| `scripts/run_smoke.sh` | Tabular + smoke D4RL |
| `configs/v8_hold.yaml` | Successful diag_v8 hold hyperparams (`λ_D=0.2`, `λ_T=1.0`, period 50k) |
| `scripts/run_matrix.sh` | Sequential CAPO vs baseline matrix (2 × 9 cells, seed 0) |
| `scripts/run_matrix_v8_hold.sh` | 9-cell v8-hold matrix (`run_tag=v8_hold`) |
| `scripts/queue_v8_hold_after_current.sh` | Wait for live matrix, then start v8_hold |
| `scripts/analyze_*.py`, `summarize_matrix.py` | Post-hoc analysis helpers |
| `tests/` | Unit tests (`test_pilot_adaptive.py`) |

`configs/matrix_defaults.yaml` is a **compatibility alias** for an older queue path; prefer `defaults.yaml` for new runs.

---

## Quick start

From the repository root:

```bash
cd /path/to/CAPO
conda activate offrl_backup   # or your env with torch + d4rl

# 1) Tabular CAPO demo (fast, no D4RL env rollouts)
python scripts/run_tabular.py
python scripts/run_tabular.py --seed 0

# 2) Full smoke: tabular + 2k-step hopper-medium-v2
bash scripts/run_smoke.sh

# 3) Single D4RL run (override algo / env on CLI)
python scripts/run_capo.py --config configs/defaults.yaml \
  --algorithm td3_bc --env_base hopper --dataset medium

# Explicit env id
python scripts/run_capo.py --config configs/smoke.yaml --algorithm td3_bc --env hopper-medium-v2
```

**18-run CAPO vs baseline matrix** (td3_bc × 9 envs × {capo, baseline}, sequential, resumable):

```bash
bash scripts/run_matrix.sh
# or background:
nohup bash scripts/run_matrix.sh > results/queue_master.log 2>&1 &
echo $! > results/queue.pid
tail -f results/queue_status.tsv
```

Optional env overrides: `SEED`, `DEVICE`, `OUT_DIR`, `PYTHON`.

**Multi-host coordination:** claim and publish status under [`host/`](host/README.md) (`host/choi.csv`, …). Local `results/queue_status.tsv` is not shared via git.

---

## Result directory structure

Runs write under `out_dir` (default `results/`):

```text
results/<env_id>/s<seed>/<MMDD_HHMM>_<run_tag>_<algo>_<env_id>_s<seed>/
```

Example:

```text
results/hopper-medium-v2/s0/0803_0012_capo_td3_bc_hopper-medium-v2_s0/
```

Typical artifacts (not committed to git):

- `train.log`, `config.json`, `metrics.jsonl`, `summary.json`
- `capo_refresh.jsonl`, `capo_ladder.jsonl` (CAPO diagnostics)
- `best.pt`, `final.pt`, `checkpoint_<steps>.pt`

Matrix queue tracking (local only):

- `results/queue_status.tsv` — per-cell status and run directory
- `results/queue_master.log` — matrix stdout/stderr when using `nohup`

---

## Key hyperparameters (`configs/defaults.yaml`)

| Group | Keys | Role |
|-------|------|------|
| Training | `max_timesteps`, `batch_size`, `eval_freq`, `n_episodes`, `discount`, `n_critics` | Standard offline RL schedule |
| CAPO schedule | `use_capo`, `capo_start_step`, `capo_period`, `n_max`, `refine_steps` | When and how often CAPO runs |
| Certificate | `beta_uncertainty`, `shift_penalty_coef`, `data_penalty_coef`, `normalize_delta_q`, `split_critics_for_certification` | Offline improvement certificate |
| τ search | `tau_controller: pilot_adaptive`, `tau_candidates`, `tau_min`/`tau_max`, `target_action_mse`, `tau_pilot_initial`, `max_action_mse` | Pilot-adaptive proximal step size |
| Teacher / actor | `lambda_D`, `lambda_T`, `teacher_hold`, `hold_teacher_on_nstar_zero`, `use_replace_gate`, `replace_cert_margin` | Student loss and teacher lifecycle |
| Eval | `eval_base_actor`, `eval_teacher_actor`, `paired_eval_episodes` | Student vs teacher rollouts |

CLI flags on `run_capo.py` override many of these without editing YAML.

---

## Method (brief)

1. **Base learner (θL):** TD3+BC, IQL, or CQL updates θL and critics; **all critic targets use θL** (and θL-target), never θR.

2. **CAPO refresh:** Every `capo_period` steps after `capo_start_step`, build a challenger teacher from θL using multi-step proximal refinement. **τ** is chosen by **`pilot_adaptive`** control (target action MSE vs certificate gain), not a full grid over all candidates each step.

3. **Certificate:** Scale-normalized ΔQ minus uncertainty, shift, and data-distance penalties yields an offline improvement certificate; the procedure selects τ and effective step count **N\***.

4. **Replace gate:** When `use_replace_gate` is true, a pairwise certificate compares incumbent vs challenger; swap only if improvement exceeds `replace_cert_margin`.

5. **Hold on N\* = 0:** With `hold_teacher_on_nstar_zero`, if the challenger is rejected (N\* = 0), keep the **incumbent teacher** for soft BC instead of turning teacher guidance off.

6. **Actor loss (TD3+BC path):**

   `L = -mean(Q / q_scale) + λ_D · BC(data) + teacher_active · λ_T · BC(teacher)`

   with element-wise MSE for BC terms and `q_scale = stopgrad(mean(|Q|) + ε)`.

See `capo/core.py` and `capo/trainer.py` module docstrings for design constraints.

---

## Testing

```bash
cd /path/to/CAPO
pytest tests/test_pilot_adaptive.py -q
# or
python -m pytest tests/test_pilot_adaptive.py
```

---

## License

See repository defaults on GitHub; add a `LICENSE` file if you open-source formally.

---

## Citation

If you use this code, please cite the associated paper (TBD) and link to [https://github.com/seonvin0319/CAPO](https://github.com/seonvin0319/CAPO).
