# Queue — `capo_antmaze_uncertainty_critic_seed0` (Phase 1 UC critic)

- Status: **RELAUNCHING** on ext_csh with **new UC critic code** (bootstrap-masked
  uncertainty + optional weight normalization; default `none`)
- Manifest: `manifests/capo_antmaze_uncertainty_critic_seed0.jsonl` (**48** cells)
- Config: `configs/sweeps/capo_antmaze_uncertainty_critic_seed0.yaml`
- Results: `results_jax_sweeps/capo_antmaze_uncertainty_critic_seed0/`
- Prior completed wave (old code, 48/48): `results_jax_sweeps/capo_antmaze_uncertainty_critic_seed0.QUARANTINE_OLDCODE_pre_bootstrap_fix_20260807T230033Z`
  - Summary snapshot: `host/sweeps/capo_antmaze_uncertainty_critic_seed0/sweep_summary_OLDCODE_pre_bootstrap_fix.md`
- Runner: tmux `capo_uc_critic_p1` · log `results_jax_sweeps/capo_uc_critic_p1_runner.log`
- Change isolated: `use_uncertainty_weighted_critic=True` + `critic_uncertainty_kappa`
- Fixed to match prior antmaze: seed0, replace_new, period=100k, margin=0,
  `split_critics_for_certification=True`
- Envs: `antmaze-umaze-v2`, `antmaze-umaze-diverse-v2`
- Axes: `n_critics∈{4,8,16}` × `λ_T∈{0,0.5}` × `κ∈{0,0.5,1,2}`
- `κ=0` must match original TD loss (unit-tested)
- **Locomotion UWC paused** (soft-stopped) — AntMaze Phase1 only; no Phase2 auto-start
- Refresh summary: `python scripts/update_uc_critic_phase1_summary.py`
- Plots: `python scripts/plot_uc_critic_phase1_curves.py`

## Follow-on (armed)
After Phase1 48/48: auto-launch loco UWC n4+n8 (tmux capo_uc_p1_loco_chain).
Chain log: results_jax_sweeps/capo_uc_p1_then_loco_uwc_chain.log
