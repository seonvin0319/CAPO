# Queue — `capo_stability_seed0_fast_antmaze` (`n_critics=4`)

- Status: **RUNNING** (resumed; summary refreshed 2026-08-06 07:58 KST) — see `sweep_summary.md` (**25/42** complete)
- Manifest: `manifests/capo_stability_seed0_fast_antmaze.jsonl` (**42** cells)
- Host: **ext_csh** · GPUs `0,1` × 6 = **12 concurrent** (bumped 2026-08-06 15:35 KST)
- Per env (7 cells):
  - `λ_T=0`: margin `0` only — **1 cell** (teacher term off; margin irrelevant)
  - `λ_T=0.5`: margin ∈ {`0`, `+1e-3`, `-1e-3`} — 3 cells
  - `λ_T=1`: margin ∈ {`0`, `+1e-3`, `-1e-3`} — 3 cells
- Fixed: `stale=replace_new`, `period=100k`, `n_critics=4`, seed 0, 1M steps
- Envs: umaze / umaze-diverse / medium-play / medium-diverse / large-play / large-diverse
- Results: `results_jax_sweeps/capo_stability_seed0_fast_antmaze/`
- Runner log: `results_jax_sweeps/capo_fast_antmaze_runner.log`
- Note: two `amm λ_T=0` dirs quarantined (`*.QUARANTINE_nocckpt_*`, no checkpoint)

## Follow-on chain (armed 2026-08-06 10:34 KST)

After this n4 matrix finishes (`final.pkl` / `checkpoint_1000000.pkl` ×42):
1. `capo_stability_seed0_fast_antmaze_n8` (42 cells, `n_critics=8`)
2. `capo_stability_seed0_fast_antmaze_n16` (42 cells, `n_critics=16`)

Waiter: `scripts/chain_antmaze_n8_n16_n32.sh`
Chain log: `results_jax_sweeps/capo_antmaze_chain_n8_n16_n32.log`

- **2026-08-07:** n32 cancelled on ext_csh — finish n16 only, no n32 launch.
