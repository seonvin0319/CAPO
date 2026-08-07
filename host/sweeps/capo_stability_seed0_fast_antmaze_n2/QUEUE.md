# Queue — `capo_stability_seed0_fast_antmaze_n2` (`n_critics=2`)

- Status: **CANCELLED** (replaced by n16 in chain) (chained after `capo_stability_seed0_fast_antmaze` n4)
- Manifest: `manifests/capo_stability_seed0_fast_antmaze_n2.jsonl` (**42** cells)
- Host: **ext_csh** · GPUs `0,1` × 6 = **12 concurrent**
- Same grid as n4 antmaze A:
  - `λ_T=0`: margin `0` only
  - `λ_T∈{0.5,1}` × margin ∈ {`0`, `+1e-3`, `-1e-3`}
- Fixed: `stale=replace_new`, `period=100k`, seed 0, 1M steps
- Results: `results_jax_sweeps/capo_stability_seed0_fast_antmaze_n2/`
