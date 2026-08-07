# Queue — `capo_stability_seed0_fast_n4_uwc`

- Status: **QUEUED** after UC Phase1 (`capo_antmaze_uncertainty_critic_seed0`)
- Manifest: `manifests/capo_stability_seed0_fast_n4_uwc_replace_new_mgrid.jsonl` (54 cells: period=100k only)
- Grid: same as choi n2 replace_new mgrid (λ∈{0.5,1} × p=100k × m∈{0,+1e-3,−1e-3} × 9 env)
- Overrides: `n_critics=4`, UWC=true, `actor_type=deterministic`, `distance_metric=amse`
- Launch: GPU0 × 2 jobs · chain `scripts/chain_uc_p1_then_loco_n4n8_uwc.sh`
