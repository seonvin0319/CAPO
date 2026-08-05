# Queue — `capo_stability_seed0_fast_n2` (`n_critics=2`)

- Manifest: `manifests/capo_stability_seed0_fast_n2.jsonl` (36 cells)
- Config: `configs/sweeps/capo_stability_seed0_fast_n2.yaml`
- Summary: `host/sweeps/capo_stability_seed0_fast_n2/sweep_summary.md`
- Envs: hopper / halfcheetah / walker2d × medium-expert + medium-replay
- Factors: `replace_new`, period=100k, λ_T∈{0,0.5,1}, margin∈{0,-1e-3}
- Chained after n4 via runner `--prepend_manifest` (n4) + `--manifest` (n2)
- Results: `results_jax_sweeps/capo_stability_seed0_fast_n2/`
