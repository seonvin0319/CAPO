# Shared CAPO sweep summaries

One markdown per sweep, shared by all hosts:

```text
host/sweeps/<sweep_name>/sweep_summary.md
```

```bash
python scripts/update_stability_sweep_summary.py --host choi --sweep capo_stability_seed0_fast_n2
git pull --rebase
git add host/sweeps/capo_stability_seed0_fast_n2/sweep_summary.md
git commit -m "update stability sweep summary"
git push
```

Do not commit `results_jax_sweeps/` checkpoints.
