# CAPO stability — `capo_stability_seed0_fast_n2`

Updated: 2026-08-06 11:13 KST · `choi` · **108/108** · ETA ~0.0h (≈08-06 11:13 KST)

Active: λ_T∈{0.5,1}, period∈{50k,100k}, margin∈{0,1e-3,−1e-3}, stale=replace_new, n_critics=2.

Shared: `host/sweeps/capo_stability_seed0_fast_n2/sweep_summary.md` · local runs: `results_jax_sweeps/capo_stability_seed0_fast_n2/`

## Best vs baseline (per env)

Best = highest final `student_d4rl_score` among **active** completes. Baseline = local seed0 TD3+BC `baseline_n2` when available, else n4 reference.

| env | done | best | baseline | Δ | setting | run |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| hopper-medium | 12/12 | **100.20** | 53.61 (n2) | +46.59 | λ0.5 / p100k / m−1e-3 | `fast_hm_s0_lt0p5_p100k_mm1e3_replace_new` |
| hopper-medium-expert | 12/12 | **112.40** | 79.49 (n2) | +32.91 | λ1 / p100k / m1e-3 | `fast_hmexp_s0_lt1_p100k_m1e3_replace_new` |
| hopper-medium-replay | 12/12 | **100.62** | 21.11 (n2) | +79.51 | λ0.5 / p100k / m0 | `fast_hmr_s0_lt0p5_p100k_m0_replace_new` |
| halfcheetah-medium | 12/12 | **52.63** | 48.21 (n2) | +4.42 | λ1 / p50k / m0 | `fast_cm_s0_lt1_p50k_m0_replace_new` |
| halfcheetah-medium-expert | 12/12 | **82.36** | 85.67 (n2) | -3.31 | λ1 / p50k / m1e-3 | `fast_cmexp_s0_lt1_p50k_m1e3_replace_new` |
| halfcheetah-medium-replay | 12/12 | **47.61** | 44.81 (n2) | +2.80 | λ1 / p50k / m1e-3 | `fast_cmr_s0_lt1_p50k_m1e3_replace_new` |
| walker2d-medium | 12/12 | **86.99** | 80.96 (n2) | +6.03 | λ1 / p50k / m−1e-3 | `fast_wm_s0_lt1_p50k_mm1e3_replace_new` |
| walker2d-medium-expert | 12/12 | **115.64** | 111.06 (n2) | +4.58 | λ0.5 / p100k / m0 | `fast_wmexp_s0_lt0p5_p100k_m0_replace_new` |
| walker2d-medium-replay | 12/12 | **92.89** | 84.16 (n2) | +8.73 | λ1 / p50k / m1e-3 | `fast_wmr_s0_lt1_p50k_m1e3_replace_new` |

## Margin effect (active completes)

| margin | n | mean | min | max |
| --- | ---: | ---: | ---: | ---: |
| 0.0 | 36 | 68.73 | 1.54 | 115.64 |
| 0.001 | 36 | 69.72 | 2.30 | 115.03 |
| -0.001 | 36 | 64.82 | 1.85 | 114.85 |

## Top-3 per env (active)

**hopper-medium** — 1. **100.20** (λ0.5 / p100k / m−1e-3) · 2. **92.17** (λ0.5 / p50k / m−1e-3) · 3. **89.00** (λ1 / p100k / m1e-3)

**hopper-medium-expert** — 1. **112.40** (λ1 / p100k / m1e-3) · 2. **111.48** (λ0.5 / p50k / m0) · 3. **108.40** (λ0.5 / p50k / m−1e-3)

**hopper-medium-replay** — 1. **100.62** (λ0.5 / p100k / m0) · 2. **98.53** (λ0.5 / p100k / m1e-3) · 3. **96.99** (λ1 / p100k / m1e-3)

**halfcheetah-medium** — 1. **52.63** (λ1 / p50k / m0) · 2. **52.11** (λ1 / p50k / m−1e-3) · 3. **51.70** (λ0.5 / p100k / m−1e-3)

**halfcheetah-medium-expert** — 1. **82.36** (λ1 / p50k / m1e-3) · 2. **80.53** (λ0.5 / p50k / m0) · 3. **75.61** (λ0.5 / p100k / m−1e-3)

**halfcheetah-medium-replay** — 1. **47.61** (λ1 / p50k / m1e-3) · 2. **47.34** (λ1 / p50k / m−1e-3) · 3. **47.27** (λ0.5 / p100k / m−1e-3)

**walker2d-medium** — 1. **86.99** (λ1 / p50k / m−1e-3) · 2. **86.64** (λ0.5 / p100k / m0) · 3. **85.41** (λ0.5 / p50k / m1e-3)

**walker2d-medium-expert** — 1. **115.64** (λ0.5 / p100k / m0) · 2. **115.03** (λ0.5 / p100k / m1e-3) · 3. **114.85** (λ1 / p100k / m−1e-3)

**walker2d-medium-replay** — 1. **92.89** (λ1 / p50k / m1e-3) · 2. **90.71** (λ1 / p100k / m0) · 3. **88.91** (λ0.5 / p100k / m1e-3)

## Notes

- Final @1M from `summary.json`; Δ = best − baseline.
- Active: replace_new only, margins `{0, 1e-3, −1e-3}`.
- Pull/rebase before editing this file; do not commit checkpoints.
