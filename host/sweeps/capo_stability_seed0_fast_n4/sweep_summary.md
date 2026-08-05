# Sweep summary — `capo_stability_seed0_fast` (`n_critics=4`)

Updated: **2026-08-05 21:13 KST**
Host: **ext_csh**
Source: `results_jax_sweeps/capo_stability_seed0_fast/`
Baseline: `results_jax_sweeps/td3bc_4critic_jax_seed0/` (TD3+BC, `n_critics=4`, seed 0)

## Best vs baseline (per env)

| env | baseline | best CAPO | Δ (best − bl) | best run | λ_T | period | margin | stale |
|---|---:|---:|---:|---|---:|---:|---:|---|
| hopper-medium | 57.60 | **100.27** | +42.68 | `fast_hm_s0_lt1_p100k_m0_replace_new` | 1.0 | 100000 | 0.0 | replace_new |
| hopper-medium-expert | 105.48 | **112.34** | +6.86 | `fast_hmexp_s0_lt0p5_p100k_m0_replace_new` | 0.5 | 100000 | 0.0 | replace_new |
| hopper-medium-replay | 23.35 | **85.89** | +62.54 | `fast_hmr_s0_lt0p5_p100k_mm1e3_replace_new` | 0.5 | 100000 | -0.001 | replace_new |
| halfcheetah-medium | 49.44 | **53.15** | +3.71 | `fast_cm_s0_lt1_p100k_m0_disable` | 1.0 | 100000 | 0.0 | disable |
| halfcheetah-medium-expert | 81.27 | **90.85** | +9.57 | `fast_cmexp_s0_lt0_p100k_m0_quarantine` | 0.0 | 100000 | 0.0 | quarantine |
| halfcheetah-medium-replay | 45.39 | **49.06** | +3.67 | `fast_cmr_s0_lt0_p100k_mm1e3_replace_new` | 0.0 | 100000 | -0.001 | replace_new |
| walker2d-medium | 83.47 | **84.96** | +1.50 | `fast_wm_s0_lt1_p100k_m0_quarantine` | 1.0 | 100000 | 0.0 | quarantine |
| walker2d-medium-expert | 110.85 | **114.92** | +4.07 | `fast_wmexp_s0_lt1_p100k_m0_replace_new` | 1.0 | 100000 | 0.0 | replace_new |
| walker2d-medium-replay | 30.78 | **91.98** | +61.19 | `fast_wmr_s0_lt0p5_p100k_m0_replace_new` | 0.5 | 100000 | 0.0 | replace_new |

## Setup

- JAX CAPO, seed 0, **`n_critics=4`**
- Active policy: `stale=replace_new`, `period=100k`, `λ_T∈{0,0.5,1}`, `margin∈{0,-1e-3}`
- Metric: final `student_d4rl_score` @1M
- Active complete: **54/54** (m0=27/27, mm1e3=27/27)
- All-history complete artifacts used for best: **158**

## Active queue progress

| env | m0 done | mm1e3 done | best any (hist) | best replace m0 | best replace −1e-3 |
|---|---:|---:|---:|---:|---:|
| hopper-medium | 3/3 | 3/3 | 100.27 | **100.27** (`lt=1.0`) | **84.72** (`lt=0.5`) |
| hopper-medium-expert | 3/3 | 3/3 | 112.34 | **112.34** (`lt=0.5`) | **100.83** (`lt=1.0`) |
| hopper-medium-replay | 3/3 | 3/3 | 85.89 | **73.80** (`lt=0.5`) | **85.89** (`lt=0.5`) |
| halfcheetah-medium | 3/3 | 3/3 | 53.15 | **51.55** (`lt=1.0`) | **52.19** (`lt=1.0`) |
| halfcheetah-medium-expert | 3/3 | 3/3 | 90.85 | **79.91** (`lt=1.0`) | **85.33** (`lt=0.0`) |
| halfcheetah-medium-replay | 3/3 | 3/3 | 49.06 | **48.52** (`lt=0.5`) | **49.06** (`lt=0.0`) |
| walker2d-medium | 3/3 | 3/3 | 84.96 | **84.38** (`lt=0.0`) | **83.02** (`lt=0.5`) |
| walker2d-medium-expert | 3/3 | 3/3 | 114.92 | **114.92** (`lt=1.0`) | **114.12** (`lt=1.0`) |
| walker2d-medium-replay | 3/3 | 3/3 | 91.98 | **91.98** (`lt=0.5`) | **89.70** (`lt=0.5`) |

## Per-env #1 (all completed artifacts)

| env | score | run | λ_T | period | margin | stale |
|---|---:|---|---:|---:|---:|---|
| hopper-medium | **100.27** | `fast_hm_s0_lt1_p100k_m0_replace_new` | 1.0 | 100000 | 0.0 | replace_new |
| hopper-medium-expert | **112.34** | `fast_hmexp_s0_lt0p5_p100k_m0_replace_new` | 0.5 | 100000 | 0.0 | replace_new |
| hopper-medium-replay | **85.89** | `fast_hmr_s0_lt0p5_p100k_mm1e3_replace_new` | 0.5 | 100000 | -0.001 | replace_new |
| halfcheetah-medium | **53.15** | `fast_cm_s0_lt1_p100k_m0_disable` | 1.0 | 100000 | 0.0 | disable |
| halfcheetah-medium-expert | **90.85** | `fast_cmexp_s0_lt0_p100k_m0_quarantine` | 0.0 | 100000 | 0.0 | quarantine |
| halfcheetah-medium-replay | **49.06** | `fast_cmr_s0_lt0_p100k_mm1e3_replace_new` | 0.0 | 100000 | -0.001 | replace_new |
| walker2d-medium | **84.96** | `fast_wm_s0_lt1_p100k_m0_quarantine` | 1.0 | 100000 | 0.0 | quarantine |
| walker2d-medium-expert | **114.92** | `fast_wmexp_s0_lt1_p100k_m0_replace_new` | 1.0 | 100000 | 0.0 | replace_new |
| walker2d-medium-replay | **91.98** | `fast_wmr_s0_lt0p5_p100k_m0_replace_new` | 0.5 | 100000 | 0.0 | replace_new |

## Active `replace_new` @100k rankings (m0 + mm1e3)

### hopper-medium

| rank | score | λ_T | margin | run |
|---:|---:|---:|---:|---|
| 1 | **100.27** | 1.0 | 0.0 | `fast_hm_s0_lt1_p100k_m0_replace_new` |
| 2 | 84.72 | 0.5 | -0.001 | `fast_hm_s0_lt0p5_p100k_mm1e3_replace_new` |
| 3 | 66.13 | 0.5 | 0.0 | `fast_hm_s0_lt0p5_p100k_m0_replace_new` |
| 4 | 65.49 | 1.0 | -0.001 | `fast_hm_s0_lt1_p100k_mm1e3_replace_new` |
| 5 | 65.20 | 0.0 | 0.0 | `fast_hm_s0_lt0_p100k_m0_replace_new` |
| 6 | 59.68 | 0.0 | -0.001 | `fast_hm_s0_lt0_p100k_mm1e3_replace_new` |

### hopper-medium-expert

| rank | score | λ_T | margin | run |
|---:|---:|---:|---:|---|
| 1 | **112.34** | 0.5 | 0.0 | `fast_hmexp_s0_lt0p5_p100k_m0_replace_new` |
| 2 | 112.16 | 0.0 | 0.0 | `fast_hmexp_s0_lt0_p100k_m0_replace_new` |
| 3 | 111.05 | 1.0 | 0.0 | `fast_hmexp_s0_lt1_p100k_m0_replace_new` |
| 4 | 100.83 | 1.0 | -0.001 | `fast_hmexp_s0_lt1_p100k_mm1e3_replace_new` |
| 5 | 80.69 | 0.0 | -0.001 | `fast_hmexp_s0_lt0_p100k_mm1e3_replace_new` |
| 6 | 56.97 | 0.5 | -0.001 | `fast_hmexp_s0_lt0p5_p100k_mm1e3_replace_new` |

### hopper-medium-replay

| rank | score | λ_T | margin | run |
|---:|---:|---:|---:|---|
| 1 | **85.89** | 0.5 | -0.001 | `fast_hmr_s0_lt0p5_p100k_mm1e3_replace_new` |
| 2 | 73.80 | 0.5 | 0.0 | `fast_hmr_s0_lt0p5_p100k_m0_replace_new` |
| 3 | 65.89 | 0.0 | 0.0 | `fast_hmr_s0_lt0_p100k_m0_replace_new` |
| 4 | 45.92 | 1.0 | 0.0 | `fast_hmr_s0_lt1_p100k_m0_replace_new` |
| 5 | 24.16 | 0.0 | -0.001 | `fast_hmr_s0_lt0_p100k_mm1e3_replace_new` |
| 6 | 22.62 | 1.0 | -0.001 | `fast_hmr_s0_lt1_p100k_mm1e3_replace_new` |

### halfcheetah-medium

| rank | score | λ_T | margin | run |
|---:|---:|---:|---:|---|
| 1 | **52.19** | 1.0 | -0.001 | `fast_cm_s0_lt1_p100k_mm1e3_replace_new` |
| 2 | 51.55 | 1.0 | 0.0 | `fast_cm_s0_lt1_p100k_m0_replace_new` |
| 3 | 51.39 | 0.0 | 0.0 | `fast_cm_s0_lt0_p100k_m0_replace_new` |
| 4 | 51.36 | 0.0 | -0.001 | `fast_cm_s0_lt0_p100k_mm1e3_replace_new` |
| 5 | 51.09 | 0.5 | -0.001 | `fast_cm_s0_lt0p5_p100k_mm1e3_replace_new` |
| 6 | 50.98 | 0.5 | 0.0 | `fast_cm_s0_lt0p5_p100k_m0_replace_new` |

### halfcheetah-medium-expert

| rank | score | λ_T | margin | run |
|---:|---:|---:|---:|---|
| 1 | **85.33** | 0.0 | -0.001 | `fast_cmexp_s0_lt0_p100k_mm1e3_replace_new` |
| 2 | 79.91 | 1.0 | 0.0 | `fast_cmexp_s0_lt1_p100k_m0_replace_new` |
| 3 | 77.63 | 0.5 | 0.0 | `fast_cmexp_s0_lt0p5_p100k_m0_replace_new` |
| 4 | 73.01 | 0.0 | 0.0 | `fast_cmexp_s0_lt0_p100k_m0_replace_new` |
| 5 | 71.01 | 0.5 | -0.001 | `fast_cmexp_s0_lt0p5_p100k_mm1e3_replace_new` |
| 6 | 70.62 | 1.0 | -0.001 | `fast_cmexp_s0_lt1_p100k_mm1e3_replace_new` |

### halfcheetah-medium-replay

| rank | score | λ_T | margin | run |
|---:|---:|---:|---:|---|
| 1 | **49.06** | 0.0 | -0.001 | `fast_cmr_s0_lt0_p100k_mm1e3_replace_new` |
| 2 | 48.52 | 0.5 | 0.0 | `fast_cmr_s0_lt0p5_p100k_m0_replace_new` |
| 3 | 47.99 | 1.0 | 0.0 | `fast_cmr_s0_lt1_p100k_m0_replace_new` |
| 4 | 47.97 | 1.0 | -0.001 | `fast_cmr_s0_lt1_p100k_mm1e3_replace_new` |
| 5 | 46.80 | 0.0 | 0.0 | `fast_cmr_s0_lt0_p100k_m0_replace_new` |
| 6 | 46.40 | 0.5 | -0.001 | `fast_cmr_s0_lt0p5_p100k_mm1e3_replace_new` |

### walker2d-medium

| rank | score | λ_T | margin | run |
|---:|---:|---:|---:|---|
| 1 | **84.38** | 0.0 | 0.0 | `fast_wm_s0_lt0_p100k_m0_replace_new` |
| 2 | 84.23 | 0.5 | 0.0 | `fast_wm_s0_lt0p5_p100k_m0_replace_new` |
| 3 | 83.02 | 0.5 | -0.001 | `fast_wm_s0_lt0p5_p100k_mm1e3_replace_new` |
| 4 | 82.08 | 1.0 | 0.0 | `fast_wm_s0_lt1_p100k_m0_replace_new` |
| 5 | 80.79 | 1.0 | -0.001 | `fast_wm_s0_lt1_p100k_mm1e3_replace_new` |
| 6 | 77.33 | 0.0 | -0.001 | `fast_wm_s0_lt0_p100k_mm1e3_replace_new` |

### walker2d-medium-expert

| rank | score | λ_T | margin | run |
|---:|---:|---:|---:|---|
| 1 | **114.92** | 1.0 | 0.0 | `fast_wmexp_s0_lt1_p100k_m0_replace_new` |
| 2 | 114.12 | 1.0 | -0.001 | `fast_wmexp_s0_lt1_p100k_mm1e3_replace_new` |
| 3 | 113.57 | 0.5 | -0.001 | `fast_wmexp_s0_lt0p5_p100k_mm1e3_replace_new` |
| 4 | 113.03 | 0.5 | 0.0 | `fast_wmexp_s0_lt0p5_p100k_m0_replace_new` |
| 5 | 112.17 | 0.0 | -0.001 | `fast_wmexp_s0_lt0_p100k_mm1e3_replace_new` |
| 6 | 111.77 | 0.0 | 0.0 | `fast_wmexp_s0_lt0_p100k_m0_replace_new` |

### walker2d-medium-replay

| rank | score | λ_T | margin | run |
|---:|---:|---:|---:|---|
| 1 | **91.98** | 0.5 | 0.0 | `fast_wmr_s0_lt0p5_p100k_m0_replace_new` |
| 2 | 89.70 | 0.5 | -0.001 | `fast_wmr_s0_lt0p5_p100k_mm1e3_replace_new` |
| 3 | 86.10 | 1.0 | 0.0 | `fast_wmr_s0_lt1_p100k_m0_replace_new` |
| 4 | 79.07 | 0.0 | -0.001 | `fast_wmr_s0_lt0_p100k_mm1e3_replace_new` |
| 5 | 77.37 | 1.0 | -0.001 | `fast_wmr_s0_lt1_p100k_mm1e3_replace_new` |
| 6 | 64.71 | 0.0 | 0.0 | `fast_wmr_s0_lt0_p100k_m0_replace_new` |

## Margin 0 vs −1e-3 (best λ_T each)

| env | best m0 | best −1e-3 | Δ (−1e-3 − m0) |
|---|---:|---:|---:|
| hopper-medium | 100.27 | 84.72 | -15.55 |
| hopper-medium-expert | 112.34 | 100.83 | -11.51 |
| hopper-medium-replay | 73.80 | 85.89 | +12.09 |
| halfcheetah-medium | 51.55 | 52.19 | +0.64 |
| halfcheetah-medium-expert | 79.91 | 85.33 | +5.42 |
| halfcheetah-medium-replay | 48.52 | 49.06 | +0.54 |
| walker2d-medium | 84.38 | 83.02 | -1.35 |
| walker2d-medium-expert | 114.92 | 114.12 | -0.80 |
| walker2d-medium-replay | 91.98 | 89.70 | -2.27 |

## Notes

- This file is **only** the `n_critics=4` campaign summary.
- Follow-on `n_critics=2` (expert/replay): `host/sweeps/capo_stability_seed0_fast_n2/`.
