# Sweep summary — `capo_stability_seed0_fast_n2` (`n_critics=2`)

Updated: **2026-08-05 21:13 KST**
Host: **ext_csh**
Source: `results_jax_sweeps/capo_stability_seed0_fast_n2/`
Baseline (ref): `td3bc_4critic_jax_seed0` — **n_critics=4** TD3+BC (no matched n=2 baseline yet)

## Progress

- Complete: **8/36** (m0=5/18, mm1e3=3/18)
- Envs: hopper / halfcheetah / walker2d × medium-expert + medium-replay
- Factors: `replace_new`, period=100k, λ_T∈{0,0.5,1}, margin∈{0,-1e-3}

## Best so far vs 4-critic baseline (per env)

| env | baseline(n4) | best CAPO n2 | Δ | best run | λ_T | margin |
|---|---:|---:|---:|---|---:|---:|
| hopper-medium-expert | 105.48 | **100.16** | -5.31 | `fastn2_hmexp_s0_lt0_p100k_mm1e3_replace_new` | 0.0 | -0.001 |
| hopper-medium-replay | 23.35 | **23.55** | +0.20 | `fastn2_hmr_s0_lt0_p100k_m0_replace_new` | 0.0 | 0.0 |
| halfcheetah-medium-expert | 81.27 | — | — | — | — | — |
| halfcheetah-medium-replay | 45.39 | — | — | — | — | — |
| walker2d-medium-expert | 110.85 | — | — | — | — | — |
| walker2d-medium-replay | 30.78 | — | — | — | — | — |

## Active queue progress

| env | m0 done | mm1e3 done | best m0 | best −1e-3 |
|---|---:|---:|---:|---:|
| hopper-medium-expert | 3/3 | 3/3 | **98.15** (`lt=0.0`) | **100.16** (`lt=0.0`) |
| hopper-medium-replay | 2/3 | 0/3 | **23.55** (`lt=0.0`) | — |
| halfcheetah-medium-expert | 0/3 | 0/3 | — | — |
| halfcheetah-medium-replay | 0/3 | 0/3 | — | — |
| walker2d-medium-expert | 0/3 | 0/3 | — | — |
| walker2d-medium-replay | 0/3 | 0/3 | — | — |

## Rankings by env

### hopper-medium-expert

| rank | score | λ_T | margin | run |
|---:|---:|---:|---:|---|
| 1 | **100.16** | 0.0 | -0.001 | `fastn2_hmexp_s0_lt0_p100k_mm1e3_replace_new` |
| 2 | 98.15 | 0.0 | 0.0 | `fastn2_hmexp_s0_lt0_p100k_m0_replace_new` |
| 3 | 34.86 | 0.5 | -0.001 | `fastn2_hmexp_s0_lt0p5_p100k_mm1e3_replace_new` |
| 4 | 25.70 | 0.5 | 0.0 | `fastn2_hmexp_s0_lt0p5_p100k_m0_replace_new` |
| 5 | 2.07 | 1.0 | 0.0 | `fastn2_hmexp_s0_lt1_p100k_m0_replace_new` |
| 6 | 1.56 | 1.0 | -0.001 | `fastn2_hmexp_s0_lt1_p100k_mm1e3_replace_new` |

### hopper-medium-replay

| rank | score | λ_T | margin | run |
|---:|---:|---:|---:|---|
| 1 | **23.55** | 0.0 | 0.0 | `fastn2_hmr_s0_lt0_p100k_m0_replace_new` |
| 2 | 23.21 | 0.5 | 0.0 | `fastn2_hmr_s0_lt0p5_p100k_m0_replace_new` |

### halfcheetah-medium-expert

_none complete yet_

### halfcheetah-medium-replay

_none complete yet_

### walker2d-medium-expert

_none complete yet_

### walker2d-medium-replay

_none complete yet_
