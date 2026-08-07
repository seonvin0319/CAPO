# Sweep summary — antmaze `n8` (`n_critics=8`)

Updated: **2026-08-07 09:05 KST**
Host: **ext_csh**
Progress: **42/42** complete, 0 running, 0 pending

## Best @1M (completed)

| env | best final | λ_T | margin | run |
|---|---:|---:|---|---|
| umaze | **60** | 0.0 | 0 | `fastamn8_amu_s0_lt0_p100k_m0_replace_new` |
| umaze-diverse | **80** | 0.5 | 0 | `fastamn8_amud_s0_lt0p5_p100k_m0_replace_new` |
| medium-play | **0** | 0.0 | 0 | `fastamn8_amm_s0_lt0_p100k_m0_replace_new` |
| medium-diverse | **0** | 0.0 | 0 | `fastamn8_ammd_s0_lt0_p100k_m0_replace_new` |
| large-play | **0** | 0.0 | 0 | `fastamn8_aml_s0_lt0_p100k_m0_replace_new` |
| large-diverse | **0** | 0.0 | 0 | `fastamn8_amld_s0_lt0_p100k_m0_replace_new` |

## Matrices

### umaze

| λ_T \ margin | 0 | +1e-3 | −1e-3 |
|---|---:|---:|---:|
| 0.0 | 60 | n/a | n/a |
| 0.5 | 0 | 30 | 20 |
| 1.0 | 0 | 30 | 10 |

### umaze-diverse

| λ_T \ margin | 0 | +1e-3 | −1e-3 |
|---|---:|---:|---:|
| 0.0 | 60 | n/a | n/a |
| 0.5 | 80 | 50 | 40 |
| 1.0 | 20 | 40 | 40 |

### medium-play

| λ_T \ margin | 0 | +1e-3 | −1e-3 |
|---|---:|---:|---:|
| 0.0 | 0 | n/a | n/a |
| 0.5 | 0 | 0 | 0 |
| 1.0 | 0 | 0 | 0 |

### medium-diverse

| λ_T \ margin | 0 | +1e-3 | −1e-3 |
|---|---:|---:|---:|
| 0.0 | 0 | n/a | n/a |
| 0.5 | 0 | 0 | 0 |
| 1.0 | 0 | 0 | 0 |

### large-play

| λ_T \ margin | 0 | +1e-3 | −1e-3 |
|---|---:|---:|---:|
| 0.0 | 0 | n/a | n/a |
| 0.5 | 0 | 0 | 0 |
| 1.0 | 0 | 0 | 0 |

### large-diverse

| λ_T \ margin | 0 | +1e-3 | −1e-3 |
|---|---:|---:|---:|
| 0.0 | 0 | n/a | n/a |
| 0.5 | 0 | 0 | 0 |
| 1.0 | 0 | 0 | 0 |

## vs n4 (paired completed)

| env | λ_T | margin | n4 | this | Δ |
|---|---:|---|---:|---:|---:|
| umaze | 0.0 | 0 | 60 | **60** | +0 |
| umaze | 0.5 | -1e-3 | 50 | **20** | -30 |
| umaze | 0.5 | 0 | 70 | **0** | -70 |
| umaze | 0.5 | +1e-3 | 90 | **30** | -60 |
| umaze | 1.0 | -1e-3 | 70 | **10** | -60 |
| umaze | 1.0 | 0 | 60 | **0** | -60 |
| umaze | 1.0 | +1e-3 | 80 | **30** | -50 |
| umaze-diverse | 0.0 | 0 | 100 | **60** | -40 |
| umaze-diverse | 0.5 | -1e-3 | 80 | **40** | -40 |
| umaze-diverse | 0.5 | 0 | 60 | **80** | +20 |
| umaze-diverse | 0.5 | +1e-3 | 0 | **50** | +50 |
| umaze-diverse | 1.0 | -1e-3 | 70 | **40** | -30 |
| umaze-diverse | 1.0 | 0 | 30 | **20** | -10 |
| umaze-diverse | 1.0 | +1e-3 | 50 | **40** | -10 |
| medium-play | 0.0 | 0 | 0 | **0** | +0 |
| medium-play | 0.5 | -1e-3 | 0 | **0** | +0 |
| medium-play | 0.5 | 0 | 0 | **0** | +0 |
| medium-play | 0.5 | +1e-3 | 0 | **0** | +0 |
| medium-play | 1.0 | -1e-3 | 0 | **0** | +0 |
| medium-play | 1.0 | 0 | 0 | **0** | +0 |
| medium-play | 1.0 | +1e-3 | 0 | **0** | +0 |
| medium-diverse | 0.0 | 0 | 0 | **0** | +0 |
| medium-diverse | 0.5 | -1e-3 | 0 | **0** | +0 |
| medium-diverse | 0.5 | 0 | 0 | **0** | +0 |
| medium-diverse | 0.5 | +1e-3 | 0 | **0** | +0 |
| medium-diverse | 1.0 | -1e-3 | 0 | **0** | +0 |
| medium-diverse | 1.0 | 0 | 0 | **0** | +0 |
| medium-diverse | 1.0 | +1e-3 | 0 | **0** | +0 |
| large-play | 0.0 | 0 | 0 | **0** | +0 |
| large-play | 0.5 | -1e-3 | 0 | **0** | +0 |
| large-play | 0.5 | 0 | 0 | **0** | +0 |
| large-play | 0.5 | +1e-3 | 0 | **0** | +0 |
| large-play | 1.0 | -1e-3 | 0 | **0** | +0 |
| large-play | 1.0 | 0 | 0 | **0** | +0 |
| large-play | 1.0 | +1e-3 | 0 | **0** | +0 |
| large-diverse | 0.0 | 0 | 0 | **0** | +0 |
| large-diverse | 0.5 | -1e-3 | 0 | **0** | +0 |
| large-diverse | 0.5 | 0 | 0 | **0** | +0 |
| large-diverse | 0.5 | +1e-3 | 0 | **0** | +0 |
| large-diverse | 1.0 | -1e-3 | 0 | **0** | +0 |
| large-diverse | 1.0 | 0 | 0 | **0** | +0 |
| large-diverse | 1.0 | +1e-3 | 0 | **0** | +0 |

- paired n=42, mean Δ=-9.3, wins=2/42
