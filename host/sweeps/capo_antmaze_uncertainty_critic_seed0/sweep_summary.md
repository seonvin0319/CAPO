# Sweep summary — `capo_antmaze_uncertainty_critic_seed0` (Phase 1 UC critic)

Updated: **2026-08-08 08:00 KST**
Host: **ext_csh**
Progress: **0/48** complete

## 1. Experiment setup

- Change isolated: `use_uncertainty_weighted_critic=True` + `critic_uncertainty_kappa`
- Envs: `antmaze-umaze-v2`, `antmaze-umaze-diverse-v2`
- Axes: `n_critics∈{4,8,16}` × `λ_T∈{0,0.5}` × `κ∈{0,0.5,1,2}`
- Fixed: seed0, `stale=replace_new`, period=100k, margin=0, 1M steps
- `κ=0` uses weighted path with weight≡1 (bit-identical to original TD loss per unit test)
- Phase 2: **not auto-started** — wait for Phase 1 answers below

## 2. λ_T = 0 paired results

| env | kappa | n4 | n8 | n16 | n8−n4 | n16−n4 |
|---|---:|---:|---:|---:|---:|---:|
| umaze | 0 | … | … | … | … | … |
| umaze | 0.5 | … | … | … | … | … |
| umaze | 1 | … | … | … | … | … |
| umaze | 2 | … | … | … | … | … |
| umaze-diverse | 0 | … | … | … | … | … |
| umaze-diverse | 0.5 | … | … | … | … | … |
| umaze-diverse | 1 | … | … | … | … | … |
| umaze-diverse | 2 | … | … | … | … | … |

### Legacy reference (prior antmaze, margin=0)

| env | n4 | n8 | n16 | n8−n4 | n16−n4 |
|---|---:|---:|---:|---:|---:|
| umaze | 60.0 | 60.0 | 70.0 | 0.0 | 10.0 |
| umaze-diverse | 100.0 | 60.0 | 30.0 | -40.0 | -70.0 |

## 3. λ_T = 0.5 paired results

| env | kappa | n4 | n8 | n16 | n8−n4 | n16−n4 |
|---|---:|---:|---:|---:|---:|---:|
| umaze | 0 | … | … | … | … | … |
| umaze | 0.5 | … | … | … | … | … |
| umaze | 1 | … | … | … | … | … |
| umaze | 2 | … | … | … | … | … |
| umaze-diverse | 0 | … | … | … | … | … |
| umaze-diverse | 0.5 | … | … | … | … | … |
| umaze-diverse | 1 | … | … | … | … | … |
| umaze-diverse | 2 | … | … | … | … | … |

## 4. Teacher-path diagnostics (λ_T=0.5 completes)

| n | kappa | env | score | accepted | accepted_cert | replace | unc_mean | w_mean |
|---:|---:|---|---:|---:|---:|---:|---:|---:|

## 5. n4/n8/n16 scaling (mean over envs)

| lambda_T | kappa | mean(n8−n4) | mean(n16−n4) |
|---:|---:|---:|---:|
| 0 | 0 | … | … |
| 0 | 0.5 | … | … |
| 0 | 1 | … | … |
| 0 | 2 | … | … |
| 0.5 | 0 | … | … |
| 0.5 | 0.5 | … | … |
| 0.5 | 1 | … | … |
| 0.5 | 2 | … | … |

## 6. Interpretation (auto draft — fill when complete)

Q1. UC critic이 n_critics 증가에 따른 base critic degradation을 줄였는가?
Q2. UC critic이 teacher-on collapse도 줄였는가?
Q3. best kappa는 무엇인가?
Q4. remaining failure가 critic learning 문제인가, gate 문제인가?
Q5. Phase 2 full sweep을 진행할 가치가 있는가?

_Incomplete: 0/48. Re-run this script after completes._
## 7. Plots

Generate after completes:
```bash
python scripts/plot_uc_critic_phase1_curves.py
```

## Notes

- Manifest: `manifests/capo_antmaze_uncertainty_critic_seed0.jsonl`
- Results: `results_jax_sweeps/capo_antmaze_uncertainty_critic_seed0/`
- Refresh: `python scripts/update_uc_critic_phase1_summary.py`
