# Sweep summary — `capo_antmaze_uncertainty_critic_seed0` (Phase 1 UC critic)

Updated: **2026-08-08 07:58 KST**
Host: **ext_csh**
Progress: **48/48** complete

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
| umaze | 0 | 60.0 | 60.0 | 60.0 | 0.0 | 0.0 |
| umaze | 0.5 | 80.0 | 70.0 | 80.0 | -10.0 | 0.0 |
| umaze | 1 | 70.0 | 60.0 | 60.0 | -10.0 | -10.0 |
| umaze | 2 | 80.0 | 70.0 | 40.0 | -10.0 | -40.0 |
| umaze-diverse | 0 | 40.0 | 60.0 | 60.0 | 20.0 | 20.0 |
| umaze-diverse | 0.5 | 50.0 | 40.0 | 70.0 | -10.0 | 20.0 |
| umaze-diverse | 1 | 0.0 | 40.0 | 20.0 | 40.0 | 20.0 |
| umaze-diverse | 2 | 40.0 | 50.0 | 70.0 | 10.0 | 30.0 |

### Legacy reference (prior antmaze, margin=0)

| env | n4 | n8 | n16 | n8−n4 | n16−n4 |
|---|---:|---:|---:|---:|---:|
| umaze | 60.0 | 60.0 | 70.0 | 0.0 | 10.0 |
| umaze-diverse | 100.0 | 60.0 | 30.0 | -40.0 | -70.0 |

## 3. λ_T = 0.5 paired results

| env | kappa | n4 | n8 | n16 | n8−n4 | n16−n4 |
|---|---:|---:|---:|---:|---:|---:|
| umaze | 0 | 90.0 | 30.0 | 0.0 | -60.0 | -90.0 |
| umaze | 0.5 | 80.0 | 50.0 | 10.0 | -30.0 | -70.0 |
| umaze | 1 | 90.0 | 40.0 | 0.0 | -50.0 | -90.0 |
| umaze | 2 | 40.0 | 60.0 | 30.0 | 20.0 | -10.0 |
| umaze-diverse | 0 | 60.0 | 70.0 | 50.0 | 10.0 | -10.0 |
| umaze-diverse | 0.5 | 60.0 | 80.0 | 40.0 | 20.0 | -20.0 |
| umaze-diverse | 1 | 40.0 | 50.0 | 70.0 | 10.0 | 30.0 |
| umaze-diverse | 2 | 0.0 | 80.0 | 70.0 | 80.0 | 70.0 |

## 4. Teacher-path diagnostics (λ_T=0.5 completes)

| n | kappa | env | score | accepted | accepted_cert | replace | unc_mean | w_mean |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | 0 | umaze-diverse | 60.0 | 0.00 | … | 9 | 4.4494 | 1.0000 |
| 4 | 0 | umaze | 90.0 | 1.00 | 0.0025 | 10 | 1.8192 | 1.0000 |
| 4 | 0.5 | umaze-diverse | 60.0 | 1.00 | 0.1575 | 9 | 31.6615 | 0.9742 |
| 4 | 0.5 | umaze | 80.0 | 1.00 | 0.0015 | 10 | 1.6886 | 0.9922 |
| 4 | 1 | umaze-diverse | 40.0 | 0.00 | … | 5 | 0.7322 | 0.9941 |
| 4 | 1 | umaze | 90.0 | 1.00 | 0.0059 | 10 | 1.7383 | 0.9831 |
| 4 | 2 | umaze-diverse | 0.0 | 1.00 | 0.2210 | 9 | 21.1336 | 0.9397 |
| 4 | 2 | umaze | 40.0 | 1.00 | 0.0043 | 10 | 1.9021 | 0.9710 |
| 8 | 0 | umaze-diverse | 70.0 | 1.00 | 0.0117 | 10 | 2.0543 | 1.0000 |
| 8 | 0 | umaze | 30.0 | 1.00 | 0.0180 | 10 | 6.6185 | 1.0000 |
| 8 | 0.5 | umaze-diverse | 80.0 | 1.00 | 0.0103 | 10 | 1.9295 | 0.9962 |
| 8 | 0.5 | umaze | 50.0 | 1.00 | 0.0243 | 10 | 7.7235 | 0.9912 |
| 8 | 1 | umaze-diverse | 50.0 | 1.00 | 0.0089 | 10 | 2.1760 | 0.9913 |
| 8 | 1 | umaze | 40.0 | 1.00 | 0.0185 | 10 | 8.3583 | 0.9835 |
| 8 | 2 | umaze-diverse | 80.0 | 1.00 | 0.0066 | 10 | 2.2581 | 0.9821 |
| 8 | 2 | umaze | 60.0 | 1.00 | 0.0228 | 10 | 7.9903 | 0.9706 |
| 16 | 0 | umaze-diverse | 50.0 | 1.00 | 0.0292 | 10 | 8.7466 | 1.0000 |
| 16 | 0 | umaze | 0.0 | 1.00 | 0.0455 | 10 | 31.7949 | 1.0000 |
| 16 | 0.5 | umaze-diverse | 40.0 | 1.00 | 0.0256 | 10 | 9.6091 | 0.9942 |
| 16 | 0.5 | umaze | 10.0 | 1.00 | 0.0383 | 10 | 30.3342 | 0.9908 |
| 16 | 1 | umaze-diverse | 70.0 | 1.00 | 0.0257 | 10 | 8.5664 | 0.9888 |
| 16 | 1 | umaze | 0.0 | 1.00 | 0.0471 | 10 | 28.8919 | 0.9848 |
| 16 | 2 | umaze-diverse | 70.0 | 1.00 | 0.0275 | 10 | 8.4692 | 0.9779 |
| 16 | 2 | umaze | 30.0 | 1.00 | 0.0536 | 10 | 23.8340 | 0.9735 |

## 5. n4/n8/n16 scaling (mean over envs)

| lambda_T | kappa | mean(n8−n4) | mean(n16−n4) |
|---:|---:|---:|---:|
| 0 | 0 | 10.0 | 10.0 |
| 0 | 0.5 | -10.0 | 10.0 |
| 0 | 1 | 15.0 | 5.0 |
| 0 | 2 | 0.0 | -5.0 |
| 0.5 | 0 | -25.0 | -50.0 |
| 0.5 | 0.5 | -5.0 | -45.0 |
| 0.5 | 1 | -20.0 | -30.0 |
| 0.5 | 2 | 50.0 | 30.0 |

## 6. Interpretation (auto draft — fill when complete)

Q1. UC critic이 n_critics 증가에 따른 base critic degradation을 줄였는가?
Q2. UC critic이 teacher-on collapse도 줄였는가?
Q3. best kappa는 무엇인가?
Q4. remaining failure가 critic learning 문제인가, gate 문제인가?
Q5. Phase 2 full sweep을 진행할 가치가 있는가?

- Draft case: **D/A-fail** (see experiment brief)
- Best kappa by λ=0 scaling gap: **0.0**
- λ=0 mean gap κ=0: 10.0 → best κ: 10.0
- λ=0.5 mean gap κ=0: -37.5 → best κ: -37.5

**Q1:** NO / weak — base scaling still poor
**Q2:** NO / weak — teacher-on still collapses
**Q3:** best kappa ≈ **0.0**
**Q4:** likely correlated-error / gate; not fixed by UC alone
**Q5:** NO — do not auto-expand; diagnose further first

## 7. Plots

Generate after completes:
```bash
python scripts/plot_uc_critic_phase1_curves.py
```

## Notes

- Manifest: `manifests/capo_antmaze_uncertainty_critic_seed0.jsonl`
- Results: `results_jax_sweeps/capo_antmaze_uncertainty_critic_seed0/`
- Refresh: `python scripts/update_uc_critic_phase1_summary.py`
