# CAPO `n_critics` 스케일링 리포트 (개정)

Updated: **2026-08-07**  
Antmaze: **ext_csh** · Locomotion n2: **choi** (108/108, pulled from `origin/main`)

---

## 핵심 수정

이전에 ext_csh의 미완료 `fastn2_*`(9/36, cancelled)로 “n2 + λ_T>0 = 붕괴”라고 말했는데, **choi가 돌린 본 캠페인(108/108)** 에서는 locomotion n2가 n4와 비슷하거나 더 좋습니다.

Antmaze에서 critic를 늘리면 나빠지는 패턴은 **그대로**입니다.

| 요약 | 값 |
|---|---|
| loco best mean Δ (n2 − n4) | **+1.7** |
| loco envs where n2 ≥ n4 | **6/9** |
| antmaze paired mean Δ (n16 − n4) | **−15.0** |

---

## 1. Locomotion — choi n2 vs ext_csh n4

### 데이터 출처

- **choi n2:** `origin/main` → `host/sweeps/capo_stability_seed0_fast_n2/sweep_summary.md` (2026-08-06, **108/108**)
- **ext_csh n4:** `results_jax_sweeps/capo_stability_seed0_fast/` `replace_new` finals

### 그리드 비교

| | choi n2 (완료) | ext_csh n4 (비교 기준) |
|---|---|---|
| cells | **108/108** | replace_new finals (~72) |
| `n_critics` | 2 | 4 |
| λ_T | `{0.5, 1}` (λ=0 제외) | `{0, 0.5, 1}` |
| period | `{50k, 100k}` | 주로 `100k` |
| margin | `{0, +1e-3, −1e-3}` | active emphasis `{0, −1e-3}` |
| envs | medium + expert + replay (9) | 동일 9 env |

> 그리드가 완전 동일하진 않음. choi가 p50k·+1e-3를 더 넓게 탐색 → best-of-grid 비교에 유리할 수 있음.

### Best-per-env (D4RL @1M)

| env | choi n2 | n2 cfg | n4 best | Δ (n2−n4) |
|---|---:|---|---:|---:|
| hopper-medium | 100.20 | λ0.5 / 100k / −1e-3 | 100.27 | −0.07 |
| hopper-medium-expert | 112.40 | λ1 / 100k / +1e-3 | 112.34 | +0.06 |
| hopper-medium-replay | **100.62** | λ0.5 / 100k / 0 | 85.89 | **+14.73** |
| halfcheetah-medium | 52.63 | λ1 / 50k / 0 | 52.19 | +0.44 |
| halfcheetah-medium-expert | 82.36 | λ1 / 50k / +1e-3 | 85.33 | −2.97 |
| halfcheetah-medium-replay | 47.61 | λ1 / 50k / +1e-3 | 49.06 | −1.45 |
| walker2d-medium | 86.99 | λ1 / 50k / −1e-3 | 84.38 | +2.61 |
| walker2d-medium-expert | 115.64 | λ0.5 / 100k / 0 | 114.92 | +0.72 |
| walker2d-medium-replay | 92.89 | λ1 / 50k / +1e-3 | 91.98 | +0.91 |

- mean Δ(n2 − n4 any-λ) = **+1.67**, win **6/9**
- teacher-on(λ>0)만 보면 mean **+2.34**, win **7/9**

### Choi n2 margin aggregate (108 cells)

| margin | n | mean | min | max |
|---|---:|---:|---:|---:|
| 0 | 36 | 68.73 | 1.54 | 115.64 |
| +1e-3 | 36 | 69.72 | 2.30 | 115.03 |
| −1e-3 | 36 | 64.82 | 1.85 | 114.85 |

min≈1.5인 셀이 있음 → n2에서도 일부 collapse는 존재. 다만 **best/평균은 n4와 경쟁력 있음**.

---

## 2. 이전 partial n2 결론이 틀린 이유

| | ext_csh partial (구) | choi full (신) |
|---|---|---|
| 완료 | 9/36 cancelled | **108/108** |
| prefix | `fastn2_*` | `fast_*` mgrid |
| λ grid | `{0, 0.5, 1}` | `{0.5, 1}` only |
| period | 100k only | 50k + 100k |
| margin | `{0, −1e-3}` | `{0, +1e-3, −1e-3}` |
| envs | expert+replay 6 | medium+expert+replay 9 |
| λ>0 hm-exp | 붕괴 (2~35) | **best 112.4** |

**교훈:** cancelled partial로 `n_critics` 효과를 단정하면 안 됨. choi 본캠페인 기준으로는 locomotion에서 n=2가 “cert 깨짐”으로 안 보임.

---

## 3. Antmaze — critic↑ 여전히 해로움 (ext_csh)

Metric: final `student_d4rl_score` @1M · `stale=replace_new` · period=100k · seed 0  
Progress: n4/n8 done · n16 ~38/42

### Best-per-env

| env | n4 | n8 | n16 |
|---|---:|---:|---:|
| umaze | **90** | 60 | 70 |
| umaze-diverse | **100** | 80 | 60 |
| medium / large (4 envs) | 0 | 0 | 0 |

### Paired Δ vs n4 (same λ, margin, env)

| Compare | Cells | Mean Δ | λ=0 Δ | λ>0 Δ | Win/Tie/Lose |
|---|---:|---:|---:|---:|---|
| n8 vs n4 | 42 | **−9.3** | −6.7 | −9.7 | 2 / 29 / 11 |
| n16 vs n4 | 38 | **−15.0** | −10.0 | −15.9 | 2 / 25 / 11 |

### Teacher-path diagnostics (umaze + umaze-diverse, λ_T>0, n=12 each)

| n_critics | Mean score | capo_accepted | accepted_cert | replace_count |
|---:|---:|---:|---:|---:|
| 4 | 59.2 | 0.75 | 0.015 | 9.6 |
| 8 | 30.0 | 1.00 | 0.019 | 10.0 |
| 16 | 16.7 | 1.00 | 0.038 | 10.0 |

score는 59→30→17로 하락하는데 acceptance·cert는 커짐 → 큰 cert 앙상블이 **과신**하는 쪽.

### 메커니즘 (antmaze)

- `split_critics_for_certification=True` → 앞절반 gen / 뒷절반 cert  
  - n=4 → 2+2 · n=8 → 4+4 · n=16 → 8+8 · n=2 → 1+1 (cert std≈0)
- Certificate = `mean_gain − β·std(per-critic ΔQ) − penalties`
- cert critic↑ → ΔQ 합의 → uncertainty↓ → LCB↑ → 나쁜 teacher도 통과 → BC(λ_T)가 sparse antmaze student 오염
- medium/large 0은 n과 무관한 도메인/그리드 한계

---

## 4. 개정 결론

1. **Locomotion (choi n2):** n=2는 n=4 대비 best 기준 동등~소폭 우위. “늘려야 한다”도 “줄이면 망한다”도 아님.
2. **Antmaze (ext_csh):** n=4 ≫ n=8 ≥ n=16. sparse + teacher 경로에서 큰 cert 앙상블이 해로움.
3. **도메인 의존:** loco에서 보인 n2 안정성이 antmaze로 자동 이전되지 않음 (antmaze n2는 choi 체인에 있을 수 있으나 이 리포트엔 미포함).

### Caveat

- best-of-grid 비교는 choi의 더 넓은 (p50k, +1e-3) 탐색에 유리할 수 있음.
- 페어드 셀 전체 평균은 choi raw `summary.json`이 로컬에 없어 미계산 (shared summary의 best/margin aggregate만 사용).

---

## 관련 경로

- Canvas: `~/.cursor/projects/home-ext-csh/canvases/capo-n-critics-report.canvas.tsx`
- Choi n2 summary: `CaPO/host/sweeps/capo_stability_seed0_fast_n2/sweep_summary.md`
- Choi n2 manifest: `CaPO/manifests/capo_stability_seed0_fast_n2_replace_new_mgrid.jsonl`
- Antmaze results: `CaPO/results_jax_sweeps/capo_stability_seed0_fast_antmaze{,_n8,_n16}/`
- n4 loco results: `CaPO/results_jax_sweeps/capo_stability_seed0_fast/`
