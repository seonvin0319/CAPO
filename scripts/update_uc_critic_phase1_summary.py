#!/usr/bin/env python3
"""Regenerate Phase-1 UC-critic AntMaze sweep_summary.md from completed runs."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9))
SWEEP = "capo_antmaze_uncertainty_critic_seed0"
MANIFEST = ROOT / "manifests" / f"{SWEEP}.jsonl"
RESULTS = ROOT / "results_jax_sweeps" / SWEEP
OUT = ROOT / "host" / "sweeps" / SWEEP / "sweep_summary.md"
LEGACY = {
    4: ROOT / "results_jax_sweeps" / "capo_stability_seed0_fast_antmaze",
    8: ROOT / "results_jax_sweeps" / "capo_stability_seed0_fast_antmaze_n8",
    16: ROOT / "results_jax_sweeps" / "capo_stability_seed0_fast_antmaze_n16",
}
ENV_SHORT = {
    "antmaze-umaze-v2": "umaze",
    "antmaze-umaze-diverse-v2": "umaze-diverse",
}


def _score(fe: Dict[str, Any]) -> Optional[float]:
    for k in ("student_d4rl_score", "d4rl_score", "base_d4rl_score"):
        if fe.get(k) is not None:
            return float(fe[k])
    return None


def load_complete(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not root.exists():
        return rows
    for d in sorted(root.iterdir()):
        if not d.is_dir() or "QUARANTINE" in d.name:
            continue
        sj = d / "summary.json"
        if not sj.exists():
            continue
        payload = json.loads(sj.read_text())
        fe = payload.get("final_eval") or {}
        if payload.get("status") != "complete":
            continue
        if int(fe.get("step") or 0) < 1_000_000:
            continue
        sc = _score(fe)
        if sc is None:
            continue
        cfg = {}
        rc = d / "resolved_config.yaml"
        cj = d / "config.json"
        if cj.exists():
            cfg = json.loads(cj.read_text())
        rows.append(
            {
                "run_id": d.name,
                "env": payload.get("env") or cfg.get("env"),
                "score": sc,
                "fe": fe,
                "cfg": cfg,
                "n_critics": int(cfg.get("n_critics") or 0),
                "lambda_T": float(cfg.get("lambda_T") or 0.0),
                "kappa": float(cfg.get("critic_uncertainty_kappa") or 0.0),
            }
        )
    return rows


def legacy_margin0() -> Dict[Tuple[str, int, float], float]:
    """env_short, n_critics, lambda_T -> score for margin=0 replace_new."""
    out: Dict[Tuple[str, int, float], float] = {}
    for n, root in LEGACY.items():
        if not root.exists():
            continue
        for d in root.iterdir():
            if not d.is_dir() or "QUARANTINE" in d.name:
                continue
            if "_m0_" not in d.name or "replace_new" not in d.name:
                continue
            sj = d / "summary.json"
            if not sj.exists():
                continue
            payload = json.loads(sj.read_text())
            fe = payload.get("final_eval") or {}
            sc = _score(fe)
            if sc is None:
                continue
            env = payload.get("env")
            m = re.search(r"_lt([0-9p]+)_", d.name)
            if not m or env not in ENV_SHORT:
                continue
            lt = float(m.group(1).replace("p", "."))
            out[(ENV_SHORT[env], n, lt)] = sc
    return out


def fmt(x: Optional[float], digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "…"
    return f"{x:.{digits}f}"


def mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", default=True)
    args = parser.parse_args()

    rows_m = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    total = len(rows_m)
    done_rows = load_complete(RESULTS)
    done = len(done_rows)
    by = {
        (ENV_SHORT.get(r["env"], r["env"]), r["kappa"], r["n_critics"], r["lambda_T"]): r
        for r in done_rows
        if r["env"] in ENV_SHORT
    }
    legacy = legacy_margin0()
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")

    def cell(env: str, kappa: float, n: int, lt: float) -> Optional[float]:
        r = by.get((env, kappa, n, lt))
        return None if r is None else r["score"]

    lines: List[str] = []
    lines += [
        f"# Sweep summary — `{SWEEP}` (Phase 1 UC critic)",
        "",
        f"Updated: **{now}**",
        "Host: **ext_csh**",
        f"Progress: **{done}/{total}** complete",
        "",
        "## 1. Experiment setup",
        "",
        "- Change isolated: `use_uncertainty_weighted_critic=True` + `critic_uncertainty_kappa`",
        "- Envs: `antmaze-umaze-v2`, `antmaze-umaze-diverse-v2`",
        "- Axes: `n_critics∈{4,8,16}` × `λ_T∈{0,0.5}` × `κ∈{0,0.5,1,2}`",
        "- Fixed: seed0, `stale=replace_new`, period=100k, margin=0, 1M steps",
        "- `κ=0` uses weighted path with weight≡1 (bit-identical to original TD loss per unit test)",
        "- Phase 2: **not auto-started** — wait for Phase 1 answers below",
        "",
        "## 2. λ_T = 0 paired results",
        "",
        "| env | kappa | n4 | n8 | n16 | n8−n4 | n16−n4 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for env in ("umaze", "umaze-diverse"):
        for kappa in (0.0, 0.5, 1.0, 2.0):
            n4, n8, n16 = cell(env, kappa, 4, 0.0), cell(env, kappa, 8, 0.0), cell(env, kappa, 16, 0.0)
            d8 = None if n4 is None or n8 is None else n8 - n4
            d16 = None if n4 is None or n16 is None else n16 - n4
            lines.append(
                f"| {env} | {kappa:g} | {fmt(n4)} | {fmt(n8)} | {fmt(n16)} | {fmt(d8)} | {fmt(d16)} |"
            )

    lines += [
        "",
        "### Legacy reference (prior antmaze, margin=0)",
        "",
        "| env | n4 | n8 | n16 | n8−n4 | n16−n4 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for env in ("umaze", "umaze-diverse"):
        n4 = legacy.get((env, 4, 0.0))
        n8 = legacy.get((env, 8, 0.0))
        n16 = legacy.get((env, 16, 0.0))
        d8 = None if n4 is None or n8 is None else n8 - n4
        d16 = None if n4 is None or n16 is None else n16 - n4
        lines.append(f"| {env} | {fmt(n4)} | {fmt(n8)} | {fmt(n16)} | {fmt(d8)} | {fmt(d16)} |")

    lines += [
        "",
        "## 3. λ_T = 0.5 paired results",
        "",
        "| env | kappa | n4 | n8 | n16 | n8−n4 | n16−n4 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for env in ("umaze", "umaze-diverse"):
        for kappa in (0.0, 0.5, 1.0, 2.0):
            n4, n8, n16 = cell(env, kappa, 4, 0.5), cell(env, kappa, 8, 0.5), cell(env, kappa, 16, 0.5)
            d8 = None if n4 is None or n8 is None else n8 - n4
            d16 = None if n4 is None or n16 is None else n16 - n4
            lines.append(
                f"| {env} | {kappa:g} | {fmt(n4)} | {fmt(n8)} | {fmt(n16)} | {fmt(d8)} | {fmt(d16)} |"
            )

    lines += [
        "",
        "## 4. Teacher-path diagnostics (λ_T=0.5 completes)",
        "",
        "| n | kappa | env | score | accepted | accepted_cert | replace | unc_mean | w_mean |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(
        [x for x in done_rows if abs(x["lambda_T"] - 0.5) < 1e-9],
        key=lambda x: (x["n_critics"], x["kappa"], x["env"]),
    ):
        fe = r["fe"]
        lines.append(
            "| {n} | {k:g} | {e} | {sc} | {acc} | {cert} | {rep} | {u} | {w} |".format(
                n=r["n_critics"],
                k=r["kappa"],
                e=ENV_SHORT.get(r["env"], r["env"]),
                sc=fmt(r["score"]),
                acc=fmt(fe.get("capo_accepted"), 2),
                cert=fmt(fe.get("capo_accepted_cert"), 4),
                rep=fmt(fe.get("replace_count"), 0),
                u=fmt(fe.get("critic/uncertainty_mean"), 4),
                w=fmt(fe.get("critic/uncertainty_weight_mean"), 4),
            )
        )

    # scaling summary per kappa
    lines += ["", "## 5. n4/n8/n16 scaling (mean over envs)", ""]
    lines += [
        "| lambda_T | kappa | mean(n8−n4) | mean(n16−n4) |",
        "|---:|---:|---:|---:|",
    ]
    for lt in (0.0, 0.5):
        for kappa in (0.0, 0.5, 1.0, 2.0):
            d8s, d16s = [], []
            for env in ("umaze", "umaze-diverse"):
                n4, n8, n16 = cell(env, kappa, 4, lt), cell(env, kappa, 8, lt), cell(env, kappa, 16, lt)
                if n4 is not None and n8 is not None:
                    d8s.append(n8 - n4)
                if n4 is not None and n16 is not None:
                    d16s.append(n16 - n4)
            lines.append(
                f"| {lt:g} | {kappa:g} | {fmt(mean(d8s))} | {fmt(mean(d16s))} |"
            )

    # best kappa heuristic once all λ=0 cells done
    lines += [
        "",
        "## 6. Interpretation (auto draft — fill when complete)",
        "",
        "Q1. UC critic이 n_critics 증가에 따른 base critic degradation을 줄였는가?",
        "Q2. UC critic이 teacher-on collapse도 줄였는가?",
        "Q3. best kappa는 무엇인가?",
        "Q4. remaining failure가 critic learning 문제인가, gate 문제인가?",
        "Q5. Phase 2 full sweep을 진행할 가치가 있는가?",
        "",
    ]
    if done < total:
        lines.append(f"_Incomplete: {done}/{total}. Re-run this script after completes._")
    else:
        # pick best kappa by maximizing mean of (n8-n4 + n16-n4) at λ=0 then check λ=0.5
        best = None
        best_score = -1e18
        for kappa in (0.0, 0.5, 1.0, 2.0):
            gaps = []
            for env in ("umaze", "umaze-diverse"):
                n4, n8, n16 = cell(env, kappa, 4, 0.0), cell(env, kappa, 8, 0.0), cell(env, kappa, 16, 0.0)
                if None in (n4, n8, n16):
                    continue
                gaps.append((n8 - n4) + (n16 - n4))
            if gaps and mean(gaps) is not None and mean(gaps) > best_score:
                best_score = mean(gaps)  # type: ignore
                best = kappa
        # classify
        k0_gaps = []
        kb_gaps = []
        for env in ("umaze", "umaze-diverse"):
            for n in (8, 16):
                a = cell(env, 0.0, 4, 0.0)
                b = cell(env, 0.0, n, 0.0)
                if a is not None and b is not None:
                    k0_gaps.append(b - a)
                if best is not None:
                    a2 = cell(env, best, 4, 0.0)
                    b2 = cell(env, best, n, 0.0)
                    if a2 is not None and b2 is not None:
                        kb_gaps.append(b2 - a2)
        improved_base = (
            best is not None
            and k0_gaps
            and kb_gaps
            and mean(kb_gaps) is not None
            and mean(k0_gaps) is not None
            and mean(kb_gaps) > mean(k0_gaps) + 5  # type: ignore
        )
        t0_gaps, tb_gaps = [], []
        for env in ("umaze", "umaze-diverse"):
            for n in (8, 16):
                a = cell(env, 0.0, 4, 0.5)
                b = cell(env, 0.0, n, 0.5)
                if a is not None and b is not None:
                    t0_gaps.append(b - a)
                if best is not None:
                    a2 = cell(env, best, 4, 0.5)
                    b2 = cell(env, best, n, 0.5)
                    if a2 is not None and b2 is not None:
                        tb_gaps.append(b2 - a2)
        improved_teacher = (
            best is not None
            and t0_gaps
            and tb_gaps
            and mean(tb_gaps) is not None
            and mean(t0_gaps) is not None
            and mean(tb_gaps) > mean(t0_gaps) + 5  # type: ignore
        )
        if improved_base and improved_teacher:
            case = "C"
        elif improved_base and not improved_teacher:
            case = "B"
        elif not improved_base and not improved_teacher:
            case = "D/A-fail"
        else:
            case = "mixed"
        lines += [
            f"- Draft case: **{case}** (see experiment brief)",
            f"- Best kappa by λ=0 scaling gap: **{best}**",
            f"- λ=0 mean gap κ=0: {fmt(mean(k0_gaps))} → best κ: {fmt(mean(kb_gaps))}",
            f"- λ=0.5 mean gap κ=0: {fmt(mean(t0_gaps))} → best κ: {fmt(mean(tb_gaps))}",
            "",
            f"**Q1:** {'YES — base scaling gap improved' if improved_base else 'NO / weak — base scaling still poor'}",
            f"**Q2:** {'YES — teacher-on gap improved' if improved_teacher else 'NO / weak — teacher-on still collapses'}",
            f"**Q3:** best kappa ≈ **{best}**",
            f"**Q4:** {'critic learning (then check gate)' if improved_base and not improved_teacher else ('critic+teacher path both helped' if improved_base and improved_teacher else 'likely correlated-error / gate; not fixed by UC alone')}",
            f"**Q5:** {'YES — proceed Phase 2 with best kappa' if improved_base else 'NO — do not auto-expand; diagnose further first'}",
            "",
        ]

    lines += [
        "## 7. Plots",
        "",
        "Generate after completes:",
        "```bash",
        "python scripts/plot_uc_critic_phase1_curves.py",
        "```",
        "",
        "## Notes",
        "",
        f"- Manifest: `manifests/{SWEEP}.jsonl`",
        f"- Results: `results_jax_sweeps/{SWEEP}/`",
        "- Refresh: `python scripts/update_uc_critic_phase1_summary.py`",
        "",
    ]

    text = "\n".join(lines)
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(text)
        print(f"wrote {OUT} ({done}/{total})")
    else:
        print(text)


if __name__ == "__main__":
    main()
