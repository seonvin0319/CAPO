#!/usr/bin/env python3
"""Certificate vs paired ΔJ calibration from capo_refresh.jsonl (+ optional control)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("capo_run_dir", type=str)
    p.add_argument("--control_run_dir", type=str, default=None)
    args = p.parse_args()
    run = Path(args.capo_run_dir)
    rows = load_jsonl(run / "capo_refresh.jsonl")
    accepted = [r for r in rows if r.get("accepted_by_cert")]
    print(f"refresh events: {len(rows)}  accepted_by_cert: {len(accepted)}")

    deltas = []
    certs = []
    labels = {"True": 0, "False": 0, "uncertain": 0}
    for r in accepted:
        c = r.get("accepted_cert", r.get("accepted_cert_per_step", [None])[-1] if r.get("accepted_cert_per_step") else None)
        d = r.get("paired_delta_d4rl", r.get("paired_delta_mean"))
        if c is None or d is None:
            continue
        certs.append(float(c))
        deltas.append(float(d))
        lab = r.get("teacher_better_by_eval")
        labels[str(lab)] = labels.get(str(lab), 0) + 1

    certs = np.asarray(certs, dtype=float)
    deltas = np.asarray(deltas, dtype=float)
    if len(deltas) == 0:
        print("no paired accepted refreshes yet")
    else:
        pos = certs > 0
        print(f"Pr(ΔJ>0 | C_accepted>0) = {(deltas[pos] > 0).mean() if pos.any() else float('nan'):.3f}")
        print(f"E[ΔJ | C_accepted>0]     = {deltas[pos].mean() if pos.any() else float('nan'):.3f}")
        if len(deltas) > 1 and certs.std() > 0 and deltas.std() > 0:
            print(f"Corr(C, ΔJ)             = {float(np.corrcoef(certs, deltas)[0, 1]):.3f}")
        print(
            "teacher_better_by_eval counts:",
            {k: labels.get(k, 0) for k in ("True", "False", "uncertain")},
        )
        print(
            f"mean paired ΔJ = {deltas.mean():.3f}  "
            f"SE = {deltas.std(ddof=1)/np.sqrt(len(deltas)):.3f}"
        )

    ladder = load_jsonl(run / "capo_ladder.jsonl")
    if ladder:
        ns = [r["selected_n"] for r in ladder]
        taus = [t for r in ladder for t in r.get("selected_taus", [])]
        print(f"Pr(N*=1)={np.mean([n==1 for n in ns]):.3f}  Pr(N*=2)={np.mean([n==2 for n in ns]):.3f}")
        if taus:
            all_tau = [
                float(c["tau"])
                for r in ladder
                for rec in r.get("records", [])
                for c in rec.get("candidates", [])
            ]
            tau_max = max(all_tau) if all_tau else max(taus)
            print(f"Pr(τ*=τ_max={tau_max:g})={np.mean([abs(float(t)-tau_max)<1e-12 for t in taus]):.3f}")

    if args.control_run_dir:
        cdir = Path(args.control_run_dir)
        # Compare final / best student curves if both finished.
        def best_score(d: Path):
            m = d / "metrics.jsonl"
            if not m.exists():
                return None
            rows = load_jsonl(m)
            key = "student_d4rl_score" if "student_d4rl_score" in (rows[-1] if rows else {}) else "d4rl_score"
            vals = [(r["step"], r.get(key)) for r in rows if r.get(key) is not None]
            if not vals:
                return None
            best = max(vals, key=lambda x: x[1])
            final = vals[-1]
            return best, final

        cb, cf = best_score(cdir) or (None, None), None
        mb = best_score(run)
        if mb and best_score(cdir):
            (cb_step, cb_sc), (cf_step, cf_sc) = best_score(cdir)
            (mb_step, mb_sc), (mf_step, mf_sc) = mb
            print(
                f"student CAPO best={mb_sc:.2f}@{mb_step} final={mf_sc:.2f} | "
                f"control best={cb_sc:.2f}@{cb_step} final={cf_sc:.2f} | "
                f"Δfinal={mf_sc-cf_sc:.2f}"
            )


if __name__ == "__main__":
    main()
