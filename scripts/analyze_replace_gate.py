#!/usr/bin/env python3
"""Analyze incumbent–challenger replacement gate from capo_refresh.jsonl."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=str)
    args = p.parse_args()
    path = Path(args.run_dir) / "capo_refresh.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    print(f"refreshes: {len(rows)}")
    print("decisions:", dict(Counter(r.get("replacement_decision") for r in rows)))

    # Replacement accuracy among times we had both old and new paired gains
    both = [
        r
        for r in rows
        if r.get("paired_delta_d4rl_old") is not None
        and r.get("paired_delta_d4rl_new") is not None
        and r.get("old_to_new_replace_cert") is not None
    ]
    if both:
        c_on = np.array([float(r["old_to_new_replace_cert"]) for r in both])
        d_old = np.array([float(r["paired_delta_d4rl_old"]) for r in both])
        d_new = np.array([float(r["paired_delta_d4rl_new"]) for r in both])
        pos = c_on > 0
        if pos.any():
            acc = float(np.mean(d_new[pos] > d_old[pos]))
            print(f"Pr(ΔJ_new>ΔJ_old | C_O→N>0) = {acc:.3f}  (n={int(pos.sum())})")
        neg = c_on <= 0
        if neg.any():
            keep_ok = float(np.mean(d_old[neg] >= d_new[neg]))
            print(f"Pr(ΔJ_old≥ΔJ_new | C_O→N≤0) = {keep_ok:.3f}  (n={int(neg.sum())})")
        regrets = [float(r["replace_regret_d4rl"]) for r in both if r.get("replace_regret_d4rl") is not None]
        if regrets:
            print(f"mean replace_regret_d4rl = {np.mean(regrets):.3f}")

    print("\nstep  decision  C_SN  C_SO  C_ON  ΔJ_old  ΔJ_new  oracle  regret")
    for r in rows:
        print(
            f"{r.get('refresh_step'):7}  {r.get('replacement_decision'):16}  "
            f"{_f(r.get('student_to_new_cert'))}  {_f(r.get('student_to_old_cert'))}  "
            f"{_f(r.get('old_to_new_replace_cert'))}  "
            f"{_f(r.get('paired_delta_d4rl_old'))}  {_f(r.get('paired_delta_d4rl_new'))}  "
            f"{r.get('oracle_best')}  {_f(r.get('replace_regret_d4rl'))}"
        )


def _f(x):
    if x is None:
        return "   nan"
    return f"{float(x):+6.2f}"


if __name__ == "__main__":
    main()
