#!/usr/bin/env python3
"""Regenerate shared host/sweeps/<sweep>/sweep_summary.md (lean)."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9))

ENV_ORDER = [
    ("hopper-medium-v2", "hopper-medium"),
    ("hopper-medium-expert-v2", "hopper-medium-expert"),
    ("hopper-medium-replay-v2", "hopper-medium-replay"),
    ("halfcheetah-medium-v2", "halfcheetah-medium"),
    ("halfcheetah-medium-expert-v2", "halfcheetah-medium-expert"),
    ("halfcheetah-medium-replay-v2", "halfcheetah-medium-replay"),
    ("walker2d-medium-v2", "walker2d-medium"),
    ("walker2d-medium-expert-v2", "walker2d-medium-expert"),
    ("walker2d-medium-replay-v2", "walker2d-medium-replay"),
    ("antmaze-umaze-v2", "antmaze-umaze"),
    ("antmaze-umaze-diverse-v2", "antmaze-umaze-diverse"),
    ("antmaze-medium-play-v2", "antmaze-medium-play"),
    ("antmaze-medium-diverse-v2", "antmaze-medium-diverse"),
    ("antmaze-large-play-v2", "antmaze-large-play"),
    ("antmaze-large-diverse-v2", "antmaze-large-diverse"),
]

# Seed-0 TD3+BC baselines used in training curves (n_critics=4).
BASELINE_N4: Dict[str, float] = {
    "hopper-medium-v2": 59.9740,
    "hopper-medium-expert-v2": 111.7949,
    "hopper-medium-replay-v2": 23.3878,
    "halfcheetah-medium-v2": 48.5142,
    "halfcheetah-medium-expert-v2": 93.3342,
    "halfcheetah-medium-replay-v2": 46.1640,
    "walker2d-medium-v2": 84.7872,
    "walker2d-medium-expert-v2": 110.8686,
    "walker2d-medium-replay-v2": 84.4880,
}


def load_baseline_n2() -> Dict[str, float]:
    """Prefer local baseline_n2 finals when present; else empty."""
    out: Dict[str, float] = {}
    root = ROOT / "results" / "td3_bc"
    if not root.exists():
        return out
    for p in root.rglob("summary.json"):
        if "baseline_n2" not in str(p):
            continue
        try:
            d = json.loads(p.read_text())
            fe = d.get("final_eval") or {}
            score = fe.get("student_d4rl_score", fe.get("d4rl_score", fe.get("base_d4rl_score")))
            env = d.get("env")
            if env and score is not None:
                out[env] = float(score)
        except Exception:
            continue
    return out


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{nd}f}"


def fmt_delta(x):
    if x is None:
        return "—"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.2f}"


def final_score(summary):
    fe = summary.get("final_eval") or {}
    for k in ("student_d4rl_score", "d4rl_score", "learn_d4rl_score", "base_d4rl_score"):
        if fe.get(k) is not None:
            return float(fe[k])
    return None


def parse_run(run_dir: Path):
    cfg_path = run_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    rid = run_dir.name
    env = cfg.get("env")
    lt = cfg.get("lambda_T")
    period = cfg.get("capo_period")
    margin = cfg.get("replace_cert_margin")
    stale = cfg.get("stale_incumbent_action")
    if lt is None:
        m = re.search(r"_lt(0p5|0|1)_", rid)
        if m:
            lt = {"0": 0.0, "0p5": 0.5, "1": 1.0}[m.group(1)]
    if period is None:
        m = re.search(r"_p(50k|100k)_", rid)
        if m:
            period = 50000 if m.group(1) == "50k" else 100000
    if margin is None:
        if "_mm1e3_" in rid:
            margin = -0.001
        elif "_m1e3_" in rid:
            margin = 0.001
        elif "_m0_" in rid:
            margin = 0.0
    if stale is None:
        for s in ("replace_new", "quarantine", "disable"):
            if rid.endswith("_" + s):
                stale = s
                break
    return {
        "run_id": rid,
        "env": env,
        "lambda_T": float(lt) if lt is not None else None,
        "period": int(period) if period is not None else None,
        "margin": float(margin) if margin is not None else None,
        "stale": stale,
    }


def cfg_short(r):
    lt = r.get("lambda_T")
    per = r.get("period")
    marg = r.get("margin")
    lt_s = {0.5: "0.5", 1.0: "1", 0.0: "0"}.get(lt, str(lt))
    per_s = {50000: "50k", 100000: "100k"}.get(per, str(per))
    if marg == 0.0:
        m_s = "0"
    elif marg == 0.001:
        m_s = "1e-3"
    elif marg == -0.001:
        m_s = "−1e-3"
    else:
        m_s = str(marg)
    return f"λ{lt_s} / p{per_s} / m{m_s}"


def live_trainers(results_root: Path, active_ids: set[str]):
    live = []
    for ent in Path("/proc").iterdir():
        if not ent.name.isdigit():
            continue
        try:
            cmd = (ent / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except Exception:
            continue
        if "run_capo_jax.py" not in cmd or results_root.name not in cmd:
            continue
        m = re.search(r"/(fast_[A-Za-z0-9_]+)/", cmd)
        if not m:
            continue
        rid = m.group(1)
        if active_ids and rid not in active_ids:
            continue
        hb = results_root / rid / "heartbeat.json"
        step = None
        if hb.exists():
            step = json.loads(hb.read_text()).get("step")
        live.append((rid, step))
    return live


def build_markdown(
    *,
    host_alias: str,
    sweep_name: str,
    results_root: Path,
    active_manifest: Path,
    note: str,
) -> str:
    now = datetime.now(KST)
    active_rows = [json.loads(l) for l in active_manifest.read_text().splitlines() if l.strip()]
    active_ids = {r["run_id"] for r in active_rows}
    active_by_env = defaultdict(list)
    for r in active_rows:
        active_by_env[r["config"]["env"]].append(r["run_id"])

    active_done = []
    for rid in sorted(active_ids):
        d = results_root / rid
        sp = d / "summary.json"
        if not sp.exists():
            continue
        summary = json.loads(sp.read_text())
        final = final_score(summary)
        if final is None:
            continue
        meta = parse_run(d)
        if meta["env"] is None:
            meta["env"] = next(
                (e for e, ids in active_by_env.items() if rid in ids), None
            )
        meta["final"] = final
        active_done.append(meta)

    n_complete = len(active_done)
    n_total = len(active_ids)
    live = live_trainers(results_root, active_ids)
    baseline_n2 = load_baseline_n2()

    walls = []
    for rid in active_ids:
        sp = results_root / rid / "summary.json"
        if sp.exists():
            try:
                walls.append(json.loads(sp.read_text())["elapsed_sec"])
            except Exception:
                pass
    eta_note = ""
    if walls:
        med = statistics.median(walls)
        partial = sum((s or 0) / 1e6 for _, s in live)
        remain = n_total - n_complete - partial
        eta_h = (remain / 2) * med / 3600
        finish = now + timedelta(hours=eta_h)
        eta_note = f"ETA ~{eta_h:.1f}h (≈{finish.strftime('%m-%d %H:%M')} KST)"

    best_by_env = {}
    for r in active_done:
        env = r["env"]
        if env is None:
            continue
        if env not in best_by_env or r["final"] > best_by_env[env]["final"]:
            best_by_env[env] = r

    lines = []
    lines.append(f"# CAPO stability — `{sweep_name}`")
    lines.append("")
    lines.append(
        f"Updated: {now.strftime('%Y-%m-%d %H:%M')} KST · `{host_alias}` · "
        f"**{n_complete}/{n_total}**"
        + (f" · {eta_note}" if eta_note else "")
    )
    lines.append("")
    lines.append(note)
    lines.append("")
    lines.append(
        "Shared: `host/sweeps/"
        + sweep_name
        + "/sweep_summary.md` · local runs: `results_jax_sweeps/"
        + sweep_name
        + "/`"
    )
    lines.append("")

    # ---- headline table ----
    lines.append("## Best vs baseline (per env)")
    lines.append("")
    lines.append(
        "Best = highest final `student_d4rl_score` among **active** completes. "
        "Baseline = local seed0 TD3+BC `baseline_n2` when available, else n4 reference."
    )
    lines.append("")
    lines.append(
        "| env | done | best | baseline | Δ | setting | run |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | --- | --- |")
    for env, lab in ENV_ORDER:
        if env not in active_by_env:
            continue
        total = len(active_by_env[env])
        done = sum(1 for rid in active_by_env[env] if (results_root / rid / "summary.json").exists())
        best = best_by_env.get(env)
        base = baseline_n2.get(env)
        base_src = "n2"
        if base is None:
            base = BASELINE_N4.get(env)
            base_src = "n4"
        if best is None:
            lines.append(
                f"| {lab} | {done}/{total} | — | {fmt(base)} ({base_src}) | — | — | — |"
            )
            continue
        delta = best["final"] - base if base is not None else None
        lines.append(
            f"| {lab} | {done}/{total} | **{fmt(best['final'])}** | "
            f"{fmt(base)} ({base_src}) | {fmt_delta(delta)} | "
            f"{cfg_short(best)} | `{best['run_id']}` |"
        )
    lines.append("")

    if live:
        lines.append("Live: " + ", ".join(
            f"`{rid}`" + (f"@{step//1000}k" if step else "") for rid, step in live
        ))
        lines.append("")

    # ---- compact margin effect ----
    lines.append("## Margin effect (active completes)")
    lines.append("")
    lines.append("| margin | n | mean | min | max |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for marg in (0.0, 0.001, -0.001):
        xs = [r["final"] for r in active_done if r["margin"] == marg]
        if not xs:
            lines.append(f"| {marg} | 0 | — | — | — |")
            continue
        lines.append(
            f"| {marg} | {len(xs)} | {fmt(statistics.mean(xs))} | "
            f"{fmt(min(xs))} | {fmt(max(xs))} |"
        )
    lines.append("")

    # ---- per-env top3 only ----
    lines.append("## Top-3 per env (active)")
    lines.append("")
    for env, lab in ENV_ORDER:
        if env not in active_by_env:
            continue
        rows = [r for r in active_done if r["env"] == env]
        if not rows:
            lines.append(f"**{lab}** — no completes yet.")
            lines.append("")
            continue
        ranked = sorted(rows, key=lambda r: -r["final"])[:3]
        bits = []
        for i, r in enumerate(ranked, 1):
            bits.append(f"{i}. **{fmt(r['final'])}** ({cfg_short(r)})")
        lines.append(f"**{lab}** — " + " · ".join(bits))
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Final @1M from `summary.json`; Δ = best − baseline.")
    lines.append("- Active: replace_new only, margins `{0, 1e-3, −1e-3}`.")
    lines.append("- Pull/rebase before editing this file; do not commit checkpoints.")
    lines.append("")
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="choi")
    p.add_argument("--sweep", default="capo_stability_seed0_fast_n2")
    p.add_argument("--results_root", default=None)
    p.add_argument(
        "--manifest",
        default="manifests/capo_stability_seed0_fast_n2_replace_new_mgrid.jsonl",
    )
    p.add_argument(
        "--note",
        default=(
            "Active: λ_T∈{0.5,1}, period∈{50k,100k}, margin∈{0,1e-3,−1e-3}, "
            "stale=replace_new, n_critics=2."
        ),
    )
    p.add_argument("--also-local-pointer", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    results_root = (
        Path(args.results_root)
        if args.results_root
        else ROOT / "results_jax_sweeps" / args.sweep
    )
    if not results_root.is_absolute():
        results_root = ROOT / results_root
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = ROOT / manifest

    out_dir = ROOT / "host" / "sweeps" / args.sweep
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sweep_summary.md"
    text = build_markdown(
        host_alias=args.host,
        sweep_name=args.sweep,
        results_root=results_root,
        active_manifest=manifest,
        note=args.note,
    )
    out_path.write_text(text)
    print(f"wrote {out_path.relative_to(ROOT)} ({text.count(chr(10))+1} lines)")

    (out_dir / "README.md").write_text(
        f"# `{args.sweep}`\n\n"
        "Shared by all hosts: **`sweep_summary.md`**\n\n"
        f"```bash\npython scripts/update_stability_sweep_summary.py --host <alias> --sweep {args.sweep}\n```\n"
    )

    if args.also_local_pointer:
        local = results_root / "sweep_summary.md"
        local.write_text(
            "# Moved\n\n"
            f"**[`host/sweeps/{args.sweep}/sweep_summary.md`]"
            f"(../../../host/sweeps/{args.sweep}/sweep_summary.md)**\n"
        )


if __name__ == "__main__":
    main()
