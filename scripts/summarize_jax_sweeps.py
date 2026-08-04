#!/usr/bin/env python3
"""Summarize results_jax_sweeps finals (student/teacher/distilled vs baseline) and replot 1M curves."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_training_curve import (  # noqa: E402
    _DEFAULT_BASELINE_FINALS,
    _load_baseline_final,
    _load_distill_series,
    load_metrics,
    plot_run,
)


def _fmt(v: Optional[float], nd: int = 1) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return f"{v:.{nd}f}"


def _final_at_step(rows: List[Dict[str, Any]], key: str, step: int) -> Optional[float]:
    for r in rows:
        if int(r.get("step", -1)) != step:
            continue
        v = r.get(key)
        if v is None:
            return None
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None
        if math.isnan(fv) or math.isinf(fv):
            return None
        return fv
    return None


def _score_at_step(rows: List[Dict[str, Any]], step: int) -> Optional[float]:
    for key in ("student_d4rl_score", "d4rl_score", "student_score"):
        v = _final_at_step(rows, key, step)
        if v is not None:
            return v
    return None


def _teacher_at_step(rows: List[Dict[str, Any]], step: int) -> Optional[float]:
    for key in ("teacher_d4rl_score", "active_teacher_score"):
        v = _final_at_step(rows, key, step)
        if v is not None:
            return v
    return None


def _last_teacher(rows: List[Dict[str, Any]], max_step: int) -> Optional[float]:
    """Latest non-null teacher eval at or before max_step (teacher may be off at 1M)."""
    last: Optional[float] = None
    for r in rows:
        st = int(r.get("step", -1))
        if st < 0 or st > max_step:
            continue
        for key in ("teacher_d4rl_score", "active_teacher_score"):
            v = r.get(key)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(fv) or math.isinf(fv):
                continue
            last = fv
            break
    return last


def _distill_final(run_dir: Path, step: int = 1_000_000) -> Optional[float]:
    xs, ys = _load_distill_series(run_dir)
    if not xs:
        return None
    for x, y in zip(xs, ys):
        if int(x) == step:
            return float(y)
    # fall back to last point if it reached end_step
    if int(xs[-1]) >= step:
        return float(ys[-1])
    return None


def collect_runs(root: Path, min_step: int = 1_000_000) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for metrics_path in sorted(root.rglob("metrics.jsonl")):
        run_dir = metrics_path.parent
        if ".runtime_overlay" in run_dir.parts:
            continue
        rows = load_metrics(metrics_path)
        if not rows:
            continue
        last_step = int(rows[-1].get("step", 0))
        cfg: Dict[str, Any] = {}
        cfg_path = run_dir / "config.json"
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text())
        env = str(cfg.get("env") or "")
        baseline = _load_baseline_final(env, run_dir)
        if baseline is None:
            baseline = _DEFAULT_BASELINE_FINALS.get(env)
        done = last_step >= min_step
        target_step = min_step if done else last_step
        student = _score_at_step(rows, target_step)
        teacher_at = _teacher_at_step(rows, target_step)
        teacher = teacher_at if teacher_at is not None else _last_teacher(rows, target_step)
        distilled = _distill_final(run_dir, min_step) if done else None
        if distilled is None and done:
            distilled = _distill_final(run_dir, last_step)
        rec = {
            "sweep": run_dir.parent.name,
            "run": run_dir.name,
            "run_dir": str(run_dir),
            "env": env,
            "seed": cfg.get("seed"),
            "use_capo": cfg.get("use_capo"),
            "n_critics": cfg.get("n_critics"),
            "stale_incumbent_action": cfg.get("stale_incumbent_action"),
            "lambda_T": cfg.get("lambda_T"),
            "capo_period": cfg.get("capo_period"),
            "replace_cert_margin": cfg.get("replace_cert_margin"),
            "step": last_step,
            "done_1m": done,
            "student_final": student,
            "teacher_final": teacher,
            "teacher_at_final_step": teacher_at,
            "distilled_final": distilled,
            "baseline_final": baseline,
            "delta_student": (student - baseline) if (student is not None and baseline is not None) else None,
            "delta_teacher": (teacher - baseline) if (teacher is not None and baseline is not None) else None,
            "delta_distilled": (distilled - baseline)
            if (distilled is not None and baseline is not None)
            else None,
            "curve": str(run_dir / "training_curve.png") if done else None,
        }
        out.append(rec)
    return out


def write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    lines = [
        "# results_jax_sweeps · final scores vs baseline",
        "",
        "Final = D4RL at 1M (incomplete runs show latest step). "
        "Teacher = score at final step if active, else last logged teacher eval (`—` if never). "
        "Distilled from `posthoc_student_distill(_jax)/distill_metrics.jsonl` when present.",
        "",
        "| sweep | run | env | step | student | teacher | distilled | baseline | Δstu | curve |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        curve = f"[png]({Path(r['curve']).name})" if r.get("curve") and r["done_1m"] else "—"
        # relative link from overview file next to sweep folders is awkward; use run-relative
        if r.get("curve") and r["done_1m"]:
            rel = Path(r["run_dir"]).name + "/training_curve.png"
            if r["sweep"] != path.parent.name:
                rel = f"{r['sweep']}/{r['run']}/training_curve.png"
            else:
                rel = f"{r['run']}/training_curve.png"
            # overview lives at results_jax_sweeps/OVERVIEW_FINALS.md
            rel = f"{r['sweep']}/{r['run']}/training_curve.png"
            curve = f"[png]({rel})"
        lines.append(
            "| {sweep} | `{run}` | {env} | {step} | {stu} | {tea} | {dis} | {base} | {dstu} | {curve} |".format(
                sweep=r["sweep"],
                run=r["run"],
                env=r["env"] or "?",
                step=r["step"],
                stu=_fmt(r["student_final"]),
                tea=_fmt(r["teacher_final"]),
                dis=_fmt(r["distilled_final"]),
                base=_fmt(r["baseline_final"]),
                dstu=_fmt(r["delta_student"]),
                curve=curve,
            )
        )
    n_done = sum(1 for r in rows if r["done_1m"])
    n_dist = sum(1 for r in rows if r["distilled_final"] is not None)
    lines.extend(
        [
            "",
            f"- Runs: {len(rows)} · done@1M: {n_done} · with distilled final: {n_dist}",
            "- Baseline source: `results/baseline_td3bc_seed0.json`",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def write_html(rows: List[Dict[str, Any]], path: Path, root: Path) -> None:
    done = [r for r in rows if r["done_1m"]]
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>jax sweeps finals</title>",
        "<style>",
        "body{font-family:ui-sans-serif,system-ui,sans-serif;margin:24px;color:#111}",
        "table{border-collapse:collapse;width:100%;font-size:13px}",
        "th,td{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left}",
        "th{position:sticky;top:0;background:#f7f7f7}",
        "td.num{text-align:right;font-variant-numeric:tabular-nums}",
        ".pos{color:#0a7a32}.neg{color:#b42318}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px;margin-top:24px}",
        ".card{border:1px solid #e5e5e5;padding:8px}",
        ".card img{width:100%;height:auto}",
        ".card h3{font-size:12px;margin:0 0 8px;font-weight:600}",
        "</style></head><body>",
        "<h1>results_jax_sweeps · finals vs baseline</h1>",
        f"<p>{len(rows)} runs · {len(done)} finished 1M · "
        f"{sum(1 for r in rows if r['distilled_final'] is not None)} with distilled</p>",
        "<table><thead><tr>",
        "<th>sweep</th><th>run</th><th>env</th><th>step</th>",
        "<th>student</th><th>teacher</th><th>distilled</th><th>baseline</th><th>Δstu</th>",
        "</tr></thead><tbody>",
    ]
    for r in rows:
        d = r["delta_student"]
        dcls = "pos" if (d is not None and d > 0) else ("neg" if (d is not None and d < 0) else "")
        parts.append(
            "<tr>"
            f"<td>{r['sweep']}</td><td><code>{r['run']}</code></td><td>{r['env']}</td>"
            f"<td class='num'>{r['step']}</td>"
            f"<td class='num'>{_fmt(r['student_final'])}</td>"
            f"<td class='num'>{_fmt(r['teacher_final'])}</td>"
            f"<td class='num'>{_fmt(r['distilled_final'])}</td>"
            f"<td class='num'>{_fmt(r['baseline_final'])}</td>"
            f"<td class='num {dcls}'>{_fmt(r['delta_student'])}</td>"
            "</tr>"
        )
    parts.append("</tbody></table><h2>Training curves (1M done)</h2><div class='grid'>")
    for r in done:
        png = Path(r["run_dir"]) / "training_curve.png"
        if not png.is_file():
            continue
        rel = png.relative_to(path.parent)
        parts.append(
            f"<div class='card'><h3>{r['sweep']} / {r['run']} · {r['env']}</h3>"
            f"<img src='{rel.as_posix()}' alt='{r['run']}' loading='lazy'></div>"
        )
    parts.append("</div></body></html>")
    path.write_text("\n".join(parts))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=str,
        default=str(ROOT / "results_jax_sweeps"),
        help="Sweep root directory",
    )
    p.add_argument("--min_step", type=int, default=1_000_000)
    p.add_argument("--no_plot", action="store_true")
    args = p.parse_args()
    root = Path(args.root).resolve()
    rows = collect_runs(root, min_step=args.min_step)
    overview_json = root / "overview_finals.json"
    overview_md = root / "OVERVIEW_FINALS.md"
    overview_html = root / "OVERVIEW_FINALS.html"
    overview_json.write_text(json.dumps(rows, indent=2))
    write_markdown(rows, overview_md)
    if not args.no_plot:
        for r in rows:
            if not r["done_1m"]:
                continue
            run_dir = Path(r["run_dir"])
            try:
                plot_run(run_dir, baseline_final=r.get("baseline_final"))
                print(f"[plot] {run_dir / 'training_curve.png'}")
            except Exception as e:
                print(f"[plot] fail {run_dir}: {e}", file=sys.stderr)
    write_html(rows, overview_html, root)
    print(f"[summary] {overview_json}")
    print(f"[summary] {overview_md}")
    print(f"[summary] {overview_html}")
    print(
        f"[summary] runs={len(rows)} done1M={sum(1 for r in rows if r['done_1m'])} "
        f"distill={sum(1 for r in rows if r['distilled_final'] is not None)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
