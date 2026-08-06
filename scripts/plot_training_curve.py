#!/usr/bin/env python3
"""Plot CAPO training curves from a run_dir's metrics.jsonl."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_metrics(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r.get("step", 0)))
    return rows


def _series(rows: List[Dict[str, Any]], key: str) -> tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    for r in rows:
        if key not in r:
            continue
        v = r[key]
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(fv) or math.isinf(fv):
            continue
        xs.append(int(r["step"]))
        ys.append(fv)
    return xs, ys


def _series_aligned(
    rows: List[Dict[str, Any]], key: str, step_key: str = "step"
) -> tuple[List[int], List[float]]:
    """Like `_series`, but insert NaNs when `key` is missing so line plots break gaps.

    Needed for teacher scores: quarantine / disabled steps omit the metric, and a
    plain connect-the-dots line invents long fake diagonals across those gaps.
    """
    xs: List[int] = []
    ys: List[float] = []
    for r in rows:
        if step_key not in r:
            continue
        xs.append(int(r[step_key]))
        v = r.get(key)
        if v is None:
            ys.append(float("nan"))
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            ys.append(float("nan"))
            continue
        if math.isnan(fv) or math.isinf(fv):
            ys.append(float("nan"))
        else:
            ys.append(fv)
    return xs, ys


def _shared_xticks(x_max: int, step: int = 200_000) -> List[int]:
    """Ticks 0, step, 2*step, ... covering [0, x_max]."""
    if x_max <= 0:
        return [0]
    last = int(math.ceil(x_max / step) * step)
    return list(range(0, last + 1, step))


def _load_refresh_events(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "capo_refresh.jsonl"
    if not path.is_file():
        return []
    events: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def _load_distill_series(run_dir: Path) -> tuple[List[int], List[float]]:
    """Load distilled actor D4RL scores from post-hoc distill metrics if present."""
    candidates = [
        run_dir / "posthoc_student_distill_jax" / "distill_metrics.jsonl",
        run_dir / "posthoc_student_distill" / "distill_metrics.jsonl",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return [], []
    xs: List[int] = []
    ys: List[float] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("distilled_score") is None:
                continue
            try:
                xs.append(int(row["checkpoint_step"]))
                ys.append(float(row["distilled_score"]))
            except (KeyError, TypeError, ValueError):
                continue
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return [xs[i] for i in order], [ys[i] for i in order]


# Seed-0 TD3+BC baseline finals (use_capo=false). Used as horizontal reference.
_DEFAULT_BASELINE_FINALS: Dict[str, float] = {
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


def _load_baseline_final(env: str, run_dir: Path) -> Optional[float]:
    """Resolve baseline final for `env` from local files or built-in table."""
    if not env:
        return None
    candidates = [
        run_dir / "baseline_finals.json",
        run_dir.parent / "baseline_finals.json",
        Path(__file__).resolve().parents[1] / "results" / "baseline_td3bc_seed0.json",
        Path(__file__).resolve().parents[1] / "results" / "baseline_td3bc_seed0.csv",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            if path.suffix == ".json":
                data = json.loads(path.read_text())
                if isinstance(data, dict) and env in data:
                    return float(data[env])
                if isinstance(data, dict) and "finals" in data and env in data["finals"]:
                    return float(data["finals"][env])
            elif path.suffix == ".csv":
                import csv

                with path.open() as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        key = row.get("env") or row.get("Env") or row.get("environment")
                        val = row.get("final") or row.get("Baseline final") or row.get("score")
                        if key == env and val not in (None, ""):
                            return float(val)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return _DEFAULT_BASELINE_FINALS.get(env)


def _y_at_step(xs: List[int], ys: List[float], step: int) -> Optional[float]:
    if not xs:
        return None
    # nearest step at or before `step`, else first point
    best_i = 0
    for i, x in enumerate(xs):
        if x <= step:
            best_i = i
        else:
            break
    return ys[best_i]


def _teacher_state_of(row: Dict[str, Any]) -> str:
    """Normalize teacher_state; fall back from has_teacher / stale mode."""
    raw = row.get("teacher_state")
    if isinstance(raw, str) and raw:
        return raw
    if float(row.get("has_teacher") or 0.0) > 0.5:
        return "active"
    # Without explicit state, treat inactive as disabled (break), not quarantine.
    return "disabled"


def _teacher_score_series_by_state(
    rows: List[Dict[str, Any]],
) -> tuple[List[int], List[float], List[float]]:
    """Split teacher D4RL into active (solid) vs quarantined (dotted carry-forward).

    - active: real eval score
    - quarantined: last known active score (visual 'still held, but off')
    - disabled / missing: NaN so the line breaks (no fake diagonal)
    """
    xs: List[int] = []
    y_active: List[float] = []
    y_quar: List[float] = []
    last_score: Optional[float] = None
    for r in rows:
        if "step" not in r:
            continue
        xs.append(int(r["step"]))
        state = _teacher_state_of(r)
        score = r.get("teacher_d4rl_score")
        if score is None:
            score = r.get("active_teacher_score")
        fv: Optional[float] = None
        if score is not None:
            try:
                cand = float(score)
                if not (math.isnan(cand) or math.isinf(cand)):
                    fv = cand
            except (TypeError, ValueError):
                fv = None

        if state == "active" and fv is not None:
            last_score = fv
            y_active.append(fv)
            y_quar.append(float("nan"))
        elif state == "quarantined" and last_score is not None:
            y_active.append(float("nan"))
            y_quar.append(last_score)
        else:
            # disabled, or quarantine before any active teacher: break
            y_active.append(float("nan"))
            y_quar.append(float("nan"))
    return xs, y_active, y_quar


def _nstar_value(row: Dict[str, Any]) -> Optional[float]:
    for key in ("capo_selected_n", "teacher_n", "active_teacher_nstar"):
        if key not in row or row[key] is None:
            continue
        try:
            return float(row[key])
        except (TypeError, ValueError):
            continue
    return None


def _nstar_series_by_state(
    rows: List[Dict[str, Any]],
) -> tuple[List[int], List[float], List[float], List[float]]:
    """N* line styled by teacher_state.

    - active: solid at N*
    - disabled: solid at 0 (same 0↔N* step look) + '|' markers
    - quarantined: dotted at held N* (incumbent kept; not dropped to 0)
    """
    xs: List[int] = []
    y_solid: List[float] = []
    y_dotted: List[float] = []
    y_disabled: List[float] = []
    last_n: Optional[float] = None
    for r in rows:
        if "step" not in r:
            continue
        xs.append(int(r["step"]))
        state = _teacher_state_of(r)
        n = _nstar_value(r)
        if state == "active":
            if n is not None:
                last_n = n
            val = last_n if last_n is not None else 0.0
            y_solid.append(val)
            y_dotted.append(float("nan"))
            y_disabled.append(float("nan"))
        elif state == "quarantined":
            if n is not None and n > 0:
                last_n = n
            val = last_n if last_n is not None else (n if n is not None else float("nan"))
            y_solid.append(float("nan"))
            y_dotted.append(val)
            y_disabled.append(float("nan"))
        else:
            # disabled: keep the classic step-to-0 solid line, mark with '|'
            y_solid.append(0.0)
            y_dotted.append(float("nan"))
            y_disabled.append(0.0)
    return xs, y_solid, y_dotted, y_disabled


def plot_run(
    run_dir: Path,
    out_path: Optional[Path] = None,
    baseline_final: Optional[float] = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator

    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"missing {metrics_path}")

    rows = load_metrics(metrics_path)
    if not rows:
        raise ValueError(f"empty metrics: {metrics_path}")

    cfg: Dict[str, Any] = {}
    cfg_path = run_dir / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())

    title_bits = [
        str(cfg.get("algorithm") or run_dir.parts[-4] if len(run_dir.parts) >= 4 else "run"),
        str(cfg.get("env") or ""),
        f"s{cfg.get('seed', '')}",
        str(cfg.get("run_tag") or ""),
    ]
    lambda_t_raw = cfg.get("lambda_T")
    if lambda_t_raw is None and rows:
        lambda_t_raw = rows[0].get("lambda_T")
    try:
        lambda_t = float(lambda_t_raw) if lambda_t_raw is not None else None
    except (TypeError, ValueError):
        lambda_t = None
    lambda_t_zero = lambda_t is not None and lambda_t <= 0.0
    if lambda_t is not None:
        title_bits.append(f"λT={lambda_t:g}")
    stale = cfg.get("stale_incumbent_action")
    if not stale and rows:
        stale = rows[0].get("stale_incumbent_action")
    # λ_T=0 → CAPO/teacher are inert; don't clutter the title with stale mode.
    if stale and not lambda_t_zero:
        title_bits.append(f"stale={stale}")
    if lambda_t_zero:
        title_bits.append("no_capo")
    title = " | ".join(b for b in title_bits if b and b != "s")

    out_path = out_path or (run_dir / "training_curve.png")

    data_max = max(int(r.get("step", 0)) for r in rows)
    x_max = max(int(cfg.get("max_timesteps") or 0), data_max)
    xticks = _shared_xticks(x_max, step=200_000)
    refresh_events = _load_refresh_events(run_dir)
    replace_new_steps = [
        int(e.get("refresh_step", e.get("step", 0)))
        for e in refresh_events
        if e.get("replacement_decision") == "replace_new"
        or e.get("accepted_by_cert") is True
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, constrained_layout=True)
    fig.suptitle(title, fontsize=11)

    # (0,0) D4RL scores — student / teacher / distilled (+ baseline dotted).
    # λ_T=0: teacher/N*/replace markers are inert — student (+ baseline) only.
    ax = axes[0, 0]
    xs, ys = _series(rows, "student_d4rl_score")
    if not xs:
        xs, ys = _series(rows, "d4rl_score")
    handles = []
    labels = []
    if xs:
        (h_s,) = ax.plot(xs, ys, color="#1f77b4", lw=1.6)
        handles.append(h_s)
        labels.append(f"student {ys[-1]:.1f}")
    if not lambda_t_zero:
        xt, yt_act, yt_quar = _teacher_score_series_by_state(rows)
        if any(not math.isnan(v) for v in yt_act):
            (h_t,) = ax.plot(xt, yt_act, color="#2ca02c", lw=1.2, alpha=0.9)
            handles.append(h_t)
            last_t = next(v for v in reversed(yt_act) if not math.isnan(v))
            labels.append(f"teacher {last_t:.1f}")
        if any(not math.isnan(v) for v in yt_quar):
            (h_q,) = ax.plot(
                xt,
                yt_quar,
                color="#2ca02c",
                lw=1.2,
                ls=":",
                alpha=0.85,
            )
            handles.append(h_q)
            labels.append("teacher (quarantine)")
        elif not any(not math.isnan(v) for v in yt_act):
            # Legacy fallback when teacher_state is absent.
            xt_legacy, yt_legacy = _series_aligned(rows, "teacher_d4rl_score")
            if not any(not math.isnan(v) for v in yt_legacy):
                xt_legacy, yt_legacy = _series_aligned(rows, "active_teacher_score")
            if any(not math.isnan(v) for v in yt_legacy):
                (h_t,) = ax.plot(xt_legacy, yt_legacy, color="#2ca02c", lw=1.2, alpha=0.85)
                handles.append(h_t)
                last_t = next(v for v in reversed(yt_legacy) if not math.isnan(v))
                labels.append(f"teacher {last_t:.1f}")
        xd, yd = _load_distill_series(run_dir)
        if xd:
            (h_d,) = ax.plot(xd, yd, color="#9467bd", lw=1.4, alpha=0.9)
            handles.append(h_d)
            labels.append(f"distilled {yd[-1]:.1f}")
    env_name = str(cfg.get("env") or "")
    bl = baseline_final if baseline_final is not None else _load_baseline_final(env_name, run_dir)
    if bl is not None:
        (h_b,) = ax.plot(
            [0, xticks[-1] if xticks else x_max],
            [bl, bl],
            color="#7f7f7f",
            lw=1.2,
            ls="--",
            alpha=0.9,
        )
        handles.append(h_b)
        labels.append(f"baseline {bl:.1f}")
    if (not lambda_t_zero) and replace_new_steps and xs:
        mx, my = [], []
        for s in replace_new_steps:
            y = _y_at_step(xs, ys, s)
            if y is None:
                continue
            mx.append(s)
            my.append(y)
        if mx:
            h_r = ax.scatter(
                mx, my, s=12, color="#d62728", zorder=3, marker="o", label="replace_new"
            )
            handles.append(h_r)
            labels.append("replace_new")
    ax.set_ylabel("D4RL score")
    ax.grid(True, alpha=0.3)
    if handles:
        ax.legend(handles, labels, loc="best", fontsize=8)

    # (0,1) N* — solid 0↔N* step; quarantine=dotted hold; disable=0 + '|' markers.
    # λ_T=0: force flat N*=0 (CAPO/teacher guidance off).
    ax = axes[0, 1]
    handles_n = []
    labels_n = []
    if lambda_t_zero:
        x0 = [0, xticks[-1] if xticks else x_max]
        (h_n,) = ax.plot(x0, [0.0, 0.0], color="#8c564b", lw=1.3)
        handles_n.append(h_n)
        labels_n.append("N*=0 (λT=0)")
    else:
        xn, yn_solid, yn_dot, yn_dis = _nstar_series_by_state(rows)
        if any(not math.isnan(v) for v in yn_solid):
            (h_n,) = ax.plot(xn, yn_solid, color="#8c564b", lw=1.3)
            handles_n.append(h_n)
            labels_n.append("N*")
        if any(not math.isnan(v) for v in yn_dot):
            (h_nq,) = ax.plot(xn, yn_dot, color="#8c564b", lw=1.3, ls=":")
            handles_n.append(h_nq)
            labels_n.append("N* (quarantine)")
        if any(not math.isnan(v) for v in yn_dis):
            (h_nd,) = ax.plot(
                xn,
                yn_dis,
                color="#8c564b",
                lw=0.0,
                marker="|",
                markersize=9,
                alpha=0.85,
            )
            handles_n.append(h_nd)
            labels_n.append("disabled")
        if not handles_n:
            # Legacy: plain N* / has_teacher when teacher_state is absent.
            xs, ys = _series(rows, "capo_selected_n")
            if xs:
                ax.plot(xs, ys, label="N*", color="#8c564b", lw=1.3)
            xs, ys = _series(rows, "has_teacher")
            if xs:
                ax.plot(xs, ys, label="has_teacher", color="#7f7f7f", lw=1.0, alpha=0.8)
    ax.set_ylabel("N*")
    ax.grid(True, alpha=0.3)
    if handles_n:
        ax.legend(handles_n, labels_n, loc="best", fontsize=8)
    elif ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=8)

    # (1,0) student distance to dataset actions (minibatch)
    # Raw mean((pi - a_data)^2); actions are tanh-squashed to ~[-1,1], so MSE
    # often sits well below 0.2. Do NOT force ylim to 1.0 (that flattens the curve).
    ax = axes[1, 0]
    xs, ys = _series(rows, "bc_data")
    if not xs:
        xs, ys = _series(rows, "dataset_action_mse")
    if not xs:
        xs, ys = _series(rows, "data_bc_loss")
    if xs:
        ax.plot(xs, ys, color="#1f77b4", lw=1.2, label="student–data")
        ax.legend(loc="best", fontsize=8)
        ymax = max(ys)
        ymin = min(ys)
        pad = max(0.005, (ymax - ymin) * 0.15) if ymax > ymin else max(0.01, ymax * 0.15)
        ax.set_ylim(max(0.0, ymin - pad), ymax + pad)
    else:
        ax.set_ylim(0.0, 0.1)
    ax.set_ylabel("dataset action MSE")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)

    # (1,1) mean(|Q|) ≈ actor_q_scale - eps (JAX logs q_scale / capo_q_scale)
    ax = axes[1, 1]
    eps = float(cfg.get("actor_q_scale_eps", 1e-4))
    xs, ys = _series(rows, "actor_q_scale")
    q_label = r"mean(|Q|)"
    if not xs:
        xs, ys = _series(rows, "q_scale")
        q_label = "q_scale"
    if not xs:
        xs, ys = _series(rows, "capo_q_scale")
        q_label = "capo_q_scale"
    if xs:
        if q_label == r"mean(|Q|)":
            y_plot = [max(0.0, y - eps) for y in ys]
        else:
            y_plot = ys
        ax.plot(xs, y_plot, color="#ff7f0e", lw=1.2, label=q_label)
        ax.legend(loc="best", fontsize=8)
    ax.set_ylabel(q_label)
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlim(0, xticks[-1] if xticks else x_max)
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.tick_params(axis="x", labelbottom=True)

    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description="Plot CAPO training_curve.png from metrics.jsonl")
    p.add_argument("run_dir", type=str, nargs="?", help="Path to a run directory")
    p.add_argument("--out", type=str, default=None, help="Output PNG path")
    p.add_argument(
        "--all-under",
        type=str,
        default=None,
        help="Replot every run_dir containing metrics.jsonl under this root",
    )
    args = p.parse_args()
    if args.all_under:
        root = Path(args.all_under).resolve()
        rc = 0
        for metrics in sorted(root.rglob("metrics.jsonl")):
            run_dir = metrics.parent
            try:
                path = plot_run(run_dir)
                print(f"[plot] {path}")
            except Exception as e:
                print(f"[plot] fail {run_dir}: {e}", file=sys.stderr)
                rc = 1
        return rc

    if not args.run_dir:
        p.error("run_dir or --all-under required")
    run_dir = Path(args.run_dir).resolve()
    out = Path(args.out).resolve() if args.out else None
    try:
        path = plot_run(run_dir, out)
    except Exception as e:
        print(f"[plot] fail {run_dir}: {e}", file=sys.stderr)
        return 1
    print(f"[plot] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
