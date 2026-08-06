#!/usr/bin/env python3
"""GPU-aware manifest runner for the JAX CAPO stability sweep."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_manifest(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_status(row: Dict[str, Any]) -> str:
    run_dir = Path(row["output_dir"])
    summary = run_dir / "summary.json"
    if summary.exists():
        try:
            payload = json.loads(summary.read_text())
            final = payload.get("final_eval") or {}
            if payload.get("status") == "complete" and int(
                final.get("step", row["config"]["max_timesteps"])
            ) >= int(row["config"]["max_timesteps"]):
                return "complete"
        except Exception:
            pass
    if (run_dir / "failure.json").exists():
        return "failed"
    if (run_dir / "latest.pkl").exists():
        return "incomplete"
    if run_dir.exists() and any(run_dir.iterdir()):
        return "started_without_checkpoint"
    return "pending"


def find_live_trainer_pid(run_id: str) -> int | None:
    """Return PID of a live run_capo_jax for this run_id, if any."""
    needle = f"/{run_id}/"
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            if "run_capo_jax.py" in cmdline and needle in cmdline:
                return int(entry.name)
    except FileNotFoundError:
        return None
    return None


def live_cuda_device(pid: int) -> str | None:
    try:
        environ = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    for item in environ:
        if item.startswith(b"CUDA_VISIBLE_DEVICES="):
            value = item.split(b"=", 1)[1].decode()
            return value.split(",")[0] if value else None
    return None


class _ExternalProcess:
    """Minimal poll() wrapper for trainers launched by a previous runner."""

    def __init__(self, pid: int):
        self.pid = pid

    def poll(self):
        if Path(f"/proc/{self.pid}").exists():
            return None
        return 0


class _NullLog:
    def close(self):
        return None


def write_progress(path: Path, rows: List[Dict[str, Any]], running: Dict[int, Any]):
    counts: Dict[str, int] = {}
    for row in rows:
        status = "running" if any(job["row"] is row for job in running.values()) else run_status(row)
        counts[status] = counts.get(status, 0) + 1
    payload = {
        "updated_unix": time.time(),
        "counts": counts,
        "running": [
            {
                "pid": pid,
                "gpu": job["gpu"],
                "run_id": job["row"]["run_id"],
                "started_unix": job["started"],
            }
            for pid, job in running.items()
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def prepare_mujoco_overlay(results_root: Path, source: str | None) -> str | None:
    if not source:
        return None
    source_path = Path(source)
    if not source_path.is_dir():
        return None
    overlay = results_root / ".runtime_overlay"
    overlay.mkdir(parents=True, exist_ok=True)
    link = overlay / "mujoco_py"
    if not link.exists():
        link.symlink_to(source_path, target_is_directory=True)
    return str(overlay)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--prepend_manifest", action="append", default=[],
        help="Priority manifest(s) placed before the main queue",
    )
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--jobs_per_gpu", type=int, default=1)
    parser.add_argument("--max_concurrent_jobs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override resolved config values (repeatable; YAML value syntax)",
    )
    parser.add_argument("--poll_sec", type=float, default=10.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--mujoco_py_source",
        default="/home/ext_csh/miniconda3/envs/capo/lib/python3.10/site-packages/mujoco_py",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    rows = read_manifest(manifest_path)
    for priority_path in reversed(args.prepend_manifest):
        rows = read_manifest(Path(priority_path).resolve()) + rows
    run_ids = [row["run_id"] for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("duplicate run IDs across priority and main manifests")
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]
    overrides = {}
    for assignment in args.override:
        if "=" not in assignment:
            raise ValueError(f"invalid --override {assignment!r}; expected KEY=VALUE")
        key, raw_value = assignment.split("=", 1)
        overrides[key] = yaml.safe_load(raw_value)
    if overrides:
        for row in rows:
            row["config"] = {**row["config"], **overrides}
            row["output_dir"] = str(
                Path(row["config"]["out_dir"])
                / row["config"]["sweep_name"]
                / row["run_id"]
            )
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU")
    slots = [gpu for gpu in gpus for _ in range(max(1, args.jobs_per_gpu))]
    max_jobs = args.max_concurrent_jobs or len(slots)
    max_jobs = max(1, min(max_jobs, len(slots)))
    slots = slots[:max_jobs]

    status_counts: Dict[str, int] = {}
    for row in rows:
        status = run_status(row)
        status_counts[status] = status_counts.get(status, 0) + 1
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "runs": len(rows),
                "gpus": gpus,
                "jobs_per_gpu": args.jobs_per_gpu,
                "max_concurrent_jobs": max_jobs,
                "resume": args.resume,
                "status": status_counts,
                "first": [row["run_id"] for row in rows[:3]],
                "last": [row["run_id"] for row in rows[-3:]],
            },
            indent=2,
        )
    )
    if args.dry_run:
        return

    results_root = Path(rows[0]["config"]["out_dir"]).resolve()
    overlay = prepare_mujoco_overlay(results_root, args.mujoco_py_source)
    progress_path = results_root / rows[0]["config"]["sweep_name"] / "sweep_progress.json"
    pending = []
    resume_first = []
    adopted: List[Dict[str, Any]] = []
    for row in rows:
        status = run_status(row)
        if args.resume and status == "complete":
            continue
        if status in ("incomplete", "failed") and not args.resume:
            raise RuntimeError(
                f"{row['run_id']} is {status}; pass --resume to continue safely"
            )
        if status == "started_without_checkpoint":
            raise RuntimeError(
                f"{row['run_id']} has outputs but no checkpoint; inspect before retry"
            )
        live_pid = find_live_trainer_pid(row["run_id"])
        if live_pid is not None:
            adopted.append({"pid": live_pid, "row": row})
            continue
        # Prefer resuming incomplete/failed cells before launching brand-new ones.
        if status in ("incomplete", "failed"):
            resume_first.append(row)
        else:
            pending.append(row)
    def _resume_progress(row: Dict[str, Any]) -> int:
        run_dir = Path(row["output_dir"])
        steps = []
        for path in run_dir.glob("checkpoint_*.pkl"):
            try:
                steps.append(int(path.stem.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        return max(steps) if steps else 0

    resume_first.sort(key=_resume_progress, reverse=True)
    pending = resume_first + pending

    running: Dict[int, Dict[str, Any]] = {}
    free_slots = list(slots)
    for job in adopted:
        gpu = live_cuda_device(job["pid"])
        if gpu in free_slots:
            free_slots.remove(gpu)
        elif free_slots:
            gpu = free_slots.pop(0)
        else:
            gpu = gpu or "?"
        running[job["pid"]] = {
            "process": _ExternalProcess(job["pid"]),
            "row": job["row"],
            "gpu": gpu,
            "started": time.time(),
            "log": _NullLog(),
        }
        print(
            f"[adopt] gpu={gpu} pid={job['pid']} {job['row']['run_id']}",
            flush=True,
        )
    try:
        while pending or running:
            while pending and free_slots:
                row = pending.pop(0)
                live_pid = find_live_trainer_pid(row["run_id"])
                if live_pid is not None:
                    print(
                        f"[skip-live] pid={live_pid} {row['run_id']}",
                        flush=True,
                    )
                    continue
                gpu = free_slots.pop(0)
                run_dir = Path(row["output_dir"]).resolve()
                run_dir.mkdir(parents=True, exist_ok=True)
                resolved_path = run_dir / "resolved_config.yaml"
                resolved_path.write_text(
                    yaml.safe_dump(row["config"], sort_keys=False)
                )
                command = [
                    args.python,
                    str(ROOT / "scripts" / "run_capo_jax.py"),
                    "--config",
                    str(resolved_path),
                ]
                if args.resume and (run_dir / "latest.pkl").exists():
                    command.extend(["--resume_run_dir", str(run_dir)])
                env = os.environ.copy()
                env.update(
                    {
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        "JAX_PLATFORMS": "cuda",
                        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                        "D4RL_SUPPRESS_IMPORT_ERROR": "1",
                        "MUJOCO_GL": env.get("MUJOCO_GL", "egl"),
                        "MUJOCO_PY_MUJOCO_PATH": env.get(
                            "MUJOCO_PY_MUJOCO_PATH",
                            "/home/ext_csh/.mujoco/mujoco210",
                        ),
                    }
                )
                if overlay:
                    env["PYTHONPATH"] = overlay + (
                        os.pathsep + env["PYTHONPATH"]
                        if env.get("PYTHONPATH")
                        else ""
                    )
                library_paths = [
                    "/home/ext_csh/.mujoco/mujoco210/bin",
                    str(Path(args.python).resolve().parents[1] / "lib"),
                    "/home/ext_csh/miniconda3/envs/capo/lib",
                    "/usr/lib/x86_64-linux-gnu",
                ]
                if env.get("LD_LIBRARY_PATH"):
                    library_paths.append(env["LD_LIBRARY_PATH"])
                env["LD_LIBRARY_PATH"] = os.pathsep.join(library_paths)
                log_stream = open(run_dir / "launcher.log", "a", buffering=1)
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                )
                running[process.pid] = {
                    "process": process,
                    "row": row,
                    "gpu": gpu,
                    "started": time.time(),
                    "log": log_stream,
                }
                print(f"[launch] gpu={gpu} pid={process.pid} {row['run_id']}", flush=True)

            write_progress(progress_path, rows, running)
            if not running:
                continue
            time.sleep(max(0.25, args.poll_sec))
            for pid, job in list(running.items()):
                returncode = job["process"].poll()
                if returncode is None:
                    continue
                job["log"].close()
                if job["gpu"] in slots:
                    free_slots.append(job["gpu"])
                    free_slots.sort(key=lambda item: slots.index(item))
                row = job["row"]
                if returncode != 0:
                    failure = {
                        "run_id": row["run_id"],
                        "returncode": returncode,
                        "failed_unix": time.time(),
                    }
                    Path(row["output_dir"], "failure.json").write_text(
                        json.dumps(failure, indent=2)
                    )
                    print(f"[failed] rc={returncode} {row['run_id']}", flush=True)
                else:
                    print(f"[complete] {row['run_id']}", flush=True)
                    run_dir = Path(row["output_dir"])
                    try:
                        plot = subprocess.run(
                            [
                                args.python,
                                str(ROOT / "scripts" / "plot_training_curve.py"),
                                str(run_dir),
                            ],
                            cwd=ROOT,
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        if plot.returncode == 0:
                            print(f"[plot] {run_dir / 'training_curve.png'}", flush=True)
                        else:
                            err = (plot.stderr or plot.stdout or "").strip()
                            print(
                                f"[plot] failed {row['run_id']}: {err[:300]}",
                                flush=True,
                            )
                    except Exception as e:
                        print(f"[plot] skipped {row['run_id']}: {e}", flush=True)
                del running[pid]
    except KeyboardInterrupt:
        for job in running.values():
            job["process"].terminate()
            job["log"].close()
        raise
    finally:
        write_progress(progress_path, rows, running)


if __name__ == "__main__":
    main()
