"""Post-hoc persistent student distillation (pure BC).

Standalone experiment: a single distillation actor tracks a chronological
sequence of frozen student checkpoints via MSE action cloning only.
Does not modify CAPO training, critics, certificates, or teacher BC.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .buffer import (
    NormalizeObsWrapper,
    ReplayBuffer,
    load_d4rl_dataset,
    make_env,
)
from .networks import Actor
from .trainer import eval_actor, set_seed

_CKPT_STEP_RE = re.compile(r"checkpoint_(\d+)\.pt$")


@dataclass
class PosthocDistillConfig:
    checkpoint_dir: str
    out_dir: str = "results/posthoc_student_distill"
    checkpoint_glob: str = "checkpoint_*.pt"
    start_step: int = 100_000
    end_step: int = 1_000_000
    checkpoint_interval: int = 50_000

    distill_steps_per_checkpoint: int = 1000
    distill_lr: Optional[float] = None
    distill_batch_size: Optional[int] = None

    device: str = "cuda"
    seed: int = 0
    n_episodes: int = 10
    eval_seed0: int = 0
    heldout_batch_size: int = 2048
    heldout_seed: int = 12345

    # If set, only process these two steps (smoke). Still validated against disk.
    smoke_steps: Optional[Tuple[int, int]] = None

    resume: bool = True


def parse_checkpoint_step(path: Path) -> Optional[int]:
    m = _CKPT_STEP_RE.search(path.name)
    if not m:
        return None
    return int(m.group(1))


def discover_checkpoints(
    checkpoint_dir: Path,
    checkpoint_glob: str,
    start_step: int,
    end_step: int,
    checkpoint_interval: int,
    smoke_steps: Optional[Tuple[int, int]] = None,
) -> List[Tuple[int, Path]]:
    """Return (step, path) sorted by step. Fail if any expected step is missing."""
    if checkpoint_interval <= 0:
        raise ValueError(f"checkpoint_interval must be > 0, got {checkpoint_interval}")
    if end_step < start_step:
        raise ValueError(f"end_step ({end_step}) < start_step ({start_step})")

    found: Dict[int, Path] = {}
    for path in checkpoint_dir.glob(checkpoint_glob):
        step = parse_checkpoint_step(path)
        if step is None:
            continue
        if step in found:
            raise FileNotFoundError(
                f"duplicate checkpoint step {step}: {found[step]} and {path}"
            )
        found[step] = path

    if smoke_steps is not None:
        expected = [int(smoke_steps[0]), int(smoke_steps[1])]
        if expected[1] <= expected[0]:
            raise ValueError(f"smoke_steps must be increasing, got {smoke_steps}")
    else:
        if (start_step - end_step) % checkpoint_interval != 0 and (
            end_step - start_step
        ) % checkpoint_interval != 0:
            # still build arithmetic sequence; end must land on grid
            pass
        if (end_step - start_step) % checkpoint_interval != 0:
            raise ValueError(
                f"end_step-start_step must be divisible by checkpoint_interval; "
                f"got start={start_step} end={end_step} interval={checkpoint_interval}"
            )
        expected = list(range(start_step, end_step + 1, checkpoint_interval))

    missing = [s for s in expected if s not in found]
    if missing:
        raise FileNotFoundError(
            f"missing checkpoint(s) under {checkpoint_dir} for steps {missing}. "
            f"found={sorted(found)}"
        )

    ordered = [(s, found[s]) for s in expected]
    steps = [s for s, _ in ordered]
    if steps != sorted(steps):
        raise RuntimeError("internal error: checkpoints not in increasing order")
    if any(steps[i] >= steps[i + 1] for i in range(len(steps) - 1)):
        raise RuntimeError("checkpoint steps are not strictly increasing")
    return ordered


def _load_run_config(checkpoint_dir: Path, first_ckpt: Path) -> Dict[str, Any]:
    cfg_path = checkpoint_dir / "config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())
    payload = torch.load(first_ckpt, map_location="cpu", weights_only=False)
    cfg = payload.get("config")
    if not isinstance(cfg, dict):
        raise FileNotFoundError(
            f"no config.json in {checkpoint_dir} and no config in {first_ckpt}"
        )
    return cfg


def _build_actor_from_state(
    state_dict: Dict[str, Tensor],
    state_dim: int,
    action_dim: int,
    max_action: float,
    hidden: int,
    device: torch.device,
) -> Actor:
    actor = Actor(state_dim, action_dim, max_action=max_action, hidden=hidden).to(device)
    actor.load_state_dict(state_dict)
    return actor


def load_student_actor(
    ckpt_path: Path,
    state_dim: int,
    action_dim: int,
    max_action: float,
    hidden: int,
    device: torch.device,
) -> Actor:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "actor" not in payload:
        raise KeyError(f"'actor' missing in {ckpt_path}; keys={list(payload.keys())}")
    actor = _build_actor_from_state(
        payload["actor"], state_dim, action_dim, max_action, hidden, device
    )
    actor.eval()
    for p in actor.parameters():
        p.requires_grad_(False)
    return actor


def parameter_l2_distance(a: nn.Module, b: nn.Module) -> float:
    total = 0.0
    for pa, pb in zip(a.parameters(), b.parameters()):
        total += float((pa.detach() - pb.detach()).pow(2).sum().item())
    return float(total ** 0.5)


@torch.no_grad()
def action_mse_between(actor_a: Actor, actor_b: Actor, states: Tensor) -> float:
    return float(F.mse_loss(actor_a.act(states), actor_b.act(states)).item())


def distill_loss_pure_bc(pred_actions: Tensor, target_actions: Tensor) -> Tensor:
    """Exact pure student-action MSE (element-wise mean)."""
    return F.mse_loss(pred_actions, target_actions)


@dataclass
class DistillRunState:
    initialized_from_first_student: bool = False
    last_completed_step: Optional[int] = None
    distill_optimizer_id: Optional[int] = None
    init_count: int = 0


class PosthocStudentDistiller:
    def __init__(self, cfg: PosthocDistillConfig):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device if (cfg.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
        )
        set_seed(cfg.seed)

        self.checkpoint_dir = Path(cfg.checkpoint_dir).expanduser().resolve()
        if not self.checkpoint_dir.is_dir():
            raise FileNotFoundError(f"checkpoint_dir not found: {self.checkpoint_dir}")

        self.checkpoints = discover_checkpoints(
            self.checkpoint_dir,
            cfg.checkpoint_glob,
            cfg.start_step,
            cfg.end_step,
            cfg.checkpoint_interval,
            smoke_steps=cfg.smoke_steps,
        )
        self.run_cfg = _load_run_config(self.checkpoint_dir, self.checkpoints[0][1])
        self.env_name = str(self.run_cfg.get("env") or "")
        if not self.env_name:
            raise ValueError("run config missing env")

        normalize = bool(self.run_cfg.get("normalize", True))
        normalize_reward = bool(self.run_cfg.get("normalize_reward", True))
        hidden = int(self.run_cfg.get("hidden", 256))
        buffer_size = int(self.run_cfg.get("buffer_size", 2_000_000))

        data, stats, _raw = load_d4rl_dataset(
            self.env_name,
            normalize=normalize,
            normalize_reward=normalize_reward,
            device=str(self.device),
        )
        self.stats = stats
        self.state_normalization_enabled = normalize
        self.state_dim = int(data["observations"].shape[1])
        self.action_dim = int(data["actions"].shape[1])
        self.max_action = float(stats.max_action)
        self.hidden = hidden

        self.buffer = ReplayBuffer(
            self.state_dim, self.action_dim, buffer_size, str(self.device)
        )
        self.buffer.load_d4rl(data)

        self.distill_lr = float(
            cfg.distill_lr
            if cfg.distill_lr is not None
            else self.run_cfg.get("actor_lr", 3e-4)
        )
        self.distill_batch_size = int(
            cfg.distill_batch_size
            if cfg.distill_batch_size is not None
            else self.run_cfg.get("batch_size", 256)
        )
        self.distill_steps = int(cfg.distill_steps_per_checkpoint)

        self.eval_env = make_env(self.env_name, seed=cfg.seed + 100)
        if normalize:
            self.eval_env = NormalizeObsWrapper(
                self.eval_env, stats.state_mean, stats.state_std
            )

        self.out_dir = Path(cfg.out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_jsonl = self.out_dir / "distill_metrics.jsonl"
        self.metrics_csv = self.out_dir / "distill_metrics.csv"
        self.state_path = self.out_dir / "distill_state.pt"

        # Fixed held-out states (normalized already if normalize=True in buffer).
        g = torch.Generator(device="cpu")
        g.manual_seed(int(cfg.heldout_seed))
        n = self.buffer.size
        idx = torch.randint(0, n, (int(cfg.heldout_batch_size),), generator=g)
        self.heldout_states = self.buffer.states[idx.to(self.device)].detach().clone()

        self.distill_actor: Optional[Actor] = None
        self.distill_opt: Optional[torch.optim.Optimizer] = None
        self.run_state = DistillRunState()
        self._student_grad_guard_hooks: List[Any] = []

        with open(self.out_dir / "experiment_config.json", "w") as f:
            json.dump(
                {
                    "posthoc": asdict(cfg),
                    "resolved_distill_lr": self.distill_lr,
                    "resolved_distill_batch_size": self.distill_batch_size,
                    "resolved_distill_steps_per_checkpoint": self.distill_steps,
                    "env": self.env_name,
                    "checkpoint_sequence": [s for s, _ in self.checkpoints],
                    "source_run_config": self.run_cfg,
                    "state_normalization_enabled": self.state_normalization_enabled,
                    "state_mean": np.asarray(stats.state_mean).tolist(),
                    "state_std": np.asarray(stats.state_std).tolist(),
                    "pure_bc_loss": "F.mse_loss(distill_actor(states), frozen_student(states))",
                },
                f,
                indent=2,
            )

    def _sample_states(self, batch_size: int) -> Tensor:
        # Buffer observations are already normalized exactly once at load time.
        s, _, _, _, _ = self.buffer.sample(batch_size)
        return s

    def _eval_seeds(self) -> List[int]:
        return [int(self.cfg.eval_seed0) + i for i in range(int(self.cfg.n_episodes))]

    def _save_state(self, checkpoint_step: int) -> None:
        assert self.distill_actor is not None and self.distill_opt is not None
        payload = {
            "distill_actor": self.distill_actor.state_dict(),
            "distill_optimizer": self.distill_opt.state_dict(),
            "checkpoint_step": int(checkpoint_step),
            "state_mean": np.asarray(self.stats.state_mean),
            "state_std": np.asarray(self.stats.state_std),
            "state_normalization_enabled": self.state_normalization_enabled,
            "experiment_config": asdict(self.cfg),
            "resolved_distill_lr": self.distill_lr,
            "resolved_distill_batch_size": self.distill_batch_size,
            "run_state": asdict(self.run_state),
            "distill_optimizer_id": id(self.distill_opt),
            "env": self.env_name,
            "hidden": self.hidden,
            "max_action": self.max_action,
        }
        torch.save(payload, self.state_path)
        step_path = self.out_dir / f"distill_actor_after_{int(checkpoint_step)}.pt"
        torch.save(
            {
                "distill_actor": self.distill_actor.state_dict(),
                "checkpoint_step": int(checkpoint_step),
            },
            step_path,
        )

    def _try_resume(self) -> Optional[int]:
        if not self.cfg.resume or not self.state_path.exists():
            return None
        payload = torch.load(self.state_path, map_location=self.device, weights_only=False)
        last = int(payload["checkpoint_step"])
        self.distill_actor = _build_actor_from_state(
            payload["distill_actor"],
            self.state_dim,
            self.action_dim,
            self.max_action,
            self.hidden,
            self.device,
        )
        self.distill_actor.train()
        self.distill_opt = torch.optim.Adam(
            self.distill_actor.parameters(), lr=self.distill_lr
        )
        self.distill_opt.load_state_dict(payload["distill_optimizer"])
        rs = payload.get("run_state") or {}
        self.run_state = DistillRunState(
            initialized_from_first_student=bool(
                rs.get("initialized_from_first_student", True)
            ),
            last_completed_step=last,
            distill_optimizer_id=id(self.distill_opt),
            init_count=int(rs.get("init_count", 1)),
        )
        print(
            f"[posthoc-distill] resume after checkpoint_step={last} "
            f"opt_id={id(self.distill_opt)}",
            flush=True,
        )
        return last

    def _append_metrics(self, row: Dict[str, Any]) -> None:
        with open(self.metrics_jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")
        write_header = not self.metrics_csv.exists()
        with open(self.metrics_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)

    def _initialize_from_student(self, student: Actor, step: int) -> None:
        if self.run_state.init_count > 0 or self.distill_actor is not None:
            raise RuntimeError(
                "distillation actor must be initialized only once "
                f"(init_count={self.run_state.init_count})"
            )
        self.distill_actor = student.copy()
        self.distill_actor.train()
        for p in self.distill_actor.parameters():
            p.requires_grad_(True)
        self.distill_opt = torch.optim.Adam(
            self.distill_actor.parameters(), lr=self.distill_lr
        )
        self.run_state.initialized_from_first_student = True
        self.run_state.init_count = 1
        self.run_state.distill_optimizer_id = id(self.distill_opt)
        print(
            f"[posthoc-distill] initialized pi_D from student@{step} "
            f"opt_id={id(self.distill_opt)} lr={self.distill_lr}",
            flush=True,
        )

    def _assert_student_frozen(self, student: Actor) -> None:
        for p in student.parameters():
            if p.requires_grad:
                raise RuntimeError("student parameter requires_grad=True")
            if p.grad is not None and float(p.grad.abs().sum()) > 0:
                raise RuntimeError("student has non-zero gradients")

    def _bc_update_loop(self, student: Actor) -> Tuple[float, float]:
        assert self.distill_actor is not None and self.distill_opt is not None
        opt_id_before = id(self.distill_opt)
        student.eval()
        for p in student.parameters():
            p.requires_grad_(False)

        # loss before
        with torch.no_grad():
            s0 = self._sample_states(self.distill_batch_size)
            loss_before = float(
                distill_loss_pure_bc(
                    self.distill_actor.act(s0), student.act(s0)
                ).item()
            )

        self.distill_actor.train()
        for _ in range(self.distill_steps):
            states = self._sample_states(self.distill_batch_size)
            with torch.no_grad():
                target_actions = student.act(states)
            pred = self.distill_actor.act(states)
            loss = distill_loss_pure_bc(pred, target_actions)
            # Safety: targets detached
            assert not target_actions.requires_grad
            self.distill_opt.zero_grad(set_to_none=True)
            loss.backward()
            self._assert_student_frozen(student)
            self.distill_opt.step()

        if id(self.distill_opt) != opt_id_before:
            raise RuntimeError("optimizer object was recreated during BC updates")
        if self.run_state.distill_optimizer_id != id(self.distill_opt):
            raise RuntimeError("optimizer id changed vs run_state")

        with torch.no_grad():
            s1 = self._sample_states(self.distill_batch_size)
            loss_after = float(
                distill_loss_pure_bc(
                    self.distill_actor.act(s1), student.act(s1)
                ).item()
            )
        return loss_before, loss_after

    def run(self) -> Path:
        last_done = self._try_resume()
        seeds = self._eval_seeds()
        init_step, init_path = self.checkpoints[0]

        print(
            f"[posthoc-distill] sequence={[s for s, _ in self.checkpoints]} "
            f"device={self.device} steps/ckpt={self.distill_steps} "
            f"batch={self.distill_batch_size} lr={self.distill_lr}",
            flush=True,
        )
        print(
            f"[posthoc-distill] student actor key='actor' "
            f"norm_enabled={self.state_normalization_enabled} "
            f"state_mean_shape={tuple(np.asarray(self.stats.state_mean).shape)}",
            flush=True,
        )

        for i, (step, path) in enumerate(self.checkpoints):
            if last_done is not None and step <= last_done:
                print(f"[posthoc-distill] skip already completed step={step}", flush=True)
                continue

            student = load_student_actor(
                path,
                self.state_dim,
                self.action_dim,
                self.max_action,
                self.hidden,
                self.device,
            )
            self._assert_student_frozen(student)

            is_init = self.distill_actor is None
            if is_init:
                if step != init_step and last_done is None:
                    raise RuntimeError(
                        "first processed checkpoint must initialize pi_D from the "
                        f"first sequence step ({init_step}), got step={step}"
                    )
                self._initialize_from_student(student, step)
                assert self.distill_actor is not None
                mse_before = action_mse_between(
                    self.distill_actor, student, self.heldout_states
                )
                loss_before = float(mse_before)
                loss_after = float(mse_before)
                mse_after = float(mse_before)
                # No BC updates on the initialization checkpoint.
            else:
                assert self.distill_actor is not None
                mse_before = action_mse_between(
                    self.distill_actor, student, self.heldout_states
                )
                loss_before, loss_after = self._bc_update_loop(student)
                if id(self.distill_opt) != self.run_state.distill_optimizer_id:
                    raise RuntimeError("optimizer recreated across checkpoints")
                mse_after = action_mse_between(
                    self.distill_actor, student, self.heldout_states
                )

            assert self.distill_actor is not None
            param_dist = parameter_l2_distance(self.distill_actor, student)

            student_eval = eval_actor(
                self.eval_env, student, str(self.device), len(seeds), episode_seeds=seeds
            )
            distill_eval = eval_actor(
                self.eval_env,
                self.distill_actor,
                str(self.device),
                len(seeds),
                episode_seeds=seeds,
            )
            student_score = float(student_eval.get("d4rl_score", student_eval["return_mean"]))
            distilled_score = float(
                distill_eval.get("d4rl_score", distill_eval["return_mean"])
            )

            row = {
                "checkpoint_step": int(step),
                "student_score": student_score,
                "distilled_score": distilled_score,
                "distilled_minus_student": distilled_score - student_score,
                "distill_steps": 0 if is_init else int(self.distill_steps),
                "distill_lr": float(self.distill_lr),
                "distill_batch_size": int(self.distill_batch_size),
                "distill_loss_before": float(loss_before),
                "distill_loss_after": float(loss_after),
                "action_mse_before": float(mse_before),
                "action_mse_after": float(mse_after),
                "actor_parameter_distance": float(param_dist),
                "state_normalization_enabled": bool(self.state_normalization_enabled),
                "initialized_from_first_student": bool(
                    self.run_state.initialized_from_first_student
                ),
                "is_initialization_checkpoint": bool(is_init),
                "optimizer_id": int(id(self.distill_opt)),
                "checkpoint_path": str(path),
            }

            self._append_metrics(row)
            self.run_state.last_completed_step = int(step)
            self._save_state(step)

            print(
                f"[posthoc-distill] step={step} init={is_init} "
                f"student={student_score:.2f} distilled={distilled_score:.2f} "
                f"Δ={distilled_score - student_score:+.2f} "
                f"mse_holdout={row['action_mse_after']:.6g} "
                f"param_L2={param_dist:.4g}",
                flush=True,
            )

            # For non-init rows, also log pre-update heldout MSE by recomputing is awkward
            # after updates; we stored loss_before from a train batch instead.

        summary = {
            "status": "done",
            "checkpoints": [s for s, _ in self.checkpoints],
            "out_dir": str(self.out_dir),
            "metrics_jsonl": str(self.metrics_jsonl),
            "metrics_csv": str(self.metrics_csv),
        }
        with open(self.out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[posthoc-distill] done → {self.out_dir}", flush=True)
        return self.out_dir
