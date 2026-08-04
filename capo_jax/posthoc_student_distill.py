"""Post-hoc persistent student distillation for JAX CAPO checkpoints.

A single actor is initialized from the first chronological student checkpoint
and then updated with pure behavior cloning against each later frozen student.
The source CAPO run, critics, certificates, and teachers are never modified.
"""
from __future__ import annotations

import csv
import json
import pickle
import re
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .networks import Actor

_CKPT_STEP_RE = re.compile(r"checkpoint_(\d+)\.pkl$")


@dataclass
class PosthocDistillConfig:
    checkpoint_dir: str
    out_dir: str = "results/posthoc_student_distill_jax"
    checkpoint_glob: str = "checkpoint_*.pkl"
    start_step: int = 100_000
    end_step: int = 1_000_000
    checkpoint_interval: int = 50_000
    distill_steps_per_checkpoint: int = 1000
    distill_lr: Optional[float] = None
    distill_batch_size: Optional[int] = None
    jit_update_chunk: int = 64
    device: str = "cuda"
    seed: int = 0
    n_episodes: int = 10
    eval_seed0: int = 0
    heldout_batch_size: int = 2048
    heldout_seed: int = 12345
    smoke_steps: Optional[Tuple[int, int]] = None
    resume: bool = True


@dataclass
class DistillRunState:
    initialized_from_first_student: bool = False
    last_completed_step: Optional[int] = None
    optimizer_init_count: int = 0
    optimizer_updates: int = 0


def parse_checkpoint_step(path: Path) -> Optional[int]:
    match = _CKPT_STEP_RE.search(path.name)
    return None if match is None else int(match.group(1))


def discover_checkpoints(
    checkpoint_dir: Path,
    checkpoint_glob: str,
    start_step: int,
    end_step: int,
    checkpoint_interval: int,
    smoke_steps: Optional[Tuple[int, int]] = None,
) -> List[Tuple[int, Path]]:
    """Return a complete, strictly increasing checkpoint sequence."""
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
        if (end_step - start_step) % checkpoint_interval != 0:
            raise ValueError(
                "end_step-start_step must be divisible by checkpoint_interval; "
                f"got start={start_step} end={end_step} interval={checkpoint_interval}"
            )
        expected = list(range(start_step, end_step + 1, checkpoint_interval))

    missing = [step for step in expected if step not in found]
    if missing:
        raise FileNotFoundError(
            f"missing checkpoint(s) under {checkpoint_dir} for steps {missing}. "
            f"found={sorted(found)}"
        )
    return [(step, found[step]) for step in expected]


def _load_pickle(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint must contain a dict: {path}")
    return payload


def _load_run_config(checkpoint_dir: Path, first_ckpt: Path) -> Dict[str, Any]:
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    config = _load_pickle(first_ckpt).get("config")
    if not isinstance(config, dict):
        raise FileNotFoundError(
            f"no config.json in {checkpoint_dir} and no config in {first_ckpt}"
        )
    return config


def load_student_params(ckpt_path: Path, device: Any = None):
    payload = _load_pickle(ckpt_path)
    if "actor" not in payload:
        raise KeyError(f"'actor' missing in {ckpt_path}; keys={list(payload)}")
    params = jax.tree_util.tree_map(jnp.asarray, payload["actor"])
    return params if device is None else jax.device_put(params, device)


def distill_loss_pure_bc(pred_actions, target_actions):
    """Element-wise mean squared action error; no Q or CAPO terms."""
    return jnp.mean((pred_actions - jax.lax.stop_gradient(target_actions)) ** 2)


def parameter_l2_distance(params_a, params_b) -> float:
    leaves = jax.tree_util.tree_map(
        lambda a, b: jnp.sum((a - b) ** 2), params_a, params_b
    )
    return float(jnp.sqrt(sum(jax.tree_util.tree_leaves(leaves))))


def action_mse_between(actor_apply, params_a, params_b, states) -> float:
    return float(
        distill_loss_pure_bc(
            actor_apply(params_a, states), actor_apply(params_b, states)
        )
    )


class PosthocStudentDistiller:
    def __init__(self, cfg: PosthocDistillConfig):
        from .buffer import (
            NormalizeObsWrapper,
            load_d4rl_dataset,
            make_env,
            resolve_jax_device,
        )
        from .trainer import eval_actor

        self.cfg = cfg
        self._eval_actor = eval_actor
        self.device = resolve_jax_device(cfg.device)
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
        data, self.stats, raw_env = load_d4rl_dataset(
            self.env_name,
            normalize=normalize,
            normalize_reward=bool(self.run_cfg.get("normalize_reward", True)),
            device=str(cfg.device),
        )
        raw_env.close()
        self.states = jax.device_put(
            jnp.asarray(data["observations"], dtype=jnp.float32), self.device
        )
        self.state_dim = int(self.states.shape[1])
        self.action_dim = int(data["actions"].shape[1])
        self.hidden = int(self.run_cfg.get("hidden", 256))
        self.max_action = float(self.stats.max_action)
        self.actor_mod = Actor(
            action_dim=self.action_dim,
            max_action=self.max_action,
            hidden=self.hidden,
        )

        def actor_apply(params, states):
            return self.actor_mod.apply({"params": params}, states)

        self.actor_apply = jax.jit(actor_apply, device=self.device)
        self.distill_lr = float(
            cfg.distill_lr
            if cfg.distill_lr is not None
            else self.run_cfg.get("actor_lr", 3e-4)
        )
        self.batch_size = int(
            cfg.distill_batch_size
            if cfg.distill_batch_size is not None
            else self.run_cfg.get("batch_size", 256)
        )
        self.distill_steps = max(0, int(cfg.distill_steps_per_checkpoint))
        self.jit_chunk = max(1, int(cfg.jit_update_chunk))
        self.tx = optax.adam(self.distill_lr)
        self.rng = jax.device_put(jax.random.PRNGKey(cfg.seed), self.device)
        self.distill_params = None
        self.opt_state = None
        self.run_state = DistillRunState()

        heldout_rng = np.random.default_rng(cfg.heldout_seed)
        heldout_idx = heldout_rng.integers(
            0, int(self.states.shape[0]), size=int(cfg.heldout_batch_size)
        )
        self.heldout_states = self.states[jnp.asarray(heldout_idx)]

        self.eval_env = make_env(self.env_name, seed=cfg.seed + 100)
        if normalize:
            self.eval_env = NormalizeObsWrapper(
                self.eval_env, self.stats.state_mean, self.stats.state_std
            )

        self.out_dir = Path(cfg.out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_jsonl = self.out_dir / "distill_metrics.jsonl"
        self.metrics_csv = self.out_dir / "distill_metrics.csv"
        self.state_path = self.out_dir / "distill_state.pkl"
        self._compiled_chunks: Dict[int, Any] = {}

        (self.out_dir / "experiment_config.json").write_text(
            json.dumps(
                {
                    "posthoc": asdict(cfg),
                    "resolved_distill_lr": self.distill_lr,
                    "resolved_distill_batch_size": self.batch_size,
                    "resolved_distill_steps_per_checkpoint": self.distill_steps,
                    "jit_update_chunk": self.jit_chunk,
                    "env": self.env_name,
                    "checkpoint_sequence": [step for step, _ in self.checkpoints],
                    "source_run_config": self.run_cfg,
                    "state_normalization_enabled": normalize,
                    "state_mean": np.asarray(self.stats.state_mean).tolist(),
                    "state_std": np.asarray(self.stats.state_std).tolist(),
                    "pure_bc_loss": "mean((pi_D(states)-stop_gradient(pi_L(states)))**2)",
                },
                indent=2,
            )
        )

    def _chunk_fn(self, length: int):
        if length not in self._compiled_chunks:
            tx = self.tx
            actor_apply = self.actor_apply
            batch_size = self.batch_size
            n_states = int(self.states.shape[0])

            @partial(jax.jit, device=self.device)
            def update_chunk(params, opt_state, rng, student_params, states):
                def body(carry, _):
                    params_i, opt_i, rng_i = carry
                    rng_i, sample_key = jax.random.split(rng_i)
                    idx = jax.random.randint(sample_key, (batch_size,), 0, n_states)
                    batch = states[idx]
                    targets = jax.lax.stop_gradient(actor_apply(student_params, batch))

                    def loss_fn(p):
                        return distill_loss_pure_bc(actor_apply(p, batch), targets)

                    loss, grads = jax.value_and_grad(loss_fn)(params_i)
                    updates, opt_i = tx.update(grads, opt_i, params_i)
                    params_i = optax.apply_updates(params_i, updates)
                    return (params_i, opt_i, rng_i), loss

                (params, opt_state, rng), losses = jax.lax.scan(
                    body, (params, opt_state, rng), xs=None, length=length
                )
                return params, opt_state, rng, losses[-1]

            self._compiled_chunks[length] = update_chunk
        return self._compiled_chunks[length]

    def _initialize(self, student_params, step: int) -> None:
        if self.distill_params is not None or self.run_state.optimizer_init_count:
            raise RuntimeError("distillation actor must be initialized exactly once")
        self.distill_params = jax.tree_util.tree_map(lambda x: jnp.array(x), student_params)
        self.opt_state = self.tx.init(self.distill_params)
        self.run_state.initialized_from_first_student = True
        self.run_state.optimizer_init_count = 1
        print(f"[posthoc-distill-jax] initialized pi_D from student@{step}", flush=True)

    def _bc_updates(self, student_params) -> Tuple[float, float]:
        assert self.distill_params is not None and self.opt_state is not None
        loss_before = action_mse_between(
            self.actor_apply,
            self.distill_params,
            student_params,
            self.heldout_states,
        )
        remaining = self.distill_steps
        while remaining:
            length = min(self.jit_chunk, remaining)
            self.distill_params, self.opt_state, self.rng, _ = self._chunk_fn(length)(
                self.distill_params,
                self.opt_state,
                self.rng,
                student_params,
                self.states,
            )
            remaining -= length
        self.run_state.optimizer_updates += self.distill_steps
        loss_after = action_mse_between(
            self.actor_apply,
            self.distill_params,
            student_params,
            self.heldout_states,
        )
        return loss_before, loss_after

    def _save_state(self, checkpoint_step: int) -> None:
        assert self.distill_params is not None and self.opt_state is not None
        payload = {
            "distill_actor": jax.device_get(self.distill_params),
            "distill_optimizer": jax.device_get(self.opt_state),
            "rng": jax.device_get(self.rng),
            "checkpoint_step": int(checkpoint_step),
            "run_state": asdict(self.run_state),
            "config": asdict(self.cfg),
            "env": self.env_name,
            "hidden": self.hidden,
            "max_action": self.max_action,
            "backend": "jax",
        }
        with open(self.state_path, "wb") as stream:
            pickle.dump(payload, stream)
        step_path = self.out_dir / f"distill_actor_after_{checkpoint_step}.pkl"
        with open(step_path, "wb") as stream:
            pickle.dump(
                {
                    "distill_actor": payload["distill_actor"],
                    "checkpoint_step": int(checkpoint_step),
                    "backend": "jax",
                },
                stream,
            )

    def _try_resume(self) -> Optional[int]:
        if not self.cfg.resume or not self.state_path.exists():
            return None
        payload = _load_pickle(self.state_path)
        last = int(payload["checkpoint_step"])
        self.distill_params = jax.device_put(payload["distill_actor"], self.device)
        self.opt_state = jax.device_put(payload["distill_optimizer"], self.device)
        self.rng = jax.device_put(payload.get("rng", self.rng), self.device)
        state = payload.get("run_state") or {}
        self.run_state = DistillRunState(
            initialized_from_first_student=bool(
                state.get("initialized_from_first_student", True)
            ),
            last_completed_step=last,
            optimizer_init_count=int(state.get("optimizer_init_count", 1)),
            optimizer_updates=int(state.get("optimizer_updates", 0)),
        )
        if self.run_state.optimizer_init_count != 1:
            raise RuntimeError("resumed distiller must have exactly one optimizer init")
        print(f"[posthoc-distill-jax] resume after checkpoint_step={last}", flush=True)
        return last

    def _append_metrics(self, row: Dict[str, Any]) -> None:
        with open(self.metrics_jsonl, "a") as stream:
            stream.write(json.dumps(row) + "\n")
        write_header = not self.metrics_csv.exists()
        with open(self.metrics_csv, "a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def run(self) -> Path:
        last_done = self._try_resume()
        eval_seeds = [self.cfg.eval_seed0 + i for i in range(self.cfg.n_episodes)]
        init_step = self.checkpoints[0][0]
        print(
            f"[posthoc-distill-jax] sequence={[s for s, _ in self.checkpoints]} "
            f"device={self.device} steps/ckpt={self.distill_steps} "
            f"batch={self.batch_size} jit_chunk={self.jit_chunk}",
            flush=True,
        )

        for step, path in self.checkpoints:
            if last_done is not None and step <= last_done:
                continue
            student_params = load_student_params(path, self.device)
            is_init = self.distill_params is None
            if is_init:
                if step != init_step and last_done is None:
                    raise RuntimeError(f"first processed checkpoint must be {init_step}")
                self._initialize(student_params, step)
                loss_before = loss_after = 0.0
            else:
                loss_before, loss_after = self._bc_updates(student_params)

            assert self.distill_params is not None
            student_eval = self._eval_actor(
                self.eval_env,
                self.actor_apply,
                student_params,
                len(eval_seeds),
                episode_seeds=eval_seeds,
            )
            distill_eval = self._eval_actor(
                self.eval_env,
                self.actor_apply,
                self.distill_params,
                len(eval_seeds),
                episode_seeds=eval_seeds,
            )
            student_score = float(student_eval.get("d4rl_score", student_eval["return_mean"]))
            distill_score = float(distill_eval.get("d4rl_score", distill_eval["return_mean"]))
            row = {
                "checkpoint_step": int(step),
                "student_score": student_score,
                "distilled_score": distill_score,
                "distilled_minus_student": distill_score - student_score,
                "distill_steps": 0 if is_init else self.distill_steps,
                "distill_lr": self.distill_lr,
                "distill_batch_size": self.batch_size,
                "distill_loss_before": float(loss_before),
                "distill_loss_after": float(loss_after),
                "action_mse_after": action_mse_between(
                    self.actor_apply,
                    self.distill_params,
                    student_params,
                    self.heldout_states,
                ),
                "actor_parameter_distance": parameter_l2_distance(
                    self.distill_params, student_params
                ),
                "initialized_from_first_student": True,
                "is_initialization_checkpoint": is_init,
                "optimizer_init_count": self.run_state.optimizer_init_count,
                "optimizer_updates": self.run_state.optimizer_updates,
                "checkpoint_path": str(path),
                "backend": "jax",
            }
            self._append_metrics(row)
            self.run_state.last_completed_step = int(step)
            self._save_state(step)
            print(
                f"[posthoc-distill-jax] step={step} init={is_init} "
                f"student={student_score:.2f} distilled={distill_score:.2f} "
                f"mse={row['action_mse_after']:.6g}",
                flush=True,
            )

        summary = {
            "status": "done",
            "checkpoints": [step for step, _ in self.checkpoints],
            "out_dir": str(self.out_dir),
            "backend": "jax",
        }
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        self.eval_env.close()
        return self.out_dir
