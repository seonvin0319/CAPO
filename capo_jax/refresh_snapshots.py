"""Stable, actor-only CAPO refresh snapshots for offline diagnostics."""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import jax
import numpy as np

FORMAT = "capo_jax_refresh_actor_bundle"
VERSION = 1


def _actor_bytes(params: Any) -> bytes:
    host_params = jax.device_get(params)
    return pickle.dumps(host_params, protocol=5)


def actor_checksum(params: Any) -> str:
    return hashlib.sha256(_actor_bytes(params)).hexdigest()


def save_refresh_actor_bundle(
    *,
    run_dir: Path,
    trainer: Any,
    refresh_step: int,
    roles: Mapping[str, Optional[Any]],
    refresh_row: Mapping[str, Any],
) -> Path:
    """Save unique actor pytrees and role references without touching RNG state."""
    actors: Dict[str, Dict[str, Any]] = {}
    role_refs: Dict[str, Optional[str]] = {}
    for role, params in roles.items():
        if params is None:
            role_refs[role] = None
            continue
        raw = _actor_bytes(params)
        checksum = hashlib.sha256(raw).hexdigest()
        role_refs[role] = checksum
        if checksum not in actors:
            actors[checksum] = {
                "sha256": checksum,
                "serialized_nbytes": len(raw),
                "params": pickle.loads(raw),
            }

    cfg = trainer.cfg
    bundle = {
        "format": FORMAT,
        "version": VERSION,
        "run_id": cfg.run_id,
        "environment": cfg.env,
        "seed": int(cfg.seed),
        "refresh_step": int(refresh_step),
        "roles": role_refs,
        "actors": actors,
        "actor_architecture": {
            "module": "capo_jax.networks.Actor",
            "state_dim": int(trainer.state_dim),
            "action_dim": int(trainer.action_dim),
            "hidden": int(cfg.hidden),
            "max_action": float(trainer.max_action),
        },
        "observation_normalization": {
            "enabled": bool(cfg.normalize),
            "mean": np.asarray(trainer.stats.state_mean, dtype=np.float32),
            "std": np.asarray(trainer.stats.state_std, dtype=np.float32),
        },
        "N_star": int(refresh_row.get("N_star", 0)),
        "selected_tau_values": list(
            refresh_row.get("selected_tau_per_ladder_step", [])
        ),
        "gate_certificates": {
            key: refresh_row.get(key)
            for key in ("C_student_to_new", "C_student_to_old", "C_old_to_new")
        },
        "gate_action": refresh_row.get("gate_action"),
        "teacher_state_before": refresh_row.get("previous_teacher_state"),
        "teacher_state_after": refresh_row.get("next_teacher_state"),
        "resolved_config": asdict(cfg),
    }
    payload = pickle.dumps(bundle, protocol=5)
    bundle["bundle_payload_sha256"] = hashlib.sha256(payload).hexdigest()

    target_dir = Path(run_dir) / "refresh_actors" / f"step_{int(refresh_step)}"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "actors.pkl"
    tmp = target.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as stream:
        pickle.dump(bundle, stream, protocol=5)
        stream.flush()
    tmp.replace(target)
    metadata = {
        "format": FORMAT,
        "version": VERSION,
        "run_id": cfg.run_id,
        "environment": cfg.env,
        "seed": int(cfg.seed),
        "refresh_step": int(refresh_step),
        "roles": role_refs,
        "unique_actor_count": len(actors),
        "actor_checksums": sorted(actors),
        "bundle_file": target.name,
        "bundle_file_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "bundle_nbytes": target.stat().st_size,
    }
    metadata_path = target_dir / "metadata.json"
    metadata_tmp = metadata_path.with_suffix(".json.tmp")
    metadata_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    metadata_tmp.replace(metadata_path)
    return target


def load_refresh_actor_bundle(path: Path, *, verify: bool = True) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "rb") as stream:
        bundle = pickle.load(stream)
    if bundle.get("format") != FORMAT or int(bundle.get("version", -1)) != VERSION:
        raise ValueError(f"unsupported refresh actor bundle: {path}")
    if verify:
        for checksum, actor in bundle["actors"].items():
            actual = actor_checksum(actor["params"])
            if checksum != actual or actor.get("sha256") != actual:
                raise ValueError(f"actor checksum mismatch in {path}: {checksum}")
        for role, checksum in bundle["roles"].items():
            if checksum is not None and checksum not in bundle["actors"]:
                raise ValueError(f"role {role!r} references missing actor {checksum}")
    return bundle


def actor_params_for_role(bundle: Mapping[str, Any], role: str) -> Any:
    checksum = bundle["roles"].get(role)
    if checksum is None:
        raise KeyError(f"actor role unavailable: {role}")
    return bundle["actors"][checksum]["params"]
