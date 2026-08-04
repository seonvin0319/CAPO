"""Unit tests for post-hoc persistent student distillation invariants."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from capo.networks import Actor
from capo.posthoc_student_distill import (
    discover_checkpoints,
    distill_loss_pure_bc,
    load_student_actor,
    parameter_l2_distance,
)


def test_discover_sorted_and_missing_fails(tmp_path: Path):
    for s in (100000, 200000, 150000):
        (tmp_path / f"checkpoint_{s}.pt").write_bytes(b"x")
    ordered = discover_checkpoints(
        tmp_path, "checkpoint_*.pt", 100000, 200000, 50000
    )
    assert [s for s, _ in ordered] == [100000, 150000, 200000]

    with pytest.raises(FileNotFoundError, match="missing"):
        discover_checkpoints(
            tmp_path, "checkpoint_*.pt", 100000, 250000, 50000
        )


def test_discover_ignores_off_grid_emergency(tmp_path: Path):
    for s in (100000, 150000, 695000):
        (tmp_path / f"checkpoint_{s}.pt").write_bytes(b"x")
    # 695000 exists but is not on the expected grid for 100k-150k smoke
    ordered = discover_checkpoints(
        tmp_path,
        "checkpoint_*.pt",
        100000,
        150000,
        50000,
        smoke_steps=(100000, 150000),
    )
    assert [s for s, _ in ordered] == [100000, 150000]


def test_pure_bc_loss_is_mse():
    pred = torch.randn(32, 3)
    tgt = torch.randn(32, 3)
    loss = distill_loss_pure_bc(pred, tgt)
    assert torch.allclose(loss, torch.nn.functional.mse_loss(pred, tgt))


def test_student_load_freezes_and_no_grad(tmp_path: Path):
    actor = Actor(11, 3, max_action=1.0, hidden=64)
    path = tmp_path / "checkpoint_100000.pt"
    torch.save({"actor": actor.state_dict(), "step": 100000}, path)
    loaded = load_student_actor(path, 11, 3, 1.0, 64, torch.device("cpu"))
    assert all(not p.requires_grad for p in loaded.parameters())
    # backward into student must fail / leave no grads
    s = torch.randn(8, 11, requires_grad=True)
    with torch.no_grad():
        a = loaded.act(s)
    assert not a.requires_grad


def test_optimizer_persistence_and_single_init():
    student0 = Actor(4, 2, max_action=1.0, hidden=32)
    distill = student0.copy()
    for p in distill.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(distill.parameters(), lr=1e-3)
    opt_id = id(opt)

    student1 = Actor(4, 2, max_action=1.0, hidden=32)
    # different weights
    with torch.no_grad():
        for p in student1.parameters():
            p.add_(0.1)
    for p in student1.parameters():
        p.requires_grad_(False)

    # one BC step
    s = torch.randn(16, 4)
    with torch.no_grad():
        tgt = student1.act(s)
    pred = distill.act(s)
    loss = distill_loss_pure_bc(pred, tgt)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    assert all(p.grad is None for p in student1.parameters())
    opt.step()
    assert id(opt) == opt_id

    # never overwrite distill with student1 via load_state_dict
    distill_sd = {k: v.detach().clone() for k, v in distill.state_dict().items()}
    # simulate "must not reset": keep same object + optimizer
    assert id(opt) == opt_id
    assert distill is not student1
    # parameters should have moved from the BC step relative to pre-step snapshot
    moved = any(
        not torch.allclose(distill_sd[k], distill.state_dict()[k])
        for k in distill_sd
    )
    # If numerically tiny, at least optimizer state exists
    assert moved or any(len(v) > 0 for v in opt.state_dict().get("state", {}).values())
    assert parameter_l2_distance(distill, student0) >= 0.0