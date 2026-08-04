from dataclasses import dataclass, replace
from types import SimpleNamespace

import jax.numpy as jnp

from capo_jax.gate_runtime import apply_teacher_replace_gate


@dataclass
class State:
    actor_params: object
    teacher_params: object
    quarantined_params: object
    has_teacher: object
    teacher_n: object
    teacher_tau: object
    has_quarantined: object
    quarantined_n: object
    quarantined_tau: object
    q_scale: object
    total_it: object
    rng: object

    def replace(self, **kwargs):
        return replace(self, **kwargs)


class Policy:
    def __init__(self, params):
        self.params = params


def fake_trainer(run_dir, paired):
    params = {"x": jnp.asarray([1.0])}
    state = State(
        actor_params=params, teacher_params=params, quarantined_params=params,
        has_teacher=jnp.asarray(0.0), teacher_n=jnp.asarray(0),
        teacher_tau=jnp.asarray(float("nan")), has_quarantined=jnp.asarray(0.0),
        quarantined_n=jnp.asarray(0), quarantined_tau=jnp.asarray(float("nan")),
        q_scale=jnp.asarray(1.0), total_it=jnp.asarray(100),
        rng=jnp.asarray([123, 456], dtype=jnp.uint32),
    )
    cfg = SimpleNamespace(
        replace_cert_margin=0.0, use_replace_gate=True,
        stale_incumbent_action="disable", nstar_zero_action="revalidate_current",
        teacher_hold=True, paired_eval_episodes=paired, paired_eval_seed0=10_000,
        save_refresh_actors=False,
    )
    return SimpleNamespace(
        cfg=cfg, state=state, gate_counts={
            "replace_count": 0, "disable_count": 0, "quarantine_count": 0,
            "reactivation_count": 0, "stale_count": 0,
        }, last_gate_action="", run_dir=run_dir,
        _policy=lambda p: Policy(p), _pairwise_lcb_cert=lambda *args: 0.1,
        actor_apply=lambda p, s: s,
    )


def test_paired_diagnostic_does_not_perturb_training_rng(tmp_path):
    result = SimpleNamespace(
        selected_n=1, accepted=True, final_policy=Policy({"x": jnp.asarray([2.0])}),
        selected_tau=[0.01], certificates=[0.1], movements=[0.1],
    )
    calls = []
    paired_fn = lambda *args: calls.append(args[-1]) or {
        "paired_delta_d4rl": 1.0, "teacher_d4rl_score": 2.0,
    }
    disabled = fake_trainer(tmp_path / "off", 0)
    enabled = fake_trainer(tmp_path / "on", 1)
    disabled.run_dir.mkdir(); enabled.run_dir.mkdir()
    key_before = enabled.state.rng.copy()
    args = (result, jnp.zeros((2, 1)), jnp.zeros((2, 1)), (), {})
    apply_teacher_replace_gate(disabled, *args, paired_fn)
    apply_teacher_replace_gate(enabled, *args, paired_fn)
    assert jnp.array_equal(enabled.state.rng, key_before)
    assert jnp.array_equal(enabled.state.actor_params["x"], disabled.state.actor_params["x"])
    assert enabled.last_gate_action == disabled.last_gate_action
    assert calls == [[10_000]]
