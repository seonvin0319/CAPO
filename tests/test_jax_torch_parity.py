import jax
import jax.numpy as jnp
import numpy as np

from capo_jax.jax_torch_parity import run_parity, write_report
from capo_jax.networks import Actor, CriticEnsemble


def test_fixed_batch_jax_torch_parity_smoke(tmp_path):
    state_dim, action_dim, hidden = 3, 2, 8
    ka, kc = jax.random.split(jax.random.PRNGKey(11))
    states = np.linspace(-1, 1, 12, dtype=np.float32).reshape(4, 3)
    actions = np.linspace(-0.5, 0.5, 8, dtype=np.float32).reshape(4, 2)
    actor = Actor(action_dim=action_dim, hidden=hidden)
    critic = CriticEnsemble(n_critics=4, hidden=hidden)
    actor_params = actor.init(ka, jnp.asarray(states))["params"]
    critic_params = critic.init(kc, jnp.asarray(states), jnp.asarray(actions))["params"]
    report = run_parity(
        actor_params=actor_params, critic_params=critic_params,
        teacher_params=actor_params, states=states, actions=actions,
        config={
            "hidden": hidden, "n_critics": 4, "max_action": 1.0,
            "td3_actor_objective": "td3bc_legacy", "alpha": 2.5,
            "actor_lr": 3e-4, "tau_pilot_initial": 0.01,
            "target_action_mse": 0.0025,
        },
    )
    assert report["parameter_conversion"] == "supported"
    assert report["q_aggregation"] == "mean_of_four_critics"
    assert report["actor_actions_before_update"]["difference"]["absolute_max"] < 1e-6
    assert max(value["absolute_max"] for value in report["q_outputs"].values()) < 1e-6
    assert report["final_gate_action"]["jax"] == report["final_gate_action"]["torch"]
    write_report(report, tmp_path / "report.json", tmp_path / "report.md")
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()
