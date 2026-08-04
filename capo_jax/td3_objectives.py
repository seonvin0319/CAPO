"""Pure TD3 actor objectives shared by training and parity tests."""
from __future__ import annotations

import jax
import jax.numpy as jnp


def td3bc_legacy_actor_components(
    q_stack, actor_actions, dataset_actions, *, alpha: float = 2.5, eps: float = 1e-4
):
    """Original local 4-critic convention: mean-Q, alpha scaling, unit BC."""
    q_values = jnp.mean(q_stack, axis=0)
    q_scale = jax.lax.stop_gradient(jnp.mean(jnp.abs(q_values)) + eps)
    td3bc_scale = jax.lax.stop_gradient(jnp.asarray(alpha) / q_scale)
    q_term = -td3bc_scale * jnp.mean(q_values)
    bc = jnp.mean((actor_actions - dataset_actions) ** 2)
    return {
        "q_values": q_values,
        "q_scale": q_scale,
        "td3bc_scale": td3bc_scale,
        "q_actor_term": q_term,
        "data_bc_loss": bc,
        "total_actor_loss": q_term + bc,
    }
