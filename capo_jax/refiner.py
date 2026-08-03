"""Functional proximal W2 actor refinement (JAX / Optax).

Solves approximately:
    π' = argmax_π  E[Q(s, π(s))] - (1/(2τ)) ||π(s) - π_center(s)||²
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import optax

from .networks import ActorPolicy, CriticAdapter, q_mean, q_min


class ProximalW2Refiner:
    """Few-step proximal refinement; each candidate gets a fresh Adam state."""

    def __init__(
        self,
        lr: float = 3e-4,
        n_steps: int = 1,
        max_grad_norm: float = 10.0,
        use_min_q: bool = False,
        actor_apply: Optional[Callable] = None,
        critic_apply: Optional[Callable] = None,
    ):
        self.lr = float(lr)
        self.n_steps = int(n_steps)
        self.max_grad_norm = float(max_grad_norm)
        self.use_min_q = bool(use_min_q)
        self.actor_apply = actor_apply
        self.critic_apply = critic_apply
        self._opt = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(self.lr),
        )

    def _get_compiled_refine(self, actor_apply: Callable, critic_apply: Callable):
        cached = getattr(self, "_compiled_refine", None)
        if (
            cached is not None
            and self._compiled_actor_apply is actor_apply
            and self._compiled_critic_apply is critic_apply
        ):
            return cached

        use_min_q = self.use_min_q

        def run(center_params, critic_params, states, tau):
            tau = jnp.maximum(tau, jnp.asarray(1e-8, dtype=states.dtype))
            w2_weight = 1.0 / (2.0 * tau)
            center_actions = jax.lax.stop_gradient(
                actor_apply(center_params, states)
            )

            def loss_fn(params):
                actions = actor_apply(params, states)
                q_stack = critic_apply(critic_params, states, actions)
                q_val = q_min(q_stack) if use_min_q else q_mean(q_stack)
                prox = ((actions - center_actions) ** 2).sum(axis=-1)
                return (-q_val + w2_weight * prox).mean()

            opt_state = self._opt.init(center_params)

            def body(_, carry):
                params_i, opt_state_i = carry
                grads = jax.grad(loss_fn)(params_i)
                updates, opt_state_i = self._opt.update(
                    grads, opt_state_i, params_i
                )
                params_i = optax.apply_updates(params_i, updates)
                return params_i, opt_state_i

            params, _ = jax.lax.fori_loop(
                0, self.n_steps, body, (center_params, opt_state)
            )
            return params

        self._compiled_actor_apply = actor_apply
        self._compiled_critic_apply = critic_apply
        self._compiled_refine = jax.jit(run)
        return self._compiled_refine

    def refine(
        self,
        policy_center: ActorPolicy,
        critics: Sequence[CriticAdapter] | Any,
        tau: float,
        states: jnp.ndarray,
        *,
        gen_critic_params: Any = None,
    ) -> ActorPolicy:
        """Refine from center params; critics may be adapters or raw ensemble params."""
        actor_apply = self.actor_apply or policy_center.apply_fn
        center_params = policy_center.params

        if gen_critic_params is not None:
            critic_apply = self.critic_apply
            assert critic_apply is not None
            run = self._get_compiled_refine(actor_apply, critic_apply)
            params = run(
                center_params,
                gen_critic_params,
                states,
                jnp.asarray(tau, dtype=states.dtype),
            )
            return ActorPolicy(params=params, apply_fn=actor_apply)

        tau = max(float(tau), 1e-8)
        w2_weight = 1.0 / (2.0 * tau)
        center_actions = jax.lax.stop_gradient(actor_apply(center_params, states))
        adapters = list(critics)

        def q_fn(actions):
            qs = jnp.stack([c.q(states, actions) for c in adapters], axis=0)
            return qs.min(axis=0) if self.use_min_q else qs.mean(axis=0)

        def loss_fn(params):
            actions = actor_apply(params, states)
            q_val = q_fn(actions)
            prox = ((actions - center_actions) ** 2).sum(axis=-1)
            return (-q_val + w2_weight * prox).mean()

        opt_state = self._opt.init(center_params)

        def body(_, carry):
            params_i, opt_state_i = carry
            grads = jax.grad(loss_fn)(params_i)
            updates, opt_state_i = self._opt.update(grads, opt_state_i, params_i)
            params_i = optax.apply_updates(params_i, updates)
            return params_i, opt_state_i

        params, _ = jax.lax.fori_loop(
            0, self.n_steps, body, (center_params, opt_state)
        )
        return ActorPolicy(params=params, apply_fn=actor_apply)


def refine_candidate(
    center_params: Any,
    critic_params: Any,
    states: jnp.ndarray,
    tau: float,
    *,
    actor_apply: Callable,
    critic_apply: Callable,
    lr: float = 3e-4,
    n_steps: int = 2,
    max_grad_norm: float = 10.0,
    use_min_q: bool = False,
) -> Any:
    """Pure-function refine used inside JIT-friendly call sites."""
    refiner = ProximalW2Refiner(
        lr=lr,
        n_steps=n_steps,
        max_grad_norm=max_grad_norm,
        use_min_q=use_min_q,
        actor_apply=actor_apply,
        critic_apply=critic_apply,
    )
    center = ActorPolicy(center_params, actor_apply)
    out = refiner.refine(
        center, critics=[], tau=tau, states=states, gen_critic_params=critic_params
    )
    return out.params
