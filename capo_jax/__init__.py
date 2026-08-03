"""CAPO (JAX) — Calibrated Adaptive Policy Optimization.

The trainer is imported lazily so lightweight modules such as ``core`` and
``networks`` do not import Gym/D4RL (and initialize MuJoCo) as a side effect.
"""

from .core import CAPOConfig, CAPOResult, calibrated_adaptive_mpi

__all__ = [
    "CAPOConfig",
    "CAPOResult",
    "CAPOTrainer",
    "TrainConfig",
    "calibrated_adaptive_mpi",
]
__version__ = "1.0.0-jax"


def __getattr__(name):
    if name in {"CAPOTrainer", "TrainConfig"}:
        from .trainer import CAPOTrainer, TrainConfig

        return {"CAPOTrainer": CAPOTrainer, "TrainConfig": TrainConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
