"""CAPO — Calibrated Adaptive Policy Optimization (teacher-guided offline RL)."""

from .core import CAPOConfig, CAPOResult, calibrated_adaptive_mpi
from .trainer import CAPOTrainer, TrainConfig

__all__ = [
    "CAPOConfig",
    "CAPOResult",
    "CAPOTrainer",
    "TrainConfig",
    "calibrated_adaptive_mpi",
]
__version__ = "1.0.0"
