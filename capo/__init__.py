"""CaPO — Calibrated Adaptive Multi-step Proximal Improvement (teacher-guided offline RL)."""

from .core import CAMPIConfig, CAMPIResult, calibrated_adaptive_mpi
from .trainer import CaPOTrainer, TrainConfig

__all__ = [
    "CAMPIConfig",
    "CAMPIResult",
    "CaPOTrainer",
    "TrainConfig",
    "calibrated_adaptive_mpi",
]
__version__ = "1.0.0"
