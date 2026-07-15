"""From-scratch arithmetic transformer built with JAX and Flax NNX."""

from .config import (
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    ScalingMetadataConfig,
    TaskConfig,
    TrainingConfig,
)
from .model import AdditionTransformer

__all__ = [
    "AdditionTransformer",
    "ExperimentConfig",
    "ModelConfig",
    "OptimizerConfig",
    "ScalingMetadataConfig",
    "TaskConfig",
    "TrainingConfig",
]
__version__ = "0.1.0"
