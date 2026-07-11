"""Validated experiment configuration."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class TaskConfig:
    max_digits: int = 3

    def __post_init__(self) -> None:
        if self.max_digits < 1:
            raise ValueError("max_digits must be positive")

    @property
    def answer_digits(self) -> int:
        return self.max_digits + 1

    @property
    def full_sequence_length(self) -> int:
        return 3 * self.max_digits + 7

    @property
    def model_input_length(self) -> int:
        return self.full_sequence_length - 1

    @property
    def prompt_length(self) -> int:
        return 2 * self.max_digits + 6

    @property
    def maximum_operand(self) -> int:
        return 10**self.max_digits - 1


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    n_layers: int = 5
    d_model: int = 320
    n_heads: int = 5
    n_kv_heads: int = 5
    d_ff: int = 2480
    max_input_length: int = 15
    vocab_size: int = 13
    norm_type: str = "layernorm"
    position_type: str = "learned"
    ffn_type: str = "gelu"
    attention_type: str = "mha"
    use_bias: bool = False
    tie_embeddings: bool = True
    compute_dtype: str = "float16"
    parameter_dtype: str = "float32"
    norm_epsilon: float = 1e-5
    rope_base: float = 10000.0
    init_scale: float = 0.02

    def __post_init__(self) -> None:
        choices = {
            "norm_type": (self.norm_type, {"layernorm", "rmsnorm"}),
            "position_type": (self.position_type, {"learned", "rope"}),
            "ffn_type": (self.ffn_type, {"gelu", "relu", "relu2", "swiglu", "geglu"}),
            "attention_type": (self.attention_type, {"mha", "gqa", "mqa"}),
        }
        for name, (value, allowed) in choices.items():
            if value not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        expected_kv = {"mha": self.n_heads, "mqa": 1}.get(self.attention_type)
        if expected_kv is not None and self.n_kv_heads != expected_kv:
            raise ValueError(f"{self.attention_type} requires n_kv_heads={expected_kv}")
        if min(self.n_layers, self.d_model, self.n_heads, self.n_kv_heads, self.d_ff) < 1:
            raise ValueError("model dimensions must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def is_exact_default(self) -> bool:
        return self == ModelConfig()


@dataclasses.dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adamw"
    peak_learning_rate: float = 1e-3
    initial_learning_rate: float = 0.0
    final_learning_rate: float = 1e-4
    warmup_steps: int = 100
    total_steps: int = 5000
    beta1: float = 0.9
    beta2: float = 0.99
    epsilon: float = 1e-8
    weight_decay: float = 0.1
    clip_global_norm: float = 1.0
    momentum: float = 0.9

    def __post_init__(self) -> None:
        if self.name not in {"adamw", "adam", "sgd"}:
            raise ValueError("optimizer name must be adamw, adam, or sgd")
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError("warmup_steps must be in [0, total_steps)")


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    train_pairs: int = 200_000
    validation_pairs: int = 20_000
    test_pairs: int = 780_000
    batch_size: int = 2048
    evaluation_batch_size: int = 4000
    max_steps: int = 5000
    validation_interval: int = 250
    logging_interval: int = 25
    checkpoint_interval: int = 500


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    task: TaskConfig = dataclasses.field(default_factory=TaskConfig)
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        if self.model.max_input_length != self.task.model_input_length:
            raise ValueError("model.max_input_length must match task.model_input_length")
        domain = (self.task.maximum_operand + 1) ** 2
        if (
            self.training.train_pairs + self.training.validation_pairs + self.training.test_pairs
            != domain
        ):
            raise ValueError("split sizes must sum to the complete ordered-pair domain")

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        return cls(
            ModelConfig(**raw["model"]),
            TaskConfig(**raw["task"]),
            OptimizerConfig(**raw["optimizer"]),
            TrainingConfig(**raw["training"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2) + "\n")
