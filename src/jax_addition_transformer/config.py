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
    architecture: str = "dense"
    num_experts: int | None = None
    top_k: int | None = None
    expert_d_ff: int | None = None
    load_balance_loss_coefficient: float = 0.0
    router_z_loss_coefficient: float = 0.0
    rematerialize_blocks: bool | None = None

    def __post_init__(self) -> None:
        choices = {
            "architecture": (self.architecture, {"dense", "moe"}),
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
        if self.load_balance_loss_coefficient < 0 or self.router_z_loss_coefficient < 0:
            raise ValueError("routing auxiliary-loss coefficients must be non-negative")
        moe_fields = (self.num_experts, self.top_k, self.expert_d_ff)
        if self.architecture == "dense":
            if any(value is not None for value in moe_fields):
                raise ValueError("dense architecture cannot set MoE-only dimension fields")
        else:
            if any(value is None for value in moe_fields):
                raise ValueError("MoE architecture requires num_experts, top_k, and expert_d_ff")
            if self.num_experts < 1:
                raise ValueError("num_experts must be at least one")
            if self.top_k < 1:
                raise ValueError("top_k must be at least one")
            if self.top_k > self.num_experts:
                raise ValueError("top_k must not exceed num_experts")
            if self.expert_d_ff < 1:
                raise ValueError("expert_d_ff must be positive")
            if self.ffn_type in {"swiglu", "geglu"}:
                raise ValueError("MoE currently supports gelu, relu, and relu2 FFNs")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def is_exact_default(self) -> bool:
        return self == ModelConfig()

    @property
    def uses_rematerialization(self) -> bool:
        """Resolve the explicit policy while preserving legacy MoE behavior."""

        if self.rematerialize_blocks is not None:
            return self.rematerialize_blocks
        return self.architecture == "moe"

    @classmethod
    def matched_moe(
        cls,
        dense: ModelConfig,
        *,
        num_experts: int = 4,
        top_k: int = 2,
        load_balance_loss_coefficient: float = 0.01,
        router_z_loss_coefficient: float = 0.001,
    ) -> ModelConfig:
        """Build an MoE whose selected expert width matches dense FFN matmul work."""
        if dense.architecture != "dense":
            raise ValueError("matched_moe requires a dense source configuration")
        if top_k < 1 or dense.d_ff % top_k:
            raise ValueError("dense d_ff must be exactly divisible by top_k")
        return dataclasses.replace(
            dense,
            architecture="moe",
            num_experts=num_experts,
            top_k=top_k,
            expert_d_ff=dense.d_ff // top_k,
            load_balance_loss_coefficient=load_balance_loss_coefficient,
            router_z_loss_coefficient=router_z_loss_coefficient,
        )


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
    end_at_last_update: bool = False

    def __post_init__(self) -> None:
        if self.name not in {"adamw", "adam", "sgd"}:
            raise ValueError("optimizer name must be adamw, adam, or sgd")
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError("warmup_steps must be in [0, total_steps)")
        if self.end_at_last_update and self.total_steps < 2:
            raise ValueError("end_at_last_update requires at least two total steps")


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    split_seed: int | None = None
    initialization_seed: int | None = None
    sampler_seed: int | None = None
    train_pairs: int = 200_000
    validation_pairs: int = 20_000
    test_pairs: int = 780_000
    batch_size: int = 2048
    evaluation_batch_size: int = 4000
    max_steps: int = 5000
    validation_interval: int = 250
    logging_interval: int = 25
    checkpoint_interval: int = 500

    @property
    def resolved_split_seed(self) -> int:
        return self.seed if self.split_seed is None else self.split_seed

    @property
    def resolved_initialization_seed(self) -> int:
        return self.seed if self.initialization_seed is None else self.initialization_seed

    @property
    def resolved_sampler_seed(self) -> int:
        return self.seed if self.sampler_seed is None else self.sampler_seed


@dataclasses.dataclass(frozen=True)
class ScalingMetadataConfig:
    """Identity and protocol binding for a generated scaling run."""

    schema_version: str
    protocol_version: str
    protocol_fingerprint: str
    run_id: str
    model_id: str
    dense_reference: str
    horizon_steps: int

    def __post_init__(self) -> None:
        if self.schema_version != "scaling-result-v2":
            raise ValueError("unsupported scaling result schema")
        if len(self.protocol_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.protocol_fingerprint
        ):
            raise ValueError("protocol_fingerprint must be a lowercase SHA-256 digest")
        if not self.run_id or not self.model_id or not self.dense_reference:
            raise ValueError("scaling run identity fields must be non-empty")
        if self.horizon_steps < 1:
            raise ValueError("horizon_steps must be positive")


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    task: TaskConfig = dataclasses.field(default_factory=TaskConfig)
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    scaling: ScalingMetadataConfig | None = None

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
        raw = dataclasses.asdict(self)
        if self.model.architecture == "dense":
            for name in (
                "architecture",
                "num_experts",
                "top_k",
                "expert_d_ff",
            ):
                raw["model"].pop(name)
            if not self.model.load_balance_loss_coefficient:
                raw["model"].pop("load_balance_loss_coefficient")
            if not self.model.router_z_loss_coefficient:
                raw["model"].pop("router_z_loss_coefficient")
        if self.model.rematerialize_blocks is None:
            raw["model"].pop("rematerialize_blocks")
        for name in ("split_seed", "initialization_seed", "sampler_seed"):
            if raw["training"][name] is None:
                raw["training"].pop(name)
        if not self.optimizer.end_at_last_update:
            raw["optimizer"].pop("end_at_last_update")
        if self.scaling is None:
            raw.pop("scaling")
        return raw

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
            ScalingMetadataConfig(**raw["scaling"]) if raw.get("scaling") else None,
        )

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2) + "\n")
