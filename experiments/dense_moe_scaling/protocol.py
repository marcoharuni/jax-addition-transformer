"""Frozen, architecture-neutral definition of the dense/MoE scaling study."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_addition_transformer.scaling import canonical_fingerprint


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "experiments" / "dense_moe_scaling"
PROTOCOL_PATH = EXPERIMENT_ROOT / "protocol.json"
COMBINED_MANIFEST_PATH = EXPERIMENT_ROOT / "manifest.json"
HORIZONS = (50, 125, 300, 750)
MODEL_SIZES_PER_ARCHITECTURE = 4
RUNS_PER_ARCHITECTURE = MODEL_SIZES_PER_ARCHITECTURE * len(HORIZONS)
FIXED_SEED = 42
PROTOCOL_VERSION = "dense-moe-scaling-v2"
RESULT_SCHEMA_VERSION = "scaling-result-v2"
OUTPUT_ROOTS = {
    "dense": "scaling/dense",
    "moe": "scaling/moe",
    "comparison": "scaling/comparison",
}


def path_has_suffix(path: Path, expected: str) -> bool:
    expected_parts = Path(expected).parts
    return tuple(path.parts[-len(expected_parts) :]) == expected_parts


def protocol_definition() -> dict[str, Any]:
    """Return the complete shared protocol, excluding architecture dimensions."""

    return {
        "protocol_version": PROTOCOL_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "study": {
            "architectures": ["dense", "moe"],
            "model_sizes_per_architecture": MODEL_SIZES_PER_ARCHITECTURE,
            "independent_horizons": list(HORIZONS),
            "runs_per_architecture": RUNS_PER_ARCHITECTURE,
            "fixed_seed": FIXED_SEED,
        },
        "seeds": {
            "split": FIXED_SEED,
            "initialization": FIXED_SEED,
            "sampler": FIXED_SEED,
        },
        "dataset": {
            "domain": "all ordered pairs (a, b) with 0 <= a,b <= 999",
            "pair_id": "1000 * a + b",
            "split_sizes": {"train": 200_000, "validation": 20_000, "test": 780_000},
            "stratification": (
                "operand lengths and carry bits ordered units-tens-hundreds; "
                "deterministic largest-remainder allocation within strata"
            ),
            "training_sampler": (
                "50% uniform over the fixed train pool and 50% uniform over "
                "non-empty train strata, with replacement"
            ),
        },
        "representation": {
            "vocabulary": "0123456789 +=",
            "vocabulary_size": 13,
            "full_record_tokens": 16,
            "model_input_tokens": 15,
            "prompt_tokens": 12,
            "supervised_answer_tokens": 4,
            "answer": "zero-padded four-digit sum in reverse digit order",
            "loss_mask": "last four next-token target positions only",
        },
        "batching": {"training_batch_size": 2048, "evaluation_batch_size": 4000},
        "objective": {
            "formula": (
                "answer_cross_entropy + 0.01 * load_balancing_loss + 0.001 * router_z_loss"
            ),
            "dense_router_losses": "defined as exact zeros",
            "validation_selection": (
                "maximum greedy exact match, then minimum validation answer cross-entropy"
            ),
            "reporting": "validation answer cross-entropy is never combined with auxiliaries",
        },
        "optimizer": {
            "name": "adamw",
            "beta1": 0.9,
            "beta2": 0.99,
            "epsilon": 1e-8,
            "global_norm_clip": 1.0,
            "weight_decay": 0.1,
            "decayed_parameters": "attention and FFN matrix kernels",
            "excluded_from_weight_decay": [
                "embeddings",
                "normalization parameters",
                "biases",
                "MoE router kernels",
            ],
            "schedule": {
                "kind": "linear-warmup cosine-decay",
                "initial_learning_rate": 0.0,
                "peak_learning_rate": 0.001,
                "final_learning_rate": 0.0001,
                "warmup_steps": "floor(horizon / 10), at least one",
                "end": "final learning rate is used by the horizon's last optimizer update",
            },
        },
        "precision": {
            "parameters": "float32",
            "matrix_operands": "float16",
            "matrix_accumulation": "float32",
            "normalization": "float32",
            "router_logits": "float32",
            "output_logits_and_loss": "float32",
        },
        "initialization": {
            "distribution": "independent Normal(0, 0.02^2) matrix parameters",
            "biases": "zeros when present",
            "implementation": "shared package modules for both architectures",
        },
        "rematerialization": {
            "transformer_blocks": True,
            "policy": "jax.checkpoint applied to every dense and MoE transformer block",
        },
        "evaluation": {
            "during_training": "one endpoint validation over the fixed 20,000-pair split",
            "metrics": [
                "teacher-forced answer cross-entropy",
                "answer-token accuracy",
                "greedy exact match",
                "invalid greedy generations",
            ],
            "decoding": "greedy autoregressive generation",
        },
        "checkpointing": {
            "latest_interval_steps": "min(50, horizon)",
            "best": "saved at endpoint validation",
            "latest": "saved at every interval and at the endpoint",
            "contents": ["parameters", "optimizer state", "sampler state", "history"],
            "retention": "best and latest checkpoints are mandatory completed-run artifacts",
        },
        "accounting": {
            "examples_seen": "horizon * 2048",
            "input_exposure": "examples_seen * 15",
            "supervised_exposure": "examples_seen * 4",
            "training_pool_repetition_ratio": "examples_seen / 200000",
            "estimated_flops": "6 * active_parameter_proxy * input_exposure",
            "flop_exclusions": [
                "MoE routing/sorting/dispatch/scatter",
                "rematerialization recomputation",
                "hardware and kernel efficiency",
            ],
        },
        "output_roots": OUTPUT_ROOTS,
    }


def protocol_fingerprint() -> str:
    return canonical_fingerprint(protocol_definition())
