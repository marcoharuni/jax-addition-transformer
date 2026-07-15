"""Shared fingerprints and normalized records for scaling experiments."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .config import ExperimentConfig


RESULT_SCHEMA_VERSION = "scaling-result"


def canonical_fingerprint(value: Any) -> str:
    """Return SHA-256 over canonical UTF-8 JSON."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_normalized_result(
    config: ExperimentConfig,
    summary: dict[str, Any],
    environment: dict[str, Any],
    split_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build the architecture-neutral scaling result written by training."""

    metadata = config.scaling
    if metadata is None:
        raise ValueError("normalized scaling results require scaling metadata")
    if metadata.horizon_steps != config.training.max_steps:
        raise ValueError("scaling horizon does not match training.max_steps")

    best = summary.get("best") or {}
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete",
        "run_id": metadata.run_id,
        "identity": {
            "protocol_version": metadata.protocol_version,
            "architecture": config.model.architecture,
            "model_id": metadata.model_id,
            "dense_reference": metadata.dense_reference,
            "horizon_steps": metadata.horizon_steps,
        },
        "fingerprints": {
            "protocol": metadata.protocol_fingerprint,
            "config": config.fingerprint,
            "split": split_metadata["fingerprint"],
        },
        "seeds": {
            "split": config.training.resolved_split_seed,
            "initialization": config.training.resolved_initialization_seed,
            "sampler": config.training.resolved_sampler_seed,
        },
        "exposure": {
            "examples_seen": summary["examples_seen"],
            "input_tokens_15_seen": summary["input_tokens_seen"],
            "supervised_tokens_4_seen": summary["answer_tokens_seen"],
            "training_pool_size": config.training.train_pairs,
            "training_pool_repetition_ratio": summary["repetition_ratio"],
        },
        "parameters": {
            "stored": summary["total_parameter_count"],
            "active_proxy": summary["active_parameter_proxy"],
        },
        "compute": {
            "estimated_training_flops": summary["estimated_training_flops"],
            "estimator": "6 * active_parameter_proxy * 15-token input exposure",
            "exclusions": [
                "MoE routing",
                "sorting",
                "dispatch",
                "scatter-add",
                "rematerialization recomputation",
                "hardware and kernel efficiency",
            ],
        },
        "timing_seconds": {
            "compilation": summary["total_compilation_seconds"],
            "training_steps": summary["training_step_seconds"],
            "validation": summary["validation_seconds"],
            "steady_state_training": summary["steady_state_training_seconds"],
            "run_segment_wallclock": summary["training_segment_seconds"],
        },
        "losses": {
            "validation_answer_cross_entropy": best.get("loss"),
            "final_training_answer_cross_entropy": summary["final_train_language_model_loss"],
            "final_load_balancing_loss": summary["final_load_balancing_loss"],
            "final_router_z_loss": summary["final_router_z_loss"],
            "final_weighted_load_balancing_loss": summary["final_weighted_load_balancing_loss"],
            "final_weighted_router_z_loss": summary["final_weighted_router_z_loss"],
            "final_total_optimized_loss": summary["final_total_optimized_loss"],
        },
        "accuracy": {
            "validation_greedy_exact_match": best.get("greedy_exact_match"),
            "final_training_answer_token_accuracy": summary["final_answer_token_accuracy"],
        },
        "router": {
            "num_experts": summary["num_experts"],
            "top_k": summary["top_k"],
            "expert_d_ff": summary["expert_d_ff"],
            "load_balance_loss_coefficient": summary["load_balance_loss_coefficient"],
            "router_z_loss_coefficient": summary["router_z_loss_coefficient"],
            "metrics": summary["routing_metrics"],
        },
        "environment": environment,
    }
