"""Generate the matched 10M MoE comparison configurations."""

from __future__ import annotations

import copy
import gc
import json
from pathlib import Path
from typing import Any

import jax
from flax import nnx

from jax_addition_transformer.config import ExperimentConfig, ModelConfig
from jax_addition_transformer.model import (
    AdditionTransformer,
    active_parameter_proxy,
    assert_parameter_count,
)


ROOT = Path(__file__).resolve().parents[2]

DENSE_CONFIG_PATH = (
    ROOT
    / "experiments"
    / "dense_scaling"
    / "configs"
    / "final_models"
    / "dense_10m.json"
)

CONFIG_DIR = ROOT / "experiments" / "moe_comparison" / "configs"
MODEL_CONFIG_PATH = CONFIG_DIR / "moe_10m.json"
RUN_CONFIG_DIR = CONFIG_DIR / "final"

MANIFEST_PATH = (
    ROOT
    / "experiments"
    / "moe_comparison"
    / "comparison_manifest.json"
)

STEPS = 175
SEEDS = (42, 123, 2026)

EXPECTED_DENSE_PARAMETERS = 10_000_000
EXPECTED_MOE_PARAMETERS = 17_942_400
EXPECTED_ACTIVE_PARAMETER_PROXY = 10_006_400


def write_json(path: Path, value: Any) -> None:
    """Write formatted JSON with a final newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def verified_parameter_counts(
    model_config: ModelConfig,
) -> tuple[int, int]:
    """Instantiate the model and verify its real parameter tree."""

    model = AdditionTransformer(
        model_config,
        rngs=nnx.Rngs(params=0),
    )

    total_parameters = assert_parameter_count(model)
    active_parameters = active_parameter_proxy(model)

    del model
    gc.collect()
    jax.clear_caches()

    return total_parameters, active_parameters


def build_comparison(
    write: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Create the matched MoE model and three independent runs."""

    dense_experiment = ExperimentConfig.load(DENSE_CONFIG_PATH)

    dense_total, _ = verified_parameter_counts(
        dense_experiment.model
    )

    if dense_total != EXPECTED_DENSE_PARAMETERS:
        raise ValueError(
            "Dense parameter mismatch: "
            f"expected {EXPECTED_DENSE_PARAMETERS:,}, "
            f"found {dense_total:,}"
        )

    moe_model = ModelConfig.matched_moe(
        dense_experiment.model
    )

    moe_total, active_proxy = verified_parameter_counts(
        moe_model
    )

    if moe_total != EXPECTED_MOE_PARAMETERS:
        raise ValueError(
            "MoE parameter mismatch: "
            f"expected {EXPECTED_MOE_PARAMETERS:,}, "
            f"found {moe_total:,}"
        )

    if active_proxy != EXPECTED_ACTIVE_PARAMETER_PROXY:
        raise ValueError(
            "Active-parameter proxy mismatch: "
            f"expected {EXPECTED_ACTIVE_PARAMETER_PROXY:,}, "
            f"found {active_proxy:,}"
        )

    base_config = copy.deepcopy(
        dense_experiment.as_dict()
    )

    moe_experiment = ExperimentConfig(
        model=moe_model,
        task=dense_experiment.task,
        optimizer=dense_experiment.optimizer,
        training=dense_experiment.training,
    )

    base_config["model"] = moe_experiment.as_dict()["model"]

    if write:
        write_json(MODEL_CONFIG_PATH, base_config)

    warmup_steps = max(1, STEPS // 10)
    manifest: list[dict[str, Any]] = []

    for seed in SEEDS:
        run_config = copy.deepcopy(base_config)

        run_config["optimizer"].update(
            warmup_steps=warmup_steps,
            total_steps=STEPS,
        )

        run_config["training"].update(
            seed=seed,
            max_steps=STEPS,
            validation_interval=STEPS,
            checkpoint_interval=25,
            logging_interval=25,
        )

        run_id = f"moe_10m_s{STEPS:04d}_seed{seed}"
        config_path = RUN_CONFIG_DIR / f"{run_id}.json"

        if write:
            write_json(config_path, run_config)

        batch_size = int(
            run_config["training"]["batch_size"]
        )
        train_pairs = int(
            run_config["training"]["train_pairs"]
        )
        sequence_length = int(
            run_config["model"]["max_input_length"]
        )
        answer_length = (
            int(run_config["task"]["max_digits"]) + 1
        )

        examples_seen = STEPS * batch_size
        sequence_tokens_seen = (
            examples_seen * sequence_length
        )
        answer_tokens_seen = (
            examples_seen * answer_length
        )

        manifest.append(
            {
                "run_id": run_id,
                "config": str(
                    config_path.relative_to(ROOT)
                ),
                "model": "moe_10m",
                "architecture": "moe",
                "dense_reference_parameters": dense_total,
                "total_parameter_count": moe_total,
                "active_parameter_proxy": active_proxy,
                "num_experts": moe_model.num_experts,
                "top_k": moe_model.top_k,
                "expert_d_ff": moe_model.expert_d_ff,
                "seed": seed,
                "steps": STEPS,
                "warmup_steps": warmup_steps,
                "batch_size": batch_size,
                "unique_train_pairs": train_pairs,
                "examples_seen": examples_seen,
                "sequence_tokens_seen": sequence_tokens_seen,
                "answer_tokens_seen": answer_tokens_seen,
                "repetition_ratio": (
                    examples_seen / train_pairs
                ),
                "estimated_training_flops": (
                    6
                    * active_proxy
                    * sequence_tokens_seen
                ),
                "flop_accounting_note": (
                    "6 * active-parameter proxy * "
                    "sequence tokens. This excludes "
                    "routing, sorting, dispatch, "
                    "scatter-add, and hardware effects."
                ),
            }
        )

    if write:
        write_json(MANIFEST_PATH, manifest)

    architecture = {
        "dense_total_parameters": dense_total,
        "moe_total_parameters": moe_total,
        "moe_active_parameter_proxy": active_proxy,
        "num_experts": moe_model.num_experts,
        "top_k": moe_model.top_k,
        "dense_d_ff": dense_experiment.model.d_ff,
        "expert_d_ff": moe_model.expert_d_ff,
        "steps": STEPS,
        "seeds": list(SEEDS),
    }

    return architecture, manifest


def main() -> None:
    """Generate files and print the verified experiment."""

    architecture, manifest = build_comparison()

    print("Matched 10M Dense-versus-MoE comparison")
    print("----------------------------------------")

    for key, value in architecture.items():
        print(f"{key}: {value}")

    print("----------------------------------------")
    print(f"Independent runs: {len(manifest)}")

    for row in manifest:
        print(row["run_id"])


if __name__ == "__main__":
    main()
