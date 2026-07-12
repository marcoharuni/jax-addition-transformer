"""Generate the final multi-seed dense scaling grid."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = ROOT / "configs" / "exact_10m_t4.json"
CONFIG_ROOT = ROOT / "experiments" / "dense_scaling" / "configs"
MODEL_CONFIG_DIR = CONFIG_ROOT / "final_models"
RUN_CONFIG_DIR = CONFIG_ROOT / "final"
MANIFEST_PATH = ROOT / "experiments" / "dense_scaling" / "final_manifest.json"

MODEL_FAMILY: dict[str, dict[str, int]] = {
    "dense_0p16m": {
        "n_layers": 2,
        "d_model": 64,
        "n_heads": 1,
        "n_kv_heads": 1,
        "d_ff": 496,
    },
    "dense_0p64m": {
        "n_layers": 2,
        "d_model": 128,
        "n_heads": 2,
        "n_kv_heads": 2,
        "d_ff": 992,
    },
    "dense_2p16m": {
        "n_layers": 3,
        "d_model": 192,
        "n_heads": 3,
        "n_kv_heads": 3,
        "d_ff": 1488,
    },
    "dense_5p12m": {
        "n_layers": 4,
        "d_model": 256,
        "n_heads": 4,
        "n_kv_heads": 4,
        "d_ff": 1984,
    },
    "dense_10m": {
        "n_layers": 5,
        "d_model": 320,
        "n_heads": 5,
        "n_kv_heads": 5,
        "d_ff": 2480,
    },
}

EXPECTED_PARAMETER_COUNTS = {
    "dense_0p16m": 162_176,
    "dense_0p64m": 643_840,
    "dense_2p16m": 2_164_608,
    "dense_5p12m": 5_123_584,
    "dense_10m": 10_000_000,
}

STEP_BUDGETS = (50, 75, 100, 125, 175, 250, 400, 750)
SEEDS = (42, 123, 2026)


def parameter_count(model: dict[str, Any]) -> int:
    layers = int(model["n_layers"])
    width = int(model["d_model"])
    feed_forward = int(model["d_ff"])
    vocabulary = int(model["vocab_size"])
    positions = int(model["max_input_length"])

    attention = 4 * width * width
    ffn = 2 * width * feed_forward
    block_norms = 4 * width
    embeddings_and_final_norm = (
        vocabulary * width
        + positions * width
        + 2 * width
    )

    return (
        layers * (attention + ffn + block_norms)
        + embeddings_and_final_norm
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    base = json.loads(BASE_CONFIG_PATH.read_text())
    manifest: list[dict[str, Any]] = []

    MODEL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, dimensions in MODEL_FAMILY.items():
        model_config = copy.deepcopy(base)
        model_config["model"].update(dimensions)

        count = parameter_count(model_config["model"])
        expected = EXPECTED_PARAMETER_COUNTS[model_name]
        if count != expected:
            raise ValueError(
                f"{model_name}: expected {expected:,}, calculated {count:,}"
            )

        model_path = MODEL_CONFIG_DIR / f"{model_name}.json"
        write_json(model_path, model_config)
        print(f"Created {model_path.relative_to(ROOT)} ({count:,} parameters)")

        for seed in SEEDS:
            for steps in STEP_BUDGETS:
                config = copy.deepcopy(model_config)
                warmup_steps = max(1, steps // 10)

                config["optimizer"]["warmup_steps"] = warmup_steps
                config["optimizer"]["total_steps"] = steps

                config["training"]["seed"] = seed
                config["training"]["max_steps"] = steps
                config["training"]["validation_interval"] = steps
                config["training"]["checkpoint_interval"] = steps
                config["training"]["logging_interval"] = min(25, steps)

                run_id = f"{model_name}_s{steps:04d}_seed{seed}"
                config_path = RUN_CONFIG_DIR / f"{run_id}.json"
                write_json(config_path, config)

                batch_size = int(config["training"]["batch_size"])
                examples_seen = steps * batch_size
                sequence_tokens = (
                    examples_seen * int(config["model"]["max_input_length"])
                )
                answer_tokens = (
                    examples_seen * (int(config["task"]["max_digits"]) + 1)
                )
                unique_train_pairs = int(config["training"]["train_pairs"])

                manifest.append(
                    {
                        "run_id": run_id,
                        "config": str(config_path.relative_to(ROOT)),
                        "model": model_name,
                        "parameter_count": count,
                        "seed": seed,
                        "steps": steps,
                        "warmup_steps": warmup_steps,
                        "batch_size": batch_size,
                        "unique_train_pairs": unique_train_pairs,
                        "examples_seen": examples_seen,
                        "sequence_tokens_seen": sequence_tokens,
                        "answer_tokens_seen": answer_tokens,
                        "repetition_ratio": examples_seen / unique_train_pairs,
                        "estimated_training_flops": (
                            6 * count * sequence_tokens
                        ),
                        "schedule": (
                            "linear warmup for 10% of steps, then cosine "
                            "decay to 1e-4 at the final step"
                        ),
                    }
                )

    write_json(MANIFEST_PATH, manifest)

    print()
    print(f"Created {len(MODEL_FAMILY)} model configurations")
    print(f"Created {len(manifest)} independent training configurations")
    print(f"Manifest: {MANIFEST_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
