"""Generate the dense scaling-law pilot configurations."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "experiments" / "dense_scaling" / "configs"
PILOT_DIR = CONFIG_DIR / "pilot"

MODEL_CONFIGS = (
    "dense_0p16m.json",
    "dense_0p96m.json",
    "dense_5p12m.json",
    "dense_10m.json",
)

STEP_BUDGETS = (50, 125, 250, 750)


def main() -> None:
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, int | float | str]] = []

    for model_filename in MODEL_CONFIGS:
        source = CONFIG_DIR / model_filename
        base_config = json.loads(source.read_text())

        for steps in STEP_BUDGETS:
            config = copy.deepcopy(base_config)
            warmup_steps = max(1, steps // 10)

            config["optimizer"]["warmup_steps"] = warmup_steps
            config["optimizer"]["total_steps"] = steps

            config["training"]["max_steps"] = steps
            config["training"]["validation_interval"] = steps
            config["training"]["checkpoint_interval"] = steps
            config["training"]["logging_interval"] = min(25, steps)

            destination = PILOT_DIR / f"{source.stem}_s{steps:04d}.json"
            destination.write_text(json.dumps(config, indent=2) + "\n")

            batch_size = config["training"]["batch_size"]
            examples_seen = steps * batch_size
            sequence_tokens = examples_seen * config["model"]["max_input_length"]
            answer_tokens = examples_seen * (config["task"]["max_digits"] + 1)
            train_pool = config["training"]["train_pairs"]

            manifest.append(
                {
                    "config": str(destination.relative_to(ROOT)),
                    "model": source.stem,
                    "steps": steps,
                    "warmup_steps": warmup_steps,
                    "batch_size": batch_size,
                    "unique_train_pairs": train_pool,
                    "examples_seen": examples_seen,
                    "sequence_tokens_seen": sequence_tokens,
                    "answer_tokens_seen": answer_tokens,
                    "repetition_ratio": examples_seen / train_pool,
                    "seed": config["training"]["seed"],
                }
            )

            print(f"Created {destination.relative_to(ROOT)}")

    manifest_path = ROOT / "experiments" / "dense_scaling" / "pilot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print()
    print(f"Created {len(manifest)} pilot configurations")
    print(f"Manifest: {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
