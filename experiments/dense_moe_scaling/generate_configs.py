"""Generate the 16 dense and 16 matched-active MoE run configurations."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from jax_addition_transformer.config import ExperimentConfig
from jax_addition_transformer.scaling import canonical_fingerprint

from experiments.moe_scaling.configs import moe_parameter_counts

from .protocol import (
    COMBINED_MANIFEST_PATH,
    EXPERIMENT_ROOT,
    FIXED_SEED,
    HORIZONS,
    OUTPUT_ROOTS,
    PROTOCOL_PATH,
    PROTOCOL_VERSION,
    RESULT_SCHEMA_VERSION,
    ROOT,
    RUNS_PER_ARCHITECTURE,
    protocol_definition,
)


DENSE_MODEL_DIR = ROOT / "experiments" / "dense_scaling" / "configs" / "final_models"
MOE_MODEL_DIR = ROOT / "experiments" / "moe_scaling" / "configs" / "final_models"
CONFIG_DIR = EXPERIMENT_ROOT / "configs"
DENSE_MANIFEST_PATH = EXPERIMENT_ROOT / "dense_manifest.json"
MOE_MANIFEST_PATH = EXPERIMENT_ROOT / "moe_manifest.json"

DENSE_MODELS = (
    "dense_0p16m",
    "dense_0p64m",
    "dense_2p16m",
    "dense_10m",
)
MOE_BY_DENSE = {
    "dense_0p16m": "moe_active_0p16m",
    "dense_0p64m": "moe_active_0p64m",
    "dense_2p16m": "moe_active_2p16m",
    "dense_10m": "moe_active_10m",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def dense_parameter_count(model: dict[str, Any]) -> int:
    layers = int(model["n_layers"])
    width = int(model["d_model"])
    feed_forward = int(model["d_ff"])
    vocabulary = int(model["vocab_size"])
    positions = int(model["max_input_length"])
    return layers * (4 * width * width + 2 * width * feed_forward + 4 * width) + (
        vocabulary * width + positions * width + 2 * width
    )


def configured_run(
    source: dict[str, Any],
    *,
    architecture: str,
    model_id: str,
    dense_reference: str,
    horizon: int,
    protocol_hash: str,
) -> tuple[dict[str, Any], str]:
    config = copy.deepcopy(source)
    config["model"]["rematerialize_blocks"] = True
    config["model"]["load_balance_loss_coefficient"] = 0.01
    config["model"]["router_z_loss_coefficient"] = 0.001
    config["optimizer"].update(
        warmup_steps=max(1, horizon // 10),
        total_steps=horizon,
        end_at_last_update=True,
    )
    config["training"].update(
        seed=FIXED_SEED,
        split_seed=FIXED_SEED,
        initialization_seed=FIXED_SEED,
        sampler_seed=FIXED_SEED,
        max_steps=horizon,
        validation_interval=horizon,
        checkpoint_interval=min(50, horizon),
        logging_interval=min(25, horizon),
    )
    run_id = f"{model_id}_h{horizon:04d}"
    config["scaling"] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_fingerprint": protocol_hash,
        "run_id": run_id,
        "model_id": model_id,
        "dense_reference": dense_reference,
        "horizon_steps": horizon,
    }
    validated = ExperimentConfig.from_dict(config)
    if validated.model.architecture != architecture:
        raise ValueError(f"{model_id}: expected architecture {architecture}")
    return validated.as_dict(), validated.fingerprint


def manifest_row(
    *,
    config: dict[str, Any],
    config_path: Path,
    config_hash: str,
    protocol_hash: str,
    architecture: str,
    model_id: str,
    dense_reference: str,
    horizon: int,
) -> dict[str, Any]:
    model = config["model"]
    stored = dense_parameter_count(model)
    active = stored
    if architecture == "moe":
        stored, active = moe_parameter_counts(model)
    examples = horizon * int(config["training"]["batch_size"])
    input_tokens = examples * 15
    supervised_tokens = examples * 4
    return {
        "run_id": config["scaling"]["run_id"],
        "architecture": architecture,
        "model_id": model_id,
        "dense_reference": dense_reference,
        "horizon_steps": horizon,
        "warmup_steps": int(config["optimizer"]["warmup_steps"]),
        "split_seed": FIXED_SEED,
        "initialization_seed": FIXED_SEED,
        "sampler_seed": FIXED_SEED,
        "config": str(config_path.relative_to(ROOT)),
        "config_fingerprint": config_hash,
        "protocol_fingerprint": protocol_hash,
        "stored_parameters": stored,
        "active_parameter_proxy": active,
        "examples_seen": examples,
        "input_tokens_15_seen": input_tokens,
        "supervised_tokens_4_seen": supervised_tokens,
        "training_pool_repetition_ratio": examples / 200_000,
        "estimated_training_flops": 6 * active * input_tokens,
        "output_root": OUTPUT_ROOTS[architecture],
    }


def build_grid(write: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = protocol_definition()
    protocol_hash = canonical_fingerprint(protocol)
    dense_rows: list[dict[str, Any]] = []
    moe_rows: list[dict[str, Any]] = []
    generated_configs: list[tuple[Path, dict[str, Any]]] = []

    for dense_name in DENSE_MODELS:
        sources = {
            "dense": json.loads((DENSE_MODEL_DIR / f"{dense_name}.json").read_text()),
            "moe": json.loads((MOE_MODEL_DIR / f"{MOE_BY_DENSE[dense_name]}.json").read_text()),
        }
        for architecture, source in sources.items():
            model_id = dense_name if architecture == "dense" else MOE_BY_DENSE[dense_name]
            target = dense_rows if architecture == "dense" else moe_rows
            for horizon in HORIZONS:
                config, config_hash = configured_run(
                    source,
                    architecture=architecture,
                    model_id=model_id,
                    dense_reference=dense_name,
                    horizon=horizon,
                    protocol_hash=protocol_hash,
                )
                config_path = CONFIG_DIR / architecture / f"{config['scaling']['run_id']}.json"
                generated_configs.append((config_path, config))
                target.append(
                    manifest_row(
                        config=config,
                        config_path=config_path,
                        config_hash=config_hash,
                        protocol_hash=protocol_hash,
                        architecture=architecture,
                        model_id=model_id,
                        dense_reference=dense_name,
                        horizon=horizon,
                    )
                )

    if len(dense_rows) != RUNS_PER_ARCHITECTURE or len(moe_rows) != RUNS_PER_ARCHITECTURE:
        raise AssertionError(
            "grid must contain exactly "
            f"{RUNS_PER_ARCHITECTURE} dense and {RUNS_PER_ARCHITECTURE} MoE runs"
        )
    if write:
        expected_paths = {path for path, _ in generated_configs}
        for architecture in ("dense", "moe"):
            for stale_path in (CONFIG_DIR / architecture).glob("*.json"):
                if stale_path not in expected_paths:
                    stale_path.unlink()
        for config_path, config in generated_configs:
            write_json(config_path, config)
        write_json(PROTOCOL_PATH, {**protocol, "protocol_fingerprint": protocol_hash})
        write_json(DENSE_MANIFEST_PATH, dense_rows)
        write_json(MOE_MANIFEST_PATH, moe_rows)
        write_json(
            COMBINED_MANIFEST_PATH,
            {
                "protocol_version": PROTOCOL_VERSION,
                "protocol_fingerprint": protocol_hash,
                "dense_manifest": str(DENSE_MANIFEST_PATH.relative_to(ROOT)),
                "moe_manifest": str(MOE_MANIFEST_PATH.relative_to(ROOT)),
                "dense_runs": dense_rows,
                "moe_runs": moe_rows,
            },
        )
    return dense_rows, moe_rows


def main() -> None:
    dense, moe = build_grid(write=True)
    print(f"Generated {len(dense)} dense and {len(moe)} MoE independent run configs.")
    print(f"Protocol: {PROTOCOL_PATH.relative_to(ROOT)}")
    print(f"Manifest: {COMBINED_MANIFEST_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
