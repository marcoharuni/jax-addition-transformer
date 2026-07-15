"""Generate the final matched-active MoE scaling grid.

The grid mirrors ``experiments/dense_scaling/final_manifest.json`` exactly:
five architecture sizes, eight exposure budgets, and three random seeds.  Each
MoE keeps the dense model's attention stack and replaces only the dense FFN
with four experts and top-2 routing.  The selected expert width is
``dense_d_ff / top_k``.  This exactly matches selected expert matrix work and
leaves a small, explicitly reported active-parameter overhead from the router.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DENSE_MODEL_DIR = ROOT / "experiments" / "dense_scaling" / "configs" / "final_models"
CONFIG_ROOT = ROOT / "experiments" / "moe_scaling" / "configs"
MODEL_CONFIG_DIR = CONFIG_ROOT / "final_models"
RUN_CONFIG_DIR = CONFIG_ROOT / "final"
MANIFEST_PATH = ROOT / "experiments" / "moe_scaling" / "final_manifest.json"

MODEL_NAMES = (
    "dense_0p16m",
    "dense_0p64m",
    "dense_2p16m",
    "dense_5p12m",
    "dense_10m",
)
MOE_NAMES = {
    "dense_0p16m": "moe_active_0p16m",
    "dense_0p64m": "moe_active_0p64m",
    "dense_2p16m": "moe_active_2p16m",
    "dense_5p12m": "moe_active_5p12m",
    "dense_10m": "moe_active_10m",
}
DENSE_PARAMETER_COUNTS = {
    "dense_0p16m": 162_176,
    "dense_0p64m": 643_840,
    "dense_2p16m": 2_164_608,
    "dense_5p12m": 5_123_584,
    "dense_10m": 10_000_000,
}
EXPECTED_MOE_TOTALS = {
    "moe_active_0p16m": 289_664,
    "moe_active_0p64m": 1_152_768,
    "moe_active_2p16m": 3_881_088,
    "moe_active_5p12m": 9_190_912,
    "moe_active_10m": 17_942_400,
}
EXPECTED_ACTIVE_PROXIES = {
    "moe_active_0p16m": 162_688,
    "moe_active_0p64m": 644_864,
    "moe_active_2p16m": 2_166_912,
    "moe_active_5p12m": 5_127_680,
    "moe_active_10m": 10_006_400,
}
STEP_BUDGETS = (50, 75, 100, 125, 175, 250, 400, 750)
SEEDS = (42, 123, 2026)
NUM_EXPERTS = 4
TOP_K = 2
LOAD_BALANCE_COEFFICIENT = 1e-2
ROUTER_Z_LOSS_COEFFICIENT = 1e-3


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")



def validate_grid_model(model: dict[str, Any]) -> None:
    """Reject architecture changes that would invalidate the closed-form counts."""
    expected = {
        "attention_type": "mha",
        "norm_type": "layernorm",
        "position_type": "learned",
        "ffn_type": "gelu",
        "use_bias": False,
        "tie_embeddings": True,
    }
    for name, value in expected.items():
        if model.get(name) != value:
            raise ValueError(
                f"MoE scaling count formula requires {name}={value!r}; "
                f"found {model.get(name)!r}"
            )
    if int(model["n_kv_heads"]) != int(model["n_heads"]):
        raise ValueError("MoE scaling count formula requires full MHA")


def dense_parameter_count(model: dict[str, Any]) -> int:
    validate_grid_model(model)
    layers = int(model["n_layers"])
    width = int(model["d_model"])
    feed_forward = int(model["d_ff"])
    vocabulary = int(model["vocab_size"])
    positions = int(model["max_input_length"])
    attention = 4 * width * width
    ffn = 2 * width * feed_forward
    norms = 4 * width
    final = vocabulary * width + positions * width + 2 * width
    return layers * (attention + ffn + norms) + final


def moe_parameter_counts(model: dict[str, Any]) -> tuple[int, int]:
    validate_grid_model(model)
    if model.get("architecture") != "moe":
        raise ValueError("moe_parameter_counts requires architecture='moe'")
    layers = int(model["n_layers"])
    width = int(model["d_model"])
    experts = int(model["num_experts"])
    top_k = int(model["top_k"])
    expert_width = int(model["expert_d_ff"])
    vocabulary = int(model["vocab_size"])
    positions = int(model["max_input_length"])

    attention = 4 * width * width
    router = width * experts
    stored_experts = 2 * experts * width * expert_width
    active_experts = 2 * top_k * width * expert_width
    norms = 4 * width
    final = vocabulary * width + positions * width + 2 * width

    total = layers * (attention + router + stored_experts + norms) + final
    active = layers * (attention + router + active_experts + norms) + final
    return total, active


def matched_model(dense_model: dict[str, Any]) -> dict[str, Any]:
    if int(dense_model["d_ff"]) % TOP_K:
        raise ValueError("dense d_ff must be exactly divisible by top_k")
    model = copy.deepcopy(dense_model)
    model.update(
        architecture="moe",
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        expert_d_ff=int(dense_model["d_ff"]) // TOP_K,
        load_balance_loss_coefficient=LOAD_BALANCE_COEFFICIENT,
        router_z_loss_coefficient=ROUTER_Z_LOSS_COEFFICIENT,
    )
    return model


def build_grid(write: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    architecture_rows: list[dict[str, Any]] = []

    for dense_name in MODEL_NAMES:
        source = DENSE_MODEL_DIR / f"{dense_name}.json"
        dense_config = json.loads(source.read_text())
        dense_count = dense_parameter_count(dense_config["model"])
        expected_dense = DENSE_PARAMETER_COUNTS[dense_name]
        if dense_count != expected_dense:
            raise ValueError(
                f"{dense_name}: expected {expected_dense:,} dense parameters, "
                f"calculated {dense_count:,}"
            )

        moe_name = MOE_NAMES[dense_name]
        model_config = copy.deepcopy(dense_config)
        model_config["model"] = matched_model(dense_config["model"])
        total, active = moe_parameter_counts(model_config["model"])

        if total != EXPECTED_MOE_TOTALS[moe_name]:
            raise ValueError(
                f"{moe_name}: expected {EXPECTED_MOE_TOTALS[moe_name]:,} total "
                f"parameters, calculated {total:,}"
            )
        if active != EXPECTED_ACTIVE_PROXIES[moe_name]:
            raise ValueError(
                f"{moe_name}: expected {EXPECTED_ACTIVE_PROXIES[moe_name]:,} active "
                f"proxy, calculated {active:,}"
            )

        active_overhead = active - dense_count
        architecture_rows.append(
            {
                "model": moe_name,
                "dense_reference": dense_name,
                "dense_parameter_count": dense_count,
                "total_parameter_count": total,
                "active_parameter_proxy": active,
                "active_overhead_parameters": active_overhead,
                "active_overhead_fraction": active_overhead / dense_count,
                "n_layers": int(model_config["model"]["n_layers"]),
                "d_model": int(model_config["model"]["d_model"]),
                "dense_d_ff": int(dense_config["model"]["d_ff"]),
                "expert_d_ff": int(model_config["model"]["expert_d_ff"]),
                "num_experts": NUM_EXPERTS,
                "top_k": TOP_K,
            }
        )

        model_path = MODEL_CONFIG_DIR / f"{moe_name}.json"
        if write:
            write_json(model_path, model_config)

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

                run_id = f"{moe_name}_s{steps:04d}_seed{seed}"
                config_path = RUN_CONFIG_DIR / f"{run_id}.json"
                if write:
                    write_json(config_path, config)

                batch_size = int(config["training"]["batch_size"])
                examples_seen = steps * batch_size
                sequence_tokens = examples_seen * int(config["model"]["max_input_length"])
                answer_tokens = examples_seen * (int(config["task"]["max_digits"]) + 1)
                unique_train_pairs = int(config["training"]["train_pairs"])

                manifest.append(
                    {
                        "run_id": run_id,
                        "config": str(config_path.relative_to(ROOT)),
                        "model": moe_name,
                        "architecture": "moe",
                        "dense_reference": dense_name,
                        "dense_reference_parameters": dense_count,
                        "total_parameter_count": total,
                        "active_parameter_proxy": active,
                        "active_overhead_parameters": active_overhead,
                        "active_overhead_fraction": active_overhead / dense_count,
                        "num_experts": NUM_EXPERTS,
                        "top_k": TOP_K,
                        "expert_d_ff": int(config["model"]["expert_d_ff"]),
                        "seed": seed,
                        "steps": steps,
                        "warmup_steps": warmup_steps,
                        "batch_size": batch_size,
                        "unique_train_pairs": unique_train_pairs,
                        "examples_seen": examples_seen,
                        "sequence_tokens_seen": sequence_tokens,
                        "answer_tokens_seen": answer_tokens,
                        "repetition_ratio": examples_seen / unique_train_pairs,
                        "estimated_training_flops": 6 * active * sequence_tokens,
                        "flop_accounting_note": (
                            "6 * active-parameter proxy * sequence tokens. This proxy "
                            "excludes routing, sorting, dispatch, scatter-add, and hardware effects."
                        ),
                        "matching_rule": (
                            "Same attention stack as the dense reference; four experts, "
                            "top-2; expert_d_ff = dense_d_ff / 2. Router parameters are "
                            "active and reported as a small explicit overhead."
                        ),
                    }
                )

    summary = {
        "model_count": len(architecture_rows),
        "run_count": len(manifest),
        "step_budgets": list(STEP_BUDGETS),
        "seeds": list(SEEDS),
        "num_experts": NUM_EXPERTS,
        "top_k": TOP_K,
        "architectures": architecture_rows,
    }

    if write:
        write_json(MANIFEST_PATH, manifest)
        write_json(ROOT / "experiments" / "moe_scaling" / "architecture_summary.json", summary)

    return manifest, summary


def main() -> None:
    manifest, summary = build_grid(write=True)
    print(f"Created {summary['model_count']} matched MoE model configurations")
    print(f"Created {len(manifest)} independent training configurations")
    for row in summary["architectures"]:
        print(
            f"{row['model']}: total={row['total_parameter_count']:,}, "
            f"active={row['active_parameter_proxy']:,}, "
            f"dense={row['dense_parameter_count']:,}, "
            f"overhead={100 * row['active_overhead_fraction']:.3f}%"
        )
    print(f"Manifest: {MANIFEST_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
