"""Validate and tabulate one completed 16-run architecture collection."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from .protocol import (
    HORIZONS,
    OUTPUT_ROOTS,
    RESULT_SCHEMA_VERSION,
    RUNS_PER_ARCHITECTURE,
    path_has_suffix,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any, label: str, run_id: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{run_id}: non-finite {label}")
    return result


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    run_id = str(record["run_id"])
    result = record["result"]
    identity = result["identity"]
    exposure = result["exposure"]
    parameters = result["parameters"]
    timing = result["timing_seconds"]
    losses = result["losses"]
    accuracy = result["accuracy"]
    router = result["router"]
    environment = result["environment"]
    return {
        "run_id": run_id,
        "architecture": identity["architecture"],
        "model_id": identity["model_id"],
        "dense_reference": identity["dense_reference"],
        "horizon_steps": int(identity["horizon_steps"]),
        "split_seed": int(result["seeds"]["split"]),
        "initialization_seed": int(result["seeds"]["initialization"]),
        "sampler_seed": int(result["seeds"]["sampler"]),
        "examples_seen": int(exposure["examples_seen"]),
        "input_tokens_15_seen": int(exposure["input_tokens_15_seen"]),
        "supervised_tokens_4_seen": int(exposure["supervised_tokens_4_seen"]),
        "training_pool_repetition_ratio": finite(
            exposure["training_pool_repetition_ratio"], "repetition ratio", run_id
        ),
        "stored_parameters": int(parameters["stored"]),
        "active_parameter_proxy": int(parameters["active_proxy"]),
        "estimated_training_flops": int(result["compute"]["estimated_training_flops"]),
        "compilation_seconds": finite(timing["compilation"], "compilation time", run_id),
        "training_step_seconds": finite(timing["training_steps"], "training step time", run_id),
        "validation_seconds": finite(timing["validation"], "validation time", run_id),
        "steady_state_training_seconds": finite(
            timing["steady_state_training"], "steady-state time", run_id
        ),
        "run_segment_wallclock_seconds": finite(
            timing["run_segment_wallclock"], "wallclock time", run_id
        ),
        "validation_answer_cross_entropy": finite(
            losses["validation_answer_cross_entropy"], "validation answer loss", run_id
        ),
        "validation_greedy_exact_match": finite(
            accuracy["validation_greedy_exact_match"], "validation exact match", run_id
        ),
        "final_training_answer_cross_entropy": finite(
            losses["final_training_answer_cross_entropy"], "training answer loss", run_id
        ),
        "final_load_balancing_loss": finite(
            losses["final_load_balancing_loss"], "balance loss", run_id
        ),
        "final_router_z_loss": finite(losses["final_router_z_loss"], "router z-loss", run_id),
        "final_weighted_load_balancing_loss": finite(
            losses["final_weighted_load_balancing_loss"], "weighted balance loss", run_id
        ),
        "final_weighted_router_z_loss": finite(
            losses["final_weighted_router_z_loss"], "weighted router z-loss", run_id
        ),
        "final_total_optimized_loss": finite(
            losses["final_total_optimized_loss"], "total optimized loss", run_id
        ),
        "num_experts": router["num_experts"],
        "top_k": router["top_k"],
        "expert_d_ff": router["expert_d_ff"],
        "protocol_fingerprint": result["fingerprints"]["protocol"],
        "config_fingerprint": result["fingerprints"]["config"],
        "split_fingerprint": result["fingerprints"]["split"],
        "backend": environment["backend"],
        "git_commit": environment["git_commit"],
        "uncertainty": None,
        "uncertainty_status": "unavailable_single_seed",
    }


def validate_collection(collection: dict[str, Any], architecture: str) -> list[dict[str, Any]]:
    if collection.get("schema_version") != "scaling-collection-v2":
        raise ValueError("unexpected collection schema")
    if collection.get("architecture") != architecture:
        raise ValueError("collection architecture mismatch")
    records = collection.get("records")
    if not isinstance(records, list) or len(records) != RUNS_PER_ARCHITECTURE:
        raise ValueError(
            f"collection must contain exactly {RUNS_PER_ARCHITECTURE} records"
        )
    rows: list[dict[str, Any]] = []
    coordinates: set[tuple[str, int]] = set()
    protocol_fingerprints: set[str] = set()
    split_fingerprints: set[str] = set()
    for record in records:
        if record.get("status") != "complete" or record.get("validation_errors"):
            raise ValueError(f"incomplete or invalid run: {record.get('run_id')}")
        result = record.get("result")
        if not isinstance(result, dict) or result.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise ValueError(f"missing normalized v2 result: {record.get('run_id')}")
        row = flatten_record(record)
        if row["architecture"] != architecture:
            raise ValueError(f"architecture mismatch: {row['run_id']}")
        if {row["split_seed"], row["initialization_seed"], row["sampler_seed"]} != {42}:
            raise ValueError(f"seed mismatch: {row['run_id']}")
        coordinates.add((str(row["model_id"]), int(row["horizon_steps"])))
        protocol_fingerprints.add(str(row["protocol_fingerprint"]))
        split_fingerprints.add(str(row["split_fingerprint"]))
        rows.append(row)
    if len(coordinates) != RUNS_PER_ARCHITECTURE or {
        horizon for _, horizon in coordinates
    } != set(HORIZONS):
        raise ValueError("collection does not cover four models by four horizons")
    if len(protocol_fingerprints) != 1:
        raise ValueError("runs do not share one protocol fingerprint")
    if len(split_fingerprints) != 1:
        raise ValueError("runs do not share one fixed split")
    rows.sort(key=lambda row: (row["active_parameter_proxy"], row["horizon_steps"]))
    return rows


def transition_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    models = sorted(
        {row["model_id"] for row in rows},
        key=lambda name: next(
            row["active_parameter_proxy"] for row in rows if row["model_id"] == name
        ),
    )
    output: list[dict[str, Any]] = []
    for model in models:
        candidates = sorted(
            (row for row in rows if row["model_id"] == model),
            key=lambda row: row["horizon_steps"],
        )
        item: dict[str, Any] = {
            "model_id": model,
            "stored_parameters": candidates[0]["stored_parameters"],
            "active_parameter_proxy": candidates[0]["active_parameter_proxy"],
        }
        for threshold in (0.5, 0.95, 0.99):
            matches = [
                row for row in candidates if row["validation_greedy_exact_match"] >= threshold
            ]
            item[f"first_horizon_at_{int(100 * threshold)}pct_em"] = (
                matches[0]["horizon_steps"] if matches else None
            )
        output.append(item)
    return output


def analyze(collection_path: Path, output_dir: Path, architecture: str) -> dict[str, Any]:
    rows = validate_collection(read_json(collection_path), architecture)
    transitions = transition_table(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "runs.csv", rows)
    write_csv(output_dir / "model_horizons.csv", rows)
    write_csv(output_dir / "transition_table.csv", transitions)
    summary = {
        "status": "verified",
        "architecture": architecture,
        "run_count": len(rows),
        "model_count": len({row["model_id"] for row in rows}),
        "horizons": list(HORIZONS),
        "seeds": {"split": 42, "initialization": 42, "sampler": 42},
        "uncertainty": None,
        "uncertainty_status": "unavailable_single_seed",
        "sample_standard_deviation_calculated": False,
        "protocol_fingerprint": rows[0]["protocol_fingerprint"],
        "split_fingerprint": rows[0]["split_fingerprint"],
        "transition_table": transitions,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("dense", "moe"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    expected_root = OUTPUT_ROOTS[args.architecture]
    if not path_has_suffix(output_root, expected_root):
        raise ValueError(f"output root must end with {expected_root!r}")
    summary = analyze(output_root / "results.json", output_root / "analysis", args.architecture)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
