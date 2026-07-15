"""Pair normalized dense and MoE results and build joint frontiers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .analyze import validate_collection
from .protocol import OUTPUT_ROOTS, RUNS_PER_ARCHITECTURE, path_has_suffix


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


def pair_rows(dense: list[dict[str, Any]], moe: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dense_index = {
        (row["model_id"], row["horizon_steps"], row["initialization_seed"]): row for row in dense
    }
    pairs: list[dict[str, Any]] = []
    for moe_row in moe:
        key = (
            moe_row["dense_reference"],
            moe_row["horizon_steps"],
            moe_row["initialization_seed"],
        )
        if key not in dense_index:
            raise ValueError(f"missing dense pair: {key}")
        dense_row = dense_index[key]
        if dense_row["protocol_fingerprint"] != moe_row["protocol_fingerprint"]:
            raise ValueError(f"protocol mismatch: {key}")
        if dense_row["split_fingerprint"] != moe_row["split_fingerprint"]:
            raise ValueError(f"split mismatch: {key}")
        pairs.append(
            {
                "dense_model": dense_row["model_id"],
                "moe_model": moe_row["model_id"],
                "horizon_steps": moe_row["horizon_steps"],
                "seed": moe_row["initialization_seed"],
                "dense_stored_parameters": dense_row["stored_parameters"],
                "dense_active_parameter_proxy": dense_row["active_parameter_proxy"],
                "moe_stored_parameters": moe_row["stored_parameters"],
                "moe_active_parameter_proxy": moe_row["active_parameter_proxy"],
                "active_parameter_proxy_ratio": (
                    moe_row["active_parameter_proxy"] / dense_row["active_parameter_proxy"]
                ),
                "examples_seen": dense_row["examples_seen"],
                "input_tokens_15_seen": dense_row["input_tokens_15_seen"],
                "supervised_tokens_4_seen": dense_row["supervised_tokens_4_seen"],
                "dense_validation_answer_cross_entropy": dense_row[
                    "validation_answer_cross_entropy"
                ],
                "moe_validation_answer_cross_entropy": moe_row["validation_answer_cross_entropy"],
                "moe_over_dense_answer_loss_ratio": (
                    moe_row["validation_answer_cross_entropy"]
                    / dense_row["validation_answer_cross_entropy"]
                ),
                "dense_validation_greedy_exact_match": dense_row["validation_greedy_exact_match"],
                "moe_validation_greedy_exact_match": moe_row["validation_greedy_exact_match"],
                "moe_minus_dense_exact_match": (
                    moe_row["validation_greedy_exact_match"]
                    - dense_row["validation_greedy_exact_match"]
                ),
                "dense_estimated_training_flops": dense_row["estimated_training_flops"],
                "moe_estimated_training_flops": moe_row["estimated_training_flops"],
                "moe_over_dense_estimated_flops_ratio": (
                    moe_row["estimated_training_flops"] / dense_row["estimated_training_flops"]
                ),
                "dense_training_step_seconds": dense_row["training_step_seconds"],
                "moe_training_step_seconds": moe_row["training_step_seconds"],
                "moe_over_dense_training_time_ratio": (
                    moe_row["training_step_seconds"] / dense_row["training_step_seconds"]
                ),
                "uncertainty": None,
                "uncertainty_status": "unavailable_single_seed",
            }
        )
    if len(pairs) != RUNS_PER_ARCHITECTURE:
        raise ValueError(f"expected {RUNS_PER_ARCHITECTURE} paired runs, found {len(pairs)}")
    pairs.sort(key=lambda row: (row["dense_stored_parameters"], row["horizon_steps"]))
    return pairs


def frontier(rows: list[dict[str, Any]], cost: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        candidates.append(
            {
                "architecture": row["architecture"],
                "model_id": row["model_id"],
                "horizon_steps": row["horizon_steps"],
                "stored_parameters": row["stored_parameters"],
                "active_parameter_proxy": row["active_parameter_proxy"],
                "cost": row[cost],
                "validation_answer_cross_entropy": row["validation_answer_cross_entropy"],
                "validation_greedy_exact_match": row["validation_greedy_exact_match"],
                "uncertainty": None,
                "uncertainty_status": "unavailable_single_seed",
            }
        )
    candidates.sort(key=lambda row: (row["cost"], row["validation_answer_cross_entropy"]))
    output: list[dict[str, Any]] = []
    best = float("inf")
    for row in candidates:
        if row["validation_answer_cross_entropy"] < best:
            output.append(row)
            best = row["validation_answer_cross_entropy"]
    return output


def plot_frontier(rows: list[dict[str, Any]], output: Path, xlabel: str) -> None:
    figure, axis = plt.subplots(figsize=(8.6, 5.3))
    for architecture in ("dense", "moe"):
        selected = [row for row in rows if row["architecture"] == architecture]
        axis.plot(
            [row["cost"] for row in selected],
            [row["validation_answer_cross_entropy"] for row in selected],
            marker="o",
            label=architecture,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Validation answer cross-entropy")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg")
    plt.close(figure)


def compare(dense_results: Path, moe_results: Path, output_root: Path) -> dict[str, Any]:
    dense = validate_collection(read_json(dense_results), "dense")
    moe = validate_collection(read_json(moe_results), "moe")
    root_identity = {
        "protocol_fingerprint": dense[0]["protocol_fingerprint"],
        "dense_split_fingerprint": dense[0]["split_fingerprint"],
        "moe_split_fingerprint": moe[0]["split_fingerprint"],
        "output_root": OUTPUT_ROOTS["comparison"],
    }
    identity_path = output_root / "root_identity.json"
    if identity_path.exists():
        if read_json(identity_path) != root_identity:
            raise ValueError("refusing incompatible existing comparison root")
    elif output_root.exists() and any(output_root.iterdir()):
        raise ValueError("refusing non-empty comparison root without study identity")
    pairs = pair_rows(dense, moe)
    combined = dense + moe
    compute_frontier = frontier(combined, "estimated_training_flops")
    wallclock_frontier = frontier(combined, "training_step_seconds")
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(identity_path, root_identity)
    write_csv(output_root / "paired_runs.csv", pairs)
    write_csv(output_root / "combined_compute_frontier.csv", compute_frontier)
    write_csv(output_root / "combined_wallclock_frontier.csv", wallclock_frontier)
    plot_frontier(
        compute_frontier,
        output_root / "figures" / "compute_frontier.svg",
        "Estimated training FLOPs",
    )
    plot_frontier(
        wallclock_frontier,
        output_root / "figures" / "wallclock_frontier.svg",
        "Measured training-step seconds",
    )
    summary = {
        "status": "verified",
        "paired_run_count": len(pairs),
        "dense_run_count": len(dense),
        "moe_run_count": len(moe),
        "fixed_seed": 42,
        "uncertainty": None,
        "uncertainty_status": "unavailable_single_seed",
        "sample_standard_deviation_calculated": False,
        "protocol_fingerprint": dense[0]["protocol_fingerprint"],
        "split_fingerprint": dense[0]["split_fingerprint"],
        "flop_caveat": (
            "6*N_active*D excludes routing, sorting, dispatch, scatter-add, "
            "rematerialization recomputation, and hardware efficiency."
        ),
    }
    write_json(output_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-root", type=Path, required=True)
    parser.add_argument("--moe-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    dense_root = args.dense_root.expanduser().resolve()
    moe_root = args.moe_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not path_has_suffix(dense_root, OUTPUT_ROOTS["dense"]):
        raise ValueError("dense root has the wrong path suffix")
    if not path_has_suffix(moe_root, OUTPUT_ROOTS["moe"]):
        raise ValueError("MoE root has the wrong path suffix")
    if not path_has_suffix(output_root, OUTPUT_ROOTS["comparison"]):
        raise ValueError("comparison root has the wrong path suffix")
    summary = compare(dense_root / "results.json", moe_root / "results.json", output_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
