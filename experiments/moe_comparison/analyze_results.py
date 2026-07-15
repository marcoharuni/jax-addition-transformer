"""Validate the three matched MoE runs and create one three-seed aggregate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(results_path: Path, output_dir: Path) -> None:
    records = read_json(results_path)
    incomplete = [row["run_id"] for row in records if row["status"] != "complete"]
    if incomplete:
        raise ValueError(
            f"cannot aggregate: {len(incomplete)} missing, partial, or failed runs; "
            f"first entries: {incomplete[:10]}"
        )
    if len(records) != 3:
        raise ValueError(f"expected 3 records, found {len(records)}")
    rows = []
    for record in records:
        summary = record["summary"]
        best = summary.get("best")
        if not isinstance(best, dict):
            raise ValueError(f"missing validation result for {record['run_id']}")
        row = {
            "run_id": record["run_id"],
            "model": record["model"],
            "seed": record["seed"],
            "steps": record["steps"],
            "total_parameter_count": summary["total_parameter_count"],
            "active_parameter_proxy": summary["active_parameter_proxy"],
            "sequence_tokens_seen": summary["sequence_tokens_seen"],
            "estimated_training_flops": summary["estimated_training_flops"],
            "validation_loss": best["loss"],
            "greedy_exact_match": best["greedy_exact_match"],
            "final_train_language_model_loss": summary["final_train_language_model_loss"],
            "final_load_balancing_loss": summary["final_load_balancing_loss"],
            "final_router_z_loss": summary["final_router_z_loss"],
            "final_total_optimized_loss": summary["final_total_optimized_loss"],
            "compilation_seconds": summary["compilation_seconds"],
            "steady_state_training_seconds": summary["steady_state_training_seconds"],
            "total_training_seconds": summary["total_training_seconds"],
            "backend": summary["backend"],
            "jax_version": summary["jax_version"],
            "git_commit": summary["git_commit"],
        }
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"non-finite {key} in {record['run_id']}")
        rows.append(row)
    rows.sort(key=lambda row: (row["total_parameter_count"], row["steps"], row["seed"]))
    grouped: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["steps"])].append(row)
    aggregates = []
    metrics = (
        "validation_loss",
        "greedy_exact_match",
        "final_train_language_model_loss",
        "final_load_balancing_loss",
        "final_router_z_loss",
        "final_total_optimized_loss",
        "total_training_seconds",
    )
    for (model, steps), group in grouped.items():
        if sorted(row["seed"] for row in group) != [42, 123, 2026]:
            raise ValueError(f"{model} at {steps} steps does not contain all three seeds")
        aggregate = {"model": model, "steps": steps, "seed_count": 3}
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            aggregate[f"{metric}_mean"] = statistics.fmean(values)
            aggregate[f"{metric}_sample_std"] = statistics.stdev(values)
        aggregates.append(aggregate)
    if len(aggregates) != 1:
        raise ValueError(f"expected 1 aggregate row, found {len(aggregates)}")
    aggregates.sort(key=lambda row: (row["model"], row["steps"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "moe_comparison_runs.csv", rows)
    write_csv(output_dir / "moe_comparison_grouped.csv", aggregates)
    (output_dir / "moe_comparison_summary.json").write_text(
        json.dumps(
            {
                "status": "verified",
                "run_count": len(rows),
                "group_count": len(aggregates),
                "models": sorted({row["model"] for row in rows}),
                "steps": sorted({row["steps"] for row in rows}),
                "seeds": sorted({row["seed"] for row in rows}),
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.results, args.output_dir)


if __name__ == "__main__":
    main()
