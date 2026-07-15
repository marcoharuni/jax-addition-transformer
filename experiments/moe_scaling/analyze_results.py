"""Validate and aggregate the 120-run matched-active MoE scaling experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


DEFAULT_RESULTS = Path("experiments/moe_scaling/results/final/final_results.json")
DEFAULT_ARTIFACT = Path("artifacts/moe-scaling-final-t4")
EXPECTED_STEPS = (50, 75, 100, 125, 175, 250, 400, 750)
EXPECTED_SEEDS = (42, 123, 2026)
EXPECTED_ENVIRONMENT = {
    "jax": "0.7.2",
    "flax": "0.12.0",
    "optax": "0.2.6",
    "orbax": "0.11.28",
    "numpy": "2.0.2",
}
EXPECTED_MODELS = {
    "moe_active_0p16m": (289_664, 162_688, "dense_0p16m", 162_176),
    "moe_active_0p64m": (1_152_768, 644_864, "dense_0p64m", 643_840),
    "moe_active_2p16m": (3_881_088, 2_166_912, "dense_2p16m", 2_164_608),
    "moe_active_5p12m": (9_190_912, 5_127_680, "dense_5p12m", 5_123_584),
    "moe_active_10m": (17_942_400, 10_006_400, "dense_10m", 10_000_000),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[float]) -> float:
    return statistics.fmean(list(values))


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def finite_float(value: Any, name: str, run_id: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {name} in {run_id}")
    return result


def routing_summary(metrics: Any) -> dict[str, float]:
    """Reduce final per-layer routing dictionaries to stable scalar diagnostics."""
    if not isinstance(metrics, list) or not metrics:
        return {
            "router_entropy_layer_mean": float("nan"),
            "expert_load_cv_layer_mean": float("nan"),
            "maximum_to_mean_load_ratio_layer_mean": float("nan"),
            "minimum_assignment_fraction": float("nan"),
            "maximum_assignment_fraction": float("nan"),
        }

    entropies: list[float] = []
    cvs: list[float] = []
    ratios: list[float] = []
    fractions: list[float] = []
    for layer in metrics:
        entropies.append(float(layer["router_entropy"]))
        cvs.append(float(layer["expert_load_coefficient_of_variation"]))
        ratios.append(float(layer["maximum_to_mean_load_ratio"]))
        fractions.extend(float(value) for value in layer["assignment_fraction_per_expert"])

    return {
        "router_entropy_layer_mean": mean(entropies),
        "expert_load_cv_layer_mean": mean(cvs),
        "maximum_to_mean_load_ratio_layer_mean": mean(ratios),
        "minimum_assignment_fraction": min(fractions),
        "maximum_assignment_fraction": max(fractions),
    }


def validate_records(records: list[dict[str, Any]]) -> None:
    if len(records) != 120:
        raise ValueError(f"expected 120 records, found {len(records)}")

    seen: set[str] = set()
    groups: defaultdict[tuple[str, int], list[int]] = defaultdict(list)
    for record in records:
        run_id = str(record["run_id"])
        if run_id in seen:
            raise ValueError(f"duplicate run_id: {run_id}")
        seen.add(run_id)
        if record.get("status") != "complete":
            raise ValueError(f"incomplete run: {run_id} ({record.get('status')})")
        if "summary" not in record:
            raise ValueError(f"missing summary: {run_id}")

        model = str(record["model"])
        if model not in EXPECTED_MODELS:
            raise ValueError(f"unexpected model in {run_id}: {model}")
        total, active, dense_name, dense_count = EXPECTED_MODELS[model]
        summary = record["summary"]
        if summary.get("architecture") != "moe":
            raise ValueError(f"non-MoE summary: {run_id}")
        if int(summary["total_parameter_count"]) != total:
            raise ValueError(f"total parameter mismatch: {run_id}")
        if int(summary["active_parameter_proxy"]) != active:
            raise ValueError(f"active parameter mismatch: {run_id}")
        if str(record["dense_reference"]) != dense_name:
            raise ValueError(f"dense reference mismatch: {run_id}")
        if int(record["dense_reference_parameters"]) != dense_count:
            raise ValueError(f"dense reference count mismatch: {run_id}")
        if int(summary["num_experts"]) != 4 or int(summary["top_k"]) != 2:
            raise ValueError(f"routing dimensions mismatch: {run_id}")
        if summary.get("backend") != "gpu":
            raise ValueError(f"non-GPU run: {run_id}")
        environment = record.get("environment")
        if not isinstance(environment, dict):
            raise ValueError(f"missing environment metadata: {run_id}")
        if environment.get("backend") != "gpu":
            raise ValueError(f"non-GPU environment: {run_id}")
        for package, expected_version in EXPECTED_ENVIRONMENT.items():
            if str(environment.get(package)) != expected_version:
                raise ValueError(
                    f"{run_id}: expected {package} {expected_version}, "
                    f"found {environment.get(package)!r}"
                )

        steps = int(record["steps"])
        seed = int(record["seed"])
        if steps not in EXPECTED_STEPS or seed not in EXPECTED_SEEDS:
            raise ValueError(f"unexpected grid coordinate: {run_id}")
        if int(summary["steps"]) < steps:
            raise ValueError(f"partial step count: {run_id}")
        best = summary.get("best")
        if not isinstance(best, dict):
            raise ValueError(f"missing validation result: {run_id}")
        finite_float(best["loss"], "validation loss", run_id)
        finite_float(best["greedy_exact_match"], "exact match", run_id)
        groups[(model, steps)].append(seed)

    if len(groups) != 40:
        raise ValueError(f"expected 40 model/budget groups, found {len(groups)}")
    for coordinate, seeds in groups.items():
        if sorted(seeds) != list(EXPECTED_SEEDS):
            raise ValueError(f"{coordinate} has seeds {sorted(seeds)}")


def flatten_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        run_id = str(record["run_id"])
        summary = record["summary"]
        best = summary["best"]
        routing = routing_summary(summary.get("routing_metrics"))
        environment = record["environment"]
        row: dict[str, Any] = {
            "run_id": run_id,
            "model": record["model"],
            "dense_reference": record["dense_reference"],
            "dense_reference_parameters": int(record["dense_reference_parameters"]),
            "total_parameter_count": int(summary["total_parameter_count"]),
            "active_parameter_proxy": int(summary["active_parameter_proxy"]),
            "active_overhead_parameters": int(record["active_overhead_parameters"]),
            "active_overhead_fraction": float(record["active_overhead_fraction"]),
            "num_experts": int(summary["num_experts"]),
            "top_k": int(summary["top_k"]),
            "expert_d_ff": int(summary["expert_d_ff"]),
            "seed": int(record["seed"]),
            "steps": int(record["steps"]),
            "examples_seen": int(summary["examples_seen"]),
            "sequence_tokens_seen": int(summary["sequence_tokens_seen"]),
            "answer_tokens_seen": int(summary["answer_tokens_seen"]),
            "repetition_ratio": float(summary["repetition_ratio"]),
            "estimated_training_flops": int(summary["estimated_training_flops"]),
            "validation_loss": finite_float(best["loss"], "validation_loss", run_id),
            "greedy_exact_match": finite_float(
                best["greedy_exact_match"], "greedy_exact_match", run_id
            ),
            "final_train_language_model_loss": finite_float(
                summary["final_train_language_model_loss"], "train loss", run_id
            ),
            "final_answer_token_accuracy": finite_float(
                summary["final_answer_token_accuracy"], "token accuracy", run_id
            ),
            "final_load_balancing_loss": finite_float(
                summary["final_load_balancing_loss"], "balance loss", run_id
            ),
            "final_router_z_loss": finite_float(
                summary["final_router_z_loss"], "z loss", run_id
            ),
            "final_total_optimized_loss": finite_float(
                summary["final_total_optimized_loss"], "optimized loss", run_id
            ),
            "compilation_seconds": finite_float(
                summary["compilation_seconds"], "compilation time", run_id
            ),
            "steady_state_training_seconds": finite_float(
                summary["steady_state_training_seconds"], "steady-state time", run_id
            ),
            "total_training_seconds": finite_float(
                summary["total_training_seconds"], "total time", run_id
            ),
            "validation_seconds": finite_float(
                summary["validation_seconds"], "validation time", run_id
            ),
            "backend": summary["backend"],
            "python_version": environment.get("python", ""),
            "jax_version": environment.get("jax", summary.get("jax_version", "")),
            "flax_version": environment.get("flax", ""),
            "optax_version": environment.get("optax", ""),
            "orbax_version": environment.get("orbax", ""),
            "numpy_version": environment.get("numpy", ""),
            "git_commit": summary.get("git_commit", environment.get("git_commit", "")),
            **routing,
        }
        rows.append(row)
    rows.sort(key=lambda row: (row["active_parameter_proxy"], row["steps"], row["seed"]))
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model"]), int(row["steps"]))].append(row)

    metrics = (
        "validation_loss",
        "greedy_exact_match",
        "final_train_language_model_loss",
        "final_answer_token_accuracy",
        "final_load_balancing_loss",
        "final_router_z_loss",
        "final_total_optimized_loss",
        "compilation_seconds",
        "steady_state_training_seconds",
        "total_training_seconds",
        "validation_seconds",
        "router_entropy_layer_mean",
        "expert_load_cv_layer_mean",
        "maximum_to_mean_load_ratio_layer_mean",
        "minimum_assignment_fraction",
        "maximum_assignment_fraction",
    )

    grouped_rows: list[dict[str, Any]] = []
    for (model, steps), group in groups.items():
        group.sort(key=lambda row: int(row["seed"]))
        first = group[0]
        aggregate: dict[str, Any] = {
            "model": model,
            "dense_reference": first["dense_reference"],
            "dense_reference_parameters": int(first["dense_reference_parameters"]),
            "total_parameter_count": int(first["total_parameter_count"]),
            "active_parameter_proxy": int(first["active_parameter_proxy"]),
            "active_overhead_parameters": int(first["active_overhead_parameters"]),
            "active_overhead_fraction": float(first["active_overhead_fraction"]),
            "num_experts": int(first["num_experts"]),
            "top_k": int(first["top_k"]),
            "expert_d_ff": int(first["expert_d_ff"]),
            "steps": steps,
            "examples_seen": int(first["examples_seen"]),
            "sequence_tokens_seen": int(first["sequence_tokens_seen"]),
            "answer_tokens_seen": int(first["answer_tokens_seen"]),
            "repetition_ratio": float(first["repetition_ratio"]),
            "estimated_training_flops": int(first["estimated_training_flops"]),
            "seed_count": len(group),
            "seeds": ",".join(str(row["seed"]) for row in group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            aggregate[f"{metric}_mean"] = mean(values)
            aggregate[f"{metric}_std"] = sample_std(values)
            aggregate[f"{metric}_min"] = min(values)
            aggregate[f"{metric}_max"] = max(values)
        grouped_rows.append(aggregate)

    grouped_rows.sort(key=lambda row: (row["active_parameter_proxy"], row["steps"]))
    return grouped_rows


def transition_table(
    grouped_rows: list[dict[str, Any]],
    thresholds: tuple[float, ...] = (0.5, 0.95, 0.99),
) -> list[dict[str, Any]]:
    by_model: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in grouped_rows:
        by_model[str(row["model"])].append(row)

    transitions: list[dict[str, Any]] = []
    for model, rows in by_model.items():
        rows.sort(key=lambda row: int(row["steps"]))
        result: dict[str, Any] = {
            "model": model,
            "dense_reference": rows[0]["dense_reference"],
            "dense_reference_parameters": int(rows[0]["dense_reference_parameters"]),
            "active_parameter_proxy": int(rows[0]["active_parameter_proxy"]),
            "total_parameter_count": int(rows[0]["total_parameter_count"]),
        }
        for threshold in thresholds:
            qualifying = [
                row
                for row in rows
                if float(row["greedy_exact_match_mean"]) >= threshold
            ]
            result[f"first_steps_at_{int(threshold * 100)}pct_em"] = (
                int(qualifying[0]["steps"]) if qualifying else None
            )
        transitions.append(result)
    transitions.sort(key=lambda row: int(row["active_parameter_proxy"]))
    return transitions


def analyze(results_path: Path, artifact: Path) -> dict[str, Any]:
    records = read_json(results_path)
    validate_records(records)
    rows = flatten_records(records)
    grouped = aggregate_rows(rows)
    transitions = transition_table(grouped)

    artifact.mkdir(parents=True, exist_ok=True)
    write_csv(artifact / "final_runs.csv", rows)
    write_csv(artifact / "final_grouped.csv", grouped)
    write_csv(artifact / "transition_table.csv", transitions)
    write_json(artifact / "final_results.json", records)

    summary = {
        "status": "verified",
        "architecture": "top-2 ragged-dot MoE",
        "run_count": len(rows),
        "group_count": len(grouped),
        "gpu_run_count": sum(row["backend"] == "gpu" for row in rows),
        "models": sorted({row["model"] for row in rows}),
        "step_budgets": sorted({row["steps"] for row in rows}),
        "seeds": sorted({row["seed"] for row in rows}),
        "python_versions": sorted({row["python_version"] for row in rows}),
        "jax_versions": sorted({row["jax_version"] for row in rows}),
        "flax_versions": sorted({row["flax_version"] for row in rows}),
        "optax_versions": sorted({row["optax_version"] for row in rows}),
        "orbax_versions": sorted({row["orbax_version"] for row in rows}),
        "numpy_versions": sorted({row["numpy_version"] for row in rows}),
        "git_commits": sorted({row["git_commit"] for row in rows if row["git_commit"]}),
        "transition_table": transitions,
        "parameter_accounting": (
            "Active proxy includes all attention, normalization, embeddings, routers, "
            "and exactly top_k expert parameter sets. Estimated FLOPs omit routing and "
            "dispatch overhead; measured wall-clock time is reported separately."
        ),
    }
    write_json(artifact / "final_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    summary = analyze(args.results, args.artifact)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
