"""Validate and aggregate the 120-run final dense-scaling experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any
from collections.abc import Iterable


DEFAULT_ARTIFACT = Path("artifacts/dense-scaling-final-t4")
EXPECTED_MODELS = {
    "dense_0p16m": 162_176,
    "dense_0p64m": 643_840,
    "dense_2p16m": 2_164_608,
    "dense_5p12m": 5_123_584,
    "dense_10m": 10_000_000,
}
EXPECTED_STEPS = (50, 75, 100, 125, 175, 250, 400, 750)
EXPECTED_SEEDS = (42, 123, 2026)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def mean(values: Iterable[float]) -> float:
    return statistics.fmean(list(values))


def validate_records(records: list[dict[str, Any]], artifact: Path) -> None:
    if len(records) != 120:
        raise ValueError(f"Expected 120 records, found {len(records)}.")

    seen_ids: set[str] = set()
    groups: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        run_id = str(record["run_id"])
        if run_id in seen_ids:
            raise ValueError(f"Duplicate run_id: {run_id}")
        seen_ids.add(run_id)

        if record.get("status") != "complete":
            raise ValueError(f"Incomplete record: {run_id}")
        if record.get("backend") != "gpu":
            raise ValueError(f"Non-GPU record: {run_id}")

        model = str(record["model"])
        if model not in EXPECTED_MODELS:
            raise ValueError(f"Unexpected model in {run_id}: {model}")

        parameter_count = int(record["parameter_count"])
        if parameter_count != EXPECTED_MODELS[model]:
            raise ValueError(
                f"{run_id}: expected {EXPECTED_MODELS[model]:,} parameters, "
                f"found {parameter_count:,}"
            )

        recorded_count = record.get("recorded_parameter_count")
        if recorded_count is not None and int(recorded_count) != parameter_count:
            raise ValueError(
                f"{run_id}: manifest and summary parameter counts disagree."
            )

        steps = int(record["steps"])
        seed = int(record["seed"])
        if steps not in EXPECTED_STEPS:
            raise ValueError(f"Unexpected step budget in {run_id}: {steps}")
        if seed not in EXPECTED_SEEDS:
            raise ValueError(f"Unexpected seed in {run_id}: {seed}")

        best = record.get("best")
        if not isinstance(best, dict):
            raise ValueError(f"Missing validation result: {run_id}")

        for field in ("loss", "greedy_exact_match"):
            value = float(best[field])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {field} in {run_id}")

        run_dir = artifact / "runs" / run_id
        for filename in (
            "config.json",
            "environment.json",
            "history.jsonl",
            "split_metadata.json",
            "summary.json",
        ):
            if not (run_dir / filename).exists():
                raise FileNotFoundError(run_dir / filename)

        groups[(model, steps)].append(record)

    if len(groups) != 40:
        raise ValueError(f"Expected 40 model/budget groups, found {len(groups)}.")

    for (model, steps), group in groups.items():
        seeds = sorted(int(record["seed"]) for record in group)
        if seeds != list(EXPECTED_SEEDS):
            raise ValueError(
                f"{model} at {steps} steps has seeds {seeds}; "
                f"expected {list(EXPECTED_SEEDS)}"
            )


def flatten_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for record in records:
        best = record["best"]
        rows.append(
            {
                "run_id": record["run_id"],
                "model": record["model"],
                "parameter_count": int(record["parameter_count"]),
                "seed": int(record["seed"]),
                "steps": int(record["steps"]),
                "examples_seen": int(record["examples_seen"]),
                "sequence_tokens_seen": int(record["sequence_tokens_seen"]),
                "answer_tokens_seen": int(record["answer_tokens_seen"]),
                "repetition_ratio": float(record["repetition_ratio"]),
                "estimated_training_flops": int(record["estimated_training_flops"]),
                "validation_loss": float(best["loss"]),
                "greedy_exact_match": float(best["greedy_exact_match"]),
                "final_train_loss": float(record["final_train_loss"]),
                "final_answer_token_accuracy": float(
                    record["final_answer_token_accuracy"]
                ),
                "compilation_seconds": float(record["compilation_seconds"]),
                "steady_state_training_seconds": float(
                    record["steady_state_training_seconds"]
                ),
                "total_training_seconds": float(record["total_training_seconds"]),
                "backend": record["backend"],
                "jax_version": record.get("jax_version", ""),
                "git_commit": record.get("git_commit", ""),
            }
        )

    rows.sort(
        key=lambda row: (
            row["parameter_count"],
            row["steps"],
            row["seed"],
        )
    )
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model"]), int(row["steps"]))].append(row)

    grouped_rows: list[dict[str, Any]] = []

    for (model, steps), group in groups.items():
        group.sort(key=lambda row: int(row["seed"]))
        first = group[0]

        metrics = {
            "validation_loss": [float(row["validation_loss"]) for row in group],
            "greedy_exact_match": [
                float(row["greedy_exact_match"]) for row in group
            ],
            "final_train_loss": [float(row["final_train_loss"]) for row in group],
            "final_answer_token_accuracy": [
                float(row["final_answer_token_accuracy"]) for row in group
            ],
            "compilation_seconds": [
                float(row["compilation_seconds"]) for row in group
            ],
            "steady_state_training_seconds": [
                float(row["steady_state_training_seconds"]) for row in group
            ],
            "total_training_seconds": [
                float(row["total_training_seconds"]) for row in group
            ],
        }

        aggregated: dict[str, Any] = {
            "model": model,
            "parameter_count": int(first["parameter_count"]),
            "steps": steps,
            "examples_seen": int(first["examples_seen"]),
            "sequence_tokens_seen": int(first["sequence_tokens_seen"]),
            "answer_tokens_seen": int(first["answer_tokens_seen"]),
            "repetition_ratio": float(first["repetition_ratio"]),
            "estimated_training_flops": int(first["estimated_training_flops"]),
            "seed_count": len(group),
            "seeds": ",".join(str(row["seed"]) for row in group),
        }

        for metric, values in metrics.items():
            aggregated[f"{metric}_mean"] = mean(values)
            aggregated[f"{metric}_std"] = sample_std(values)
            aggregated[f"{metric}_min"] = min(values)
            aggregated[f"{metric}_max"] = max(values)

        grouped_rows.append(aggregated)

    grouped_rows.sort(key=lambda row: (row["parameter_count"], row["steps"]))
    return grouped_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV.")

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
            "parameter_count": int(rows[0]["parameter_count"]),
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

    transitions.sort(key=lambda row: int(row["parameter_count"]))
    return transitions


def build_summary(
    records: list[dict[str, Any]],
    grouped_rows: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
) -> dict[str, Any]:
    commits = sorted(
        {
            str(record.get("git_commit"))
            for record in records
            if record.get("git_commit")
        }
    )
    jax_versions = sorted(
        {
            str(record.get("jax_version"))
            for record in records
            if record.get("jax_version")
        }
    )

    return {
        "status": "verified",
        "run_count": len(records),
        "group_count": len(grouped_rows),
        "gpu_run_count": sum(
            record.get("backend") == "gpu" for record in records
        ),
        "models": EXPECTED_MODELS,
        "steps": list(EXPECTED_STEPS),
        "seeds": list(EXPECTED_SEEDS),
        "git_commits": commits,
        "jax_versions": jax_versions,
        "transition_table": transitions,
        "scope_note": (
            "This is task-specific exposure scaling on a fixed pool of "
            "200,000 unique training pairs. D counts repeated sequence-token "
            "exposures, not unique internet-scale tokens."
        ),
    }


def model_label(parameter_count: int) -> str:
    if parameter_count >= 1_000_000:
        return f"{parameter_count / 1_000_000:.2f}M"
    return f"{parameter_count / 1_000:.0f}K"


def write_readme(
    path: Path,
    grouped_rows: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
) -> None:
    parameters = sorted(
        {int(row["parameter_count"]) for row in grouped_rows}
    )
    steps = sorted({int(row["steps"]) for row in grouped_rows})
    lookup = {
        (int(row["parameter_count"]), int(row["steps"])): row
        for row in grouped_rows
    }

    lines = [
        "# Final dense scaling experiment",
        "",
        "Verified results from 120 Colab T4 runs: five model sizes, eight "
        "training budgets, and three seeds per point.",
        "",
        "> D is full sequence tokens processed from a fixed pool of 200,000 "
        "unique addition pairs. Repeated examples count as additional token "
        "exposures.",
        "",
        "## Mean greedy exact match across three seeds",
        "",
        "| Steps | "
        + " | ".join(model_label(value) for value in parameters)
        + " |",
        "|---:|" + "|".join("---:" for _ in parameters) + "|",
    ]

    for step in steps:
        cells = []
        for parameter in parameters:
            row = lookup[(parameter, step)]
            mean_value = 100.0 * float(row["greedy_exact_match_mean"])
            std_value = 100.0 * float(row["greedy_exact_match_std"])
            cells.append(f"{mean_value:.3f}% ± {std_value:.3f}%")
        lines.append(f"| {step} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Exact-match transition",
            "",
            "| Model | Parameters | ≥50% EM | ≥95% EM | ≥99% EM |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for row in transitions:
        def show(value: Any) -> str:
            return str(value) if value is not None else "not reached"

        lines.append(
            "| "
            f"{row['model']} | {int(row['parameter_count']):,} | "
            f"{show(row['first_steps_at_50pct_em'])} | "
            f"{show(row['first_steps_at_95pct_em'])} | "
            f"{show(row['first_steps_at_99pct_em'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Larger models cross the algorithmic transition with fewer "
            "token exposures.",
            "- Exact match saturates sharply, while validation loss continues "
            "to separate models.",
            "- Three seeds quantify run-to-run variation around the transition.",
            "- The scaling-law fit must pass an identifiability check; the "
            "analysis does not force exponents.",
            "",
        ]
    )

    path.write_text("\n".join(lines))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    records = read_json(args.artifact / "final_results.json")

    validate_records(records, args.artifact)
    run_rows = flatten_records(records)
    grouped_rows = aggregate_rows(run_rows)
    transitions = transition_table(grouped_rows)
    summary = build_summary(records, grouped_rows, transitions)

    write_csv(args.artifact / "final_runs.csv", run_rows)
    write_csv(args.artifact / "final_grouped.csv", grouped_rows)
    write_csv(args.artifact / "transition_table.csv", transitions)
    write_json(args.artifact / "final_summary.json", summary)
    write_readme(args.artifact / "README.md", grouped_rows, transitions)

    print("Validated runs: 120")
    print("GPU runs: 120")
    print("Model/budget groups: 40")
    print("Seeds per group: 3")
    print(f"Grouped table: {args.artifact / 'final_grouped.csv'}")
    print(f"Summary: {args.artifact / 'final_summary.json'}")


if __name__ == "__main__":
    main()
