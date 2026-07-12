"""Validate, tabulate, and summarize the T4 dense-scaling pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from fit_scaling_law import fit_records


DEFAULT_ARTIFACT = Path("artifacts/dense-scaling-pilot-t4")


def _best(record: dict[str, Any]) -> dict[str, Any]:
    best = record.get("best")
    if not isinstance(best, dict):
        raise ValueError(f"Run has no validation result: {record['run_id']}")
    return best


def validate_records(records: list[dict[str, Any]]) -> None:
    if len(records) != 16:
        raise ValueError(f"Expected 16 pilot runs, found {len(records)}.")

    incomplete = [
        record["run_id"]
        for record in records
        if record.get("status") != "complete"
    ]
    if incomplete:
        raise ValueError(f"Incomplete runs: {incomplete}")

    non_gpu = [
        record["run_id"]
        for record in records
        if record.get("backend") != "gpu"
    ]
    if non_gpu:
        raise ValueError(f"Runs not recorded on a GPU: {non_gpu}")

    fingerprints = {
        record.get("git_commit")
        for record in records
        if record.get("git_commit")
    }
    if len(fingerprints) != 1:
        raise ValueError(f"Expected one source commit, found: {fingerprints}")


def table_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for record in records:
        best = _best(record)
        parameters = int(record["parameter_count"])
        sequence_tokens = int(record["sequence_tokens_seen"])

        rows.append(
            {
                "run_id": record["run_id"],
                "parameter_count": parameters,
                "steps": int(record["steps"]),
                "examples_seen": int(record["examples_seen"]),
                "sequence_tokens_seen": sequence_tokens,
                "answer_tokens_seen": int(record["answer_tokens_seen"]),
                "repetition_ratio": float(record["repetition_ratio"]),
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
                "total_training_seconds": float(
                    record["total_training_seconds"]
                ),
                "estimated_training_flops": 6 * parameters * sequence_tokens,
                "backend": record["backend"],
                "seed": int(record["seed"]),
                "git_commit": record.get("git_commit", ""),
            }
        )

    rows.sort(key=lambda row: (row["parameter_count"], row["steps"]))
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_model(parameters: int) -> str:
    if parameters >= 1_000_000:
        return f"{parameters / 1_000_000:.2f}M"
    return f"{parameters / 1_000:.0f}K"


def write_summary(
    rows: list[dict[str, Any]],
    fit: dict[str, Any],
    path: Path,
) -> None:
    parameters = sorted({int(row["parameter_count"]) for row in rows})
    steps = sorted({int(row["steps"]) for row in rows})
    lookup = {
        (int(row["parameter_count"]), int(row["steps"])): row
        for row in rows
    }

    lines = [
        "# Dense scaling pilot on a Colab T4",
        "",
        "This artifact summarizes 16 completed GPU runs from one random seed.",
        "It is a pilot used to locate the task's transition and saturation regions.",
        "",
        "## Greedy exact match",
        "",
        "| Steps | "
        + " | ".join(_format_model(value) for value in parameters)
        + " |",
        "|---:|" + "|".join("---:" for _ in parameters) + "|",
    ]

    for step in steps:
        values = [
            lookup[(parameter, step)]["greedy_exact_match"]
            for parameter in parameters
        ]
        lines.append(
            f"| {step} | "
            + " | ".join(f"{100.0 * value:.3f}%" for value in values)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Validation loss",
            "",
            "| Steps | "
            + " | ".join(_format_model(value) for value in parameters)
            + " |",
            "|---:|" + "|".join("---:" for _ in parameters) + "|",
        ]
    )

    for step in steps:
        values = [
            lookup[(parameter, step)]["validation_loss"]
            for parameter in parameters
        ]
        lines.append(
            f"| {step} | "
            + " | ".join(f"{value:.8f}" for value in values)
            + " |"
        )

    additive = fit["additive_chinchilla_candidate"]
    multiplicative = fit["multiplicative_diagnostic"]

    lines.extend(
        [
            "",
            "## Scaling-law identifiability check",
            "",
            f"- status: `{fit['status']}`",
            f"- additive candidate alpha: {additive['alpha']:.6f}",
            f"- additive candidate beta: {additive['beta']:.6f}",
            f"- additive log10 R-squared: {additive['log10_r_squared']:.6f}",
            (
                "- additive floor collapsed to zero: "
                f"{additive['floor_collapsed_to_zero']}"
            ),
            (
                "- multiplicative diagnostic alpha: "
                f"{multiplicative['alpha']:.6f}"
            ),
            (
                "- multiplicative diagnostic beta: "
                f"{multiplicative['beta']:.6f}"
            ),
            "",
            "The pilot does not support a defensible compute-optimal exponent. "
            "It has one seed, four model sizes, budget-specific learning-rate "
            "schedules, a sharp phase transition, and many saturated observations.",
            "",
            "## Main observations",
            "",
            "- All four models are undertrained at 50 steps.",
            "- Larger models cross the exact-match transition with fewer examples.",
            "- Exact match saturates before validation loss, so loss is the primary fit target.",
            "- The 5.12M and 10M models show diminishing returns after the task is solved.",
            "",
        ]
    )

    path.write_text("\n".join(lines))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--min-loss", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    results_path = args.artifact / "pilot_results.json"
    records = json.loads(results_path.read_text())

    validate_records(records)
    rows = table_rows(records)
    fit = fit_records(records, min_loss=args.min_loss)

    table_path = args.artifact / "pilot_table.csv"
    fit_path = args.artifact / "pilot_fit.json"
    summary_path = args.artifact / "README.md"

    write_csv(rows, table_path)
    fit_path.write_text(json.dumps(fit, indent=2) + "\n")
    write_summary(rows, fit, summary_path)

    print("Validated runs: 16")
    print("GPU runs: 16")
    print(f"Table: {table_path}")
    print(f"Fit: {fit_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
