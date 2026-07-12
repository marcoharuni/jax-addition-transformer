"""Create publication-ready SVG plots from the dense-scaling pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DEFAULT_RESULTS = Path("artifacts/dense-scaling-pilot-t4/pilot_results.json")
DEFAULT_ASSETS = Path("assets")


def _label(parameters: int) -> str:
    if parameters >= 1_000_000:
        return f"{parameters / 1_000_000:.2f}M parameters"
    return f"{parameters / 1_000:.0f}K parameters"


def _group(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}

    for record in records:
        if record.get("status") != "complete":
            continue
        parameter_count = int(record["parameter_count"])
        grouped.setdefault(parameter_count, []).append(record)

    for values in grouped.values():
        values.sort(key=lambda record: int(record["sequence_tokens_seen"]))

    return dict(sorted(grouped.items()))


def _best(record: dict[str, Any]) -> dict[str, Any]:
    best = record.get("best")
    if not isinstance(best, dict):
        raise ValueError(f"Missing validation result for {record['run_id']}")
    return best


def plot_loss(
    grouped: dict[int, list[dict[str, Any]]],
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.2))

    for parameters, records in grouped.items():
        axis.plot(
            [record["sequence_tokens_seen"] for record in records],
            [_best(record)["loss"] for record in records],
            marker="o",
            label=_label(parameters),
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Full sequence tokens processed")
    axis.set_ylabel("Validation answer-token cross-entropy")
    axis.set_title("Dense arithmetic scaling pilot: validation loss")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, format="svg")
    plt.close(figure)


def plot_exact_match(
    grouped: dict[int, list[dict[str, Any]]],
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.2))

    for parameters, records in grouped.items():
        axis.plot(
            [record["sequence_tokens_seen"] for record in records],
            [_best(record)["greedy_exact_match"] for record in records],
            marker="o",
            label=_label(parameters),
        )

    axis.set_xscale("log")
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("Full sequence tokens processed")
    axis.set_ylabel("Validation greedy exact match")
    axis.set_title("Dense arithmetic scaling pilot: exact-match transition")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, format="svg")
    plt.close(figure)


def plot_compute(
    records: list[dict[str, Any]],
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.2))

    for record in sorted(
        records,
        key=lambda item: (
            int(item["parameter_count"]),
            int(item["sequence_tokens_seen"]),
        ),
    ):
        parameters = int(record["parameter_count"])
        tokens = int(record["sequence_tokens_seen"])
        compute = 6 * parameters * tokens
        axis.scatter(
            compute,
            _best(record)["loss"],
            marker="o",
        )
        axis.annotate(
            f"{_label(parameters).split()[0]}, {record['steps']} steps",
            (compute, _best(record)["loss"]),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Approximate training FLOPs, 6 × N × sequence tokens")
    axis.set_ylabel("Validation answer-token cross-entropy")
    axis.set_title("Dense arithmetic scaling pilot: loss versus compute")
    axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, format="svg")
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    records = json.loads(args.results.read_text())
    complete = [
        record
        for record in records
        if record.get("status") == "complete"
    ]

    args.assets.mkdir(parents=True, exist_ok=True)
    grouped = _group(complete)

    loss_path = args.assets / "dense_scaling_pilot_loss.svg"
    exact_match_path = args.assets / "dense_scaling_pilot_exact_match.svg"
    compute_path = args.assets / "dense_scaling_pilot_compute.svg"

    plot_loss(grouped, loss_path)
    plot_exact_match(grouped, exact_match_path)
    plot_compute(complete, compute_path)

    print(f"Saved: {loss_path}")
    print(f"Saved: {exact_match_path}")
    print(f"Saved: {compute_path}")


if __name__ == "__main__":
    main()
