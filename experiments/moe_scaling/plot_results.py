"""Generate final matched-active MoE scaling figures from three-seed aggregates."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_GROUPED = Path(
    "artifacts/moe-scaling-final-t4/final_grouped.csv"
)
DEFAULT_FRONTIER = Path(
    "artifacts/moe-scaling-final-t4/empirical_compute_frontier.csv"
)
DEFAULT_ASSETS = Path("assets")


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def model_label(parameters: int) -> str:
    if parameters >= 1_000_000:
        return f"{parameters / 1_000_000:.2f}M active"
    return f"{parameters / 1_000:.0f}K active"


def group_models(
    rows: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    grouped: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(float(row["active_parameter_proxy"]))].append(row)

    for values in grouped.values():
        values.sort(key=lambda row: int(float(row["sequence_tokens_seen"])))

    return dict(sorted(grouped.items()))


def plot_loss(
    grouped: dict[int, list[dict[str, Any]]],
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.2))

    for parameters, rows in grouped.items():
        x = np.asarray(
            [float(row["sequence_tokens_seen"]) for row in rows]
        )
        mean = np.asarray(
            [float(row["validation_loss_mean"]) for row in rows]
        )
        std = np.asarray(
            [float(row["validation_loss_std"]) for row in rows]
        )
        lower = np.maximum(mean - std, np.finfo(float).tiny)
        upper = mean + std

        line = axis.plot(
            x,
            mean,
            marker="o",
            label=model_label(parameters),
        )[0]
        axis.fill_between(
            x,
            lower,
            upper,
            alpha=0.18,
            color=line.get_color(),
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Full sequence tokens processed")
    axis.set_ylabel("Validation answer-token cross-entropy")
    axis.set_title("Final MoE scaling: mean validation loss ± 1 SD")
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

    for parameters, rows in grouped.items():
        x = np.asarray(
            [float(row["sequence_tokens_seen"]) for row in rows]
        )
        mean = np.asarray(
            [float(row["greedy_exact_match_mean"]) for row in rows]
        )
        std = np.asarray(
            [float(row["greedy_exact_match_std"]) for row in rows]
        )

        axis.errorbar(
            x,
            mean,
            yerr=std,
            marker="o",
            capsize=3,
            label=model_label(parameters),
        )

    axis.set_xscale("log")
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("Full sequence tokens processed")
    axis.set_ylabel("Validation greedy exact match")
    axis.set_title("Final MoE scaling: exact-match transition")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, format="svg")
    plt.close(figure)


def plot_compute(
    grouped: dict[int, list[dict[str, Any]]],
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.2))

    for parameters, rows in grouped.items():
        x = np.asarray(
            [float(row["estimated_training_flops"]) for row in rows]
        )
        mean = np.asarray(
            [float(row["validation_loss_mean"]) for row in rows]
        )
        std = np.asarray(
            [float(row["validation_loss_std"]) for row in rows]
        )

        axis.errorbar(
            x,
            mean,
            yerr=std,
            marker="o",
            capsize=3,
            label=model_label(parameters),
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Approximate training FLOPs, 6 × active N × sequence tokens")
    axis.set_ylabel("Validation answer-token cross-entropy")
    axis.set_title("Final MoE scaling: loss versus compute")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, format="svg")
    plt.close(figure)


def plot_frontier(
    rows: list[dict[str, Any]],
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.2))

    x = np.asarray(
        [float(row["estimated_training_flops"]) for row in rows]
    )
    y = np.asarray(
        [float(row["validation_loss_mean"]) for row in rows]
    )
    yerr = np.asarray(
        [float(row["validation_loss_std"]) for row in rows]
    )

    axis.errorbar(x, y, yerr=yerr, marker="o", capsize=3)

    for row in rows:
        axis.annotate(
            f"{row['model']}, {int(float(row['steps']))} steps",
            (
                float(row["estimated_training_flops"]),
                float(row["validation_loss_mean"]),
            ),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Approximate training FLOPs")
    axis.set_ylabel("Best measured mean validation loss")
    axis.set_title("Empirical MoE compute-loss Pareto frontier")
    axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, format="svg")
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grouped", type=Path, default=DEFAULT_GROUPED)
    parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    grouped_rows = read_csv(args.grouped)
    frontier_rows = read_csv(args.frontier)
    grouped = group_models(grouped_rows)

    args.assets.mkdir(parents=True, exist_ok=True)

    outputs = (
        args.assets / "moe_scaling_final_loss.svg",
        args.assets / "moe_scaling_final_exact_match.svg",
        args.assets / "moe_scaling_final_compute.svg",
        args.assets / "moe_scaling_final_frontier.svg",
    )

    plot_loss(grouped, outputs[0])
    plot_exact_match(grouped, outputs[1])
    plot_compute(grouped, outputs[2])
    plot_frontier(frontier_rows, outputs[3])

    for output in outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
