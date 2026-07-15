"""Create paired dense-versus-MoE tables and combined empirical frontiers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DEFAULT_DENSE = Path("artifacts/dense-scaling-final-t4")
DEFAULT_MOE = Path("artifacts/moe-scaling-final-t4")
DEFAULT_OUTPUT = Path("artifacts/dense-vs-moe-final-t4")
DEFAULT_ASSETS = Path("assets")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def f(row: dict[str, Any], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name}")
    return value


def paired_runs(dense_rows: list[dict[str, str]], moe_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    dense_index = {
        (row["model"], int(row["steps"]), int(row["seed"])): row
        for row in dense_rows
    }
    pairs: list[dict[str, Any]] = []
    for moe in moe_rows:
        key = (moe["dense_reference"], int(moe["steps"]), int(moe["seed"]))
        if key not in dense_index:
            raise ValueError(f"missing dense match for {key}")
        dense = dense_index[key]
        dense_loss = f(dense, "validation_loss")
        moe_loss = f(moe, "validation_loss")
        dense_time = f(dense, "total_training_seconds")
        moe_time = f(moe, "total_training_seconds")
        pairs.append(
            {
                "dense_model": dense["model"],
                "moe_model": moe["model"],
                "seed": int(moe["seed"]),
                "steps": int(moe["steps"]),
                "dense_parameters": int(float(dense["parameter_count"])),
                "moe_active_parameter_proxy": int(float(moe["active_parameter_proxy"])),
                "moe_total_parameter_count": int(float(moe["total_parameter_count"])),
                "dense_validation_loss": dense_loss,
                "moe_validation_loss": moe_loss,
                "moe_over_dense_loss_ratio": moe_loss / dense_loss,
                "dense_exact_match": f(dense, "greedy_exact_match"),
                "moe_exact_match": f(moe, "greedy_exact_match"),
                "moe_minus_dense_exact_match": (
                    f(moe, "greedy_exact_match") - f(dense, "greedy_exact_match")
                ),
                "dense_total_training_seconds": dense_time,
                "moe_total_training_seconds": moe_time,
                "moe_over_dense_time_ratio": moe_time / dense_time,
                "dense_estimated_training_flops": int(float(dense["estimated_training_flops"])),
                "moe_estimated_training_flops": int(float(moe["estimated_training_flops"])),
                "moe_over_dense_estimated_flops_ratio": (
                    float(moe["estimated_training_flops"])
                    / float(dense["estimated_training_flops"])
                ),
                "moe_router_entropy": f(moe, "router_entropy_layer_mean"),
                "moe_expert_load_cv": f(moe, "expert_load_cv_layer_mean"),
            }
        )
    if len(pairs) != 120:
        raise ValueError(f"expected 120 paired runs, found {len(pairs)}")
    pairs.sort(key=lambda row: (row["dense_parameters"], row["steps"], row["seed"]))
    return pairs


def aggregate_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        groups[(str(row["dense_model"]), int(row["steps"]))].append(row)
    metrics = (
        "dense_validation_loss",
        "moe_validation_loss",
        "moe_over_dense_loss_ratio",
        "dense_exact_match",
        "moe_exact_match",
        "moe_minus_dense_exact_match",
        "dense_total_training_seconds",
        "moe_total_training_seconds",
        "moe_over_dense_time_ratio",
        "moe_over_dense_estimated_flops_ratio",
        "moe_router_entropy",
        "moe_expert_load_cv",
    )
    result: list[dict[str, Any]] = []
    for (dense_model, steps), group in groups.items():
        group.sort(key=lambda row: int(row["seed"]))
        first = group[0]
        row: dict[str, Any] = {
            "dense_model": dense_model,
            "moe_model": first["moe_model"],
            "dense_parameters": first["dense_parameters"],
            "moe_active_parameter_proxy": first["moe_active_parameter_proxy"],
            "moe_total_parameter_count": first["moe_total_parameter_count"],
            "steps": steps,
            "seed_count": len(group),
        }
        for metric in metrics:
            values = [float(item[metric]) for item in group]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_std"] = statistics.stdev(values)
        result.append(row)
    if len(result) != 40:
        raise ValueError(f"expected 40 paired aggregates, found {len(result)}")
    result.sort(key=lambda row: (row["dense_parameters"], row["steps"]))
    return result


def combined_frontier(
    dense_grouped: list[dict[str, str]], moe_grouped: list[dict[str, str]], cost_field: str
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in dense_grouped:
        candidates.append(
            {
                "architecture": "dense",
                "model": row["model"],
                "active_parameters": int(float(row["parameter_count"])),
                "stored_parameters": int(float(row["parameter_count"])),
                "steps": int(float(row["steps"])),
                "cost": f(row, cost_field),
                "validation_loss_mean": f(row, "validation_loss_mean"),
                "validation_loss_std": f(row, "validation_loss_std"),
                "greedy_exact_match_mean": f(row, "greedy_exact_match_mean"),
            }
        )
    for row in moe_grouped:
        candidates.append(
            {
                "architecture": "moe",
                "model": row["model"],
                "active_parameters": int(float(row["active_parameter_proxy"])),
                "stored_parameters": int(float(row["total_parameter_count"])),
                "steps": int(float(row["steps"])),
                "cost": f(row, cost_field),
                "validation_loss_mean": f(row, "validation_loss_mean"),
                "validation_loss_std": f(row, "validation_loss_std"),
                "greedy_exact_match_mean": f(row, "greedy_exact_match_mean"),
            }
        )
    candidates.sort(key=lambda row: (row["cost"], row["validation_loss_mean"]))
    frontier: list[dict[str, Any]] = []
    best = float("inf")
    for row in candidates:
        if row["validation_loss_mean"] < best:
            frontier.append(row)
            best = row["validation_loss_mean"]
    return frontier


def transition_comparison(dense_path: Path, moe_path: Path) -> list[dict[str, Any]]:
    dense = {row["model"]: row for row in read_csv(dense_path)}
    rows: list[dict[str, Any]] = []
    for moe in read_csv(moe_path):
        reference = moe["dense_reference"]
        d = dense[reference]
        row: dict[str, Any] = {
            "dense_model": reference,
            "moe_model": moe["model"],
            "dense_parameters": int(float(d["parameter_count"])),
            "moe_active_parameter_proxy": int(float(moe["active_parameter_proxy"])),
            "moe_total_parameter_count": int(float(moe["total_parameter_count"])),
        }
        for threshold in (50, 95, 99):
            field = f"first_steps_at_{threshold}pct_em"
            dense_value = d[field]
            moe_value = moe[field]
            row[f"dense_{field}"] = int(float(dense_value)) if dense_value else None
            row[f"moe_{field}"] = int(float(moe_value)) if moe_value else None
        rows.append(row)
    rows.sort(key=lambda row: row["dense_parameters"])
    return rows


def plot_frontier(rows: list[dict[str, Any]], output: Path, title: str, xlabel: str) -> None:
    figure, axis = plt.subplots(figsize=(8.6, 5.3))
    for architecture in ("dense", "moe"):
        subset = [row for row in rows if row["architecture"] == architecture]
        if not subset:
            continue
        axis.errorbar(
            [row["cost"] for row in subset],
            [row["validation_loss_mean"] for row in subset],
            yerr=[row["validation_loss_std"] for row in subset],
            marker="o",
            capsize=3,
            label=architecture,
        )
        for row in subset:
            axis.annotate(
                f"{row['model']}\n{row['steps']} steps",
                (row["cost"], row["validation_loss_mean"]),
                fontsize=6.5,
                xytext=(3, 3),
                textcoords="offset points",
            )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Mean validation answer-token cross-entropy")
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg")
    plt.close(figure)


def plot_time_ratios(rows: list[dict[str, Any]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.6, 5.3))
    by_steps: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_steps[int(row["steps"])].append(row)
    for steps, group in sorted(by_steps.items()):
        group.sort(key=lambda row: int(row["dense_parameters"]))
        axis.plot(
            [row["dense_parameters"] for row in group],
            [row["moe_over_dense_time_ratio_mean"] for row in group],
            marker="o",
            label=f"{steps} steps",
        )
    axis.axhline(1.0, linestyle="--", linewidth=1)
    axis.set_xscale("log")
    axis.set_xlabel("Dense reference parameters")
    axis.set_ylabel("MoE / dense mean training-time ratio")
    axis.set_title("Measured routing overhead on NVIDIA T4")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg")
    plt.close(figure)


def analyze(dense: Path, moe: Path, output: Path, assets: Path) -> dict[str, Any]:
    dense_runs = read_csv(dense / "final_runs.csv")
    moe_runs = read_csv(moe / "final_runs.csv")
    dense_grouped = read_csv(dense / "final_grouped.csv")
    moe_grouped = read_csv(moe / "final_grouped.csv")

    pairs = paired_runs(dense_runs, moe_runs)
    paired_grouped = aggregate_pairs(pairs)
    compute_frontier = combined_frontier(
        dense_grouped, moe_grouped, "estimated_training_flops"
    )
    wallclock_frontier = combined_frontier(
        dense_grouped, moe_grouped, "total_training_seconds_mean"
    )
    transitions = transition_comparison(
        dense / "transition_table.csv", moe / "transition_table.csv"
    )

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "paired_runs.csv", pairs)
    write_csv(output / "paired_grouped.csv", paired_grouped)
    write_csv(output / "combined_compute_frontier.csv", compute_frontier)
    write_csv(output / "combined_wallclock_frontier.csv", wallclock_frontier)
    write_csv(output / "transition_comparison.csv", transitions)

    plot_frontier(
        compute_frontier,
        assets / "dense_vs_moe_compute_frontier.svg",
        "Dense versus MoE empirical compute-loss frontier",
        "Approximate training FLOPs",
    )
    plot_frontier(
        wallclock_frontier,
        assets / "dense_vs_moe_wallclock_frontier.svg",
        "Dense versus MoE measured wall-clock frontier",
        "Mean total training seconds",
    )
    plot_time_ratios(paired_grouped, assets / "dense_vs_moe_time_ratio.svg")

    summary = {
        "status": "verified",
        "paired_run_count": len(pairs),
        "paired_group_count": len(paired_grouped),
        "compute_frontier_points": len(compute_frontier),
        "wallclock_frontier_points": len(wallclock_frontier),
        "compute_frontier_architecture_counts": {
            architecture: sum(row["architecture"] == architecture for row in compute_frontier)
            for architecture in ("dense", "moe")
        },
        "wallclock_frontier_architecture_counts": {
            architecture: sum(row["architecture"] == architecture for row in wallclock_frontier)
            for architecture in ("dense", "moe")
        },
        "comparison_rule": (
            "Pairs share dense architecture size, training-step budget, seed, data split, "
            "sampler, optimizer, and evaluation protocol. MoE active expert width matches "
            "dense FFN matrix work; router overhead is reported explicitly."
        ),
        "caveat": (
            "The 6*N*D FLOP proxy excludes MoE routing, sorting, dispatch, scatter-add, "
            "and kernel-efficiency effects. The wall-clock frontier is therefore equally important."
        ),
    }
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", type=Path, default=DEFAULT_DENSE)
    parser.add_argument("--moe", type=Path, default=DEFAULT_MOE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    args = parser.parse_args()
    summary = analyze(args.dense, args.moe, args.output, args.assets)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
