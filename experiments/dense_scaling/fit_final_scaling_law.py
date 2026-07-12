"""Fit and diagnose task-specific dense scaling laws."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_GROUPED = Path(
    "artifacts/dense-scaling-final-t4/final_grouped.csv"
)
DEFAULT_OUTPUT = Path(
    "artifacts/dense-scaling-final-t4/final_fit.json"
)
DEFAULT_FRONTIER = Path(
    "artifacts/dense-scaling-final-t4/empirical_compute_frontier.csv"
)
N_REFERENCE = 1_000_000.0
D_REFERENCE = 1_000_000.0
THRESHOLDS = (1e-4, 1e-3, 1e-2, 5e-2, 1e-1)


def read_grouped(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def nonnegative_linear_fit(
    design: np.ndarray,
    losses: np.ndarray,
) -> tuple[float, np.ndarray] | None:
    best: tuple[float, np.ndarray] | None = None
    active_sets = (
        (0, 1, 2),
        (1, 2),
        (0, 1),
        (0, 2),
        (1,),
        (2,),
    )

    for active in active_sets:
        active_design = design[:, active]
        active_coefficients, *_ = np.linalg.lstsq(
            active_design,
            losses,
            rcond=None,
        )
        if np.any(active_coefficients < 0.0):
            continue

        coefficients = np.zeros(3, dtype=np.float64)
        coefficients[list(active)] = active_coefficients
        predictions = design @ coefficients
        if np.any(predictions <= 0.0):
            continue

        residuals = np.log10(predictions) - np.log10(losses)
        objective = float(np.mean(np.square(residuals)))

        if best is None or objective < best[0]:
            best = objective, coefficients

    return best


def fit_at_exponents(
    n_scaled: np.ndarray,
    d_scaled: np.ndarray,
    losses: np.ndarray,
    alpha: float,
    beta: float,
) -> tuple[float, np.ndarray] | None:
    design = np.column_stack(
        (
            np.ones_like(losses),
            np.power(n_scaled, -alpha),
            np.power(d_scaled, -beta),
        )
    )
    return nonnegative_linear_fit(design, losses)


def search(
    n_scaled: np.ndarray,
    d_scaled: np.ndarray,
    losses: np.ndarray,
    alpha_values: np.ndarray,
    beta_values: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    best: tuple[float, float, float, np.ndarray] | None = None

    for alpha in alpha_values:
        for beta in beta_values:
            candidate = fit_at_exponents(
                n_scaled,
                d_scaled,
                losses,
                float(alpha),
                float(beta),
            )
            if candidate is None:
                continue

            objective, coefficients = candidate
            if best is None or objective < best[0]:
                best = (
                    objective,
                    float(alpha),
                    float(beta),
                    coefficients,
                )

    if best is None:
        raise RuntimeError("No non-negative candidate was found.")

    return best


def log_r_squared(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    observed_log = np.log10(observed)
    predicted_log = np.log10(predicted)
    residual_sum = float(np.sum(np.square(observed_log - predicted_log)))
    total_sum = float(np.sum(np.square(observed_log - observed_log.mean())))
    return 1.0 - residual_sum / total_sum if total_sum > 0.0 else float("nan")


def fit_threshold(
    rows: list[dict[str, Any]],
    minimum_loss: float,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if float(row["validation_loss_mean"]) >= minimum_loss
    ]

    if len(selected) < 10:
        return {
            "minimum_loss": minimum_loss,
            "status": "insufficient_observations",
            "observations": len(selected),
        }

    parameters = np.asarray(
        [float(row["parameter_count"]) for row in selected],
        dtype=np.float64,
    )
    tokens = np.asarray(
        [float(row["sequence_tokens_seen"]) for row in selected],
        dtype=np.float64,
    )
    losses = np.asarray(
        [float(row["validation_loss_mean"]) for row in selected],
        dtype=np.float64,
    )

    n_scaled = parameters / N_REFERENCE
    d_scaled = tokens / D_REFERENCE

    _, coarse_alpha, coarse_beta, _ = search(
        n_scaled,
        d_scaled,
        losses,
        np.linspace(0.05, 5.0, 72),
        np.linspace(0.05, 5.0, 72),
    )

    alpha_low = max(0.01, coarse_alpha - 0.25)
    alpha_high = coarse_alpha + 0.25
    beta_low = max(0.01, coarse_beta - 0.25)
    beta_high = coarse_beta + 0.25

    objective, alpha, beta, coefficients = search(
        n_scaled,
        d_scaled,
        losses,
        np.linspace(alpha_low, alpha_high, 100),
        np.linspace(beta_low, beta_high, 100),
    )

    irreducible, model_coefficient, data_coefficient = (
        float(value) for value in coefficients
    )
    predictions = (
        irreducible
        + model_coefficient * np.power(n_scaled, -alpha)
        + data_coefficient * np.power(d_scaled, -beta)
    )

    boundary_hit = (
        math.isclose(alpha, alpha_low)
        or math.isclose(alpha, alpha_high)
        or math.isclose(beta, beta_low)
        or math.isclose(beta, beta_high)
    )
    floor_collapsed = irreducible <= 1e-12
    r2 = log_r_squared(losses, predictions)
    rmse = objective**0.5
    candidate_stable = (
        not boundary_hit
        and not floor_collapsed
        and r2 >= 0.9
        and rmse <= 0.25
    )

    result: dict[str, Any] = {
        "minimum_loss": minimum_loss,
        "status": "candidate_stable" if candidate_stable else "unstable",
        "observations": len(selected),
        "E": irreducible,
        "A": model_coefficient,
        "B": data_coefficient,
        "alpha": alpha,
        "beta": beta,
        "log10_rmse": rmse,
        "log10_r_squared": r2,
        "boundary_hit": boundary_hit,
        "floor_collapsed_to_zero": floor_collapsed,
    }

    if candidate_stable:
        result["compute_optimal_parameter_exponent"] = beta / (alpha + beta)
        result["compute_optimal_data_exponent"] = alpha / (alpha + beta)

    return result


def threshold_stability(fits: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        fit for fit in fits if fit.get("status") == "candidate_stable"
    ]

    if len(candidates) < 3:
        return {
            "status": "not_identified",
            "reason": (
                "Fewer than three loss thresholds produced individually "
                "stable additive fits."
            ),
            "stable_threshold_count": len(candidates),
        }

    alphas = np.asarray(
        [float(fit["alpha"]) for fit in candidates],
        dtype=np.float64,
    )
    betas = np.asarray(
        [float(fit["beta"]) for fit in candidates],
        dtype=np.float64,
    )

    alpha_cv = float(alphas.std(ddof=1) / abs(alphas.mean()))
    beta_cv = float(betas.std(ddof=1) / abs(betas.mean()))
    identified = alpha_cv <= 0.25 and beta_cv <= 0.25

    result: dict[str, Any] = {
        "status": "identified" if identified else "not_identified",
        "stable_threshold_count": len(candidates),
        "alpha_mean": float(alphas.mean()),
        "alpha_std": float(alphas.std(ddof=1)),
        "alpha_coefficient_of_variation": alpha_cv,
        "beta_mean": float(betas.mean()),
        "beta_std": float(betas.std(ddof=1)),
        "beta_coefficient_of_variation": beta_cv,
    }

    if identified:
        alpha = float(alphas.mean())
        beta = float(betas.mean())
        result["compute_optimal_parameter_exponent"] = beta / (alpha + beta)
        result["compute_optimal_data_exponent"] = alpha / (alpha + beta)
    else:
        result["reason"] = (
            "Exponent estimates vary too much across saturation cutoffs."
        )

    return result


def empirical_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            int(float(row["estimated_training_flops"])),
            float(row["validation_loss_mean"]),
        ),
    )

    frontier: list[dict[str, Any]] = []
    best_loss = float("inf")

    for row in ordered:
        loss = float(row["validation_loss_mean"])
        if loss < best_loss:
            frontier.append(
                {
                    "model": row["model"],
                    "parameter_count": int(float(row["parameter_count"])),
                    "steps": int(float(row["steps"])),
                    "sequence_tokens_seen": int(
                        float(row["sequence_tokens_seen"])
                    ),
                    "estimated_training_flops": int(
                        float(row["estimated_training_flops"])
                    ),
                    "validation_loss_mean": loss,
                    "validation_loss_std": float(
                        row["validation_loss_std"]
                    ),
                    "greedy_exact_match_mean": float(
                        row["greedy_exact_match_mean"]
                    ),
                    "greedy_exact_match_std": float(
                        row["greedy_exact_match_std"]
                    ),
                }
            )
            best_loss = loss

    return frontier


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty frontier.")

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grouped", type=Path, default=DEFAULT_GROUPED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    rows = read_grouped(args.grouped)

    fits = [fit_threshold(rows, threshold) for threshold in THRESHOLDS]
    stability = threshold_stability(fits)
    frontier = empirical_frontier(rows)

    result = {
        "status": stability["status"],
        "equation": (
            "L(N,D) = E + A*(N/1e6)^(-alpha) "
            "+ B*(D/1e6)^(-beta)"
        ),
        "data_definition": (
            "D is repeated full-sequence token exposure from a fixed pool "
            "of 200,000 unique training pairs."
        ),
        "group_count": len(rows),
        "seed_count_per_group": 3,
        "threshold_fits": fits,
        "threshold_stability": stability,
        "empirical_frontier_points": len(frontier),
        "conclusion": (
            "Use compute-optimal exponents only when status is 'identified'. "
            "Otherwise report the empirical compute-loss frontier and the "
            "failure of the additive surface to remain stable across cutoffs."
        ),
    }

    args.output.write_text(json.dumps(result, indent=2) + "\n")
    write_csv(args.frontier, frontier)

    print(json.dumps(result, indent=2))
    print(f"\nFit report: {args.output}")
    print(f"Empirical frontier: {args.frontier}")


if __name__ == "__main__":
    main()
