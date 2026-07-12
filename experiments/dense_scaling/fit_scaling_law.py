"""Diagnose whether the dense-scaling pilot identifies a Chinchilla surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RESULTS = Path("artifacts/dense-scaling-pilot-t4/pilot_results.json")
DEFAULT_OUTPUT = Path("artifacts/dense-scaling-pilot-t4/pilot_fit.json")
N_REFERENCE = 1_000_000.0
D_REFERENCE = 1_000_000.0


def _loss_from_record(record: dict[str, Any]) -> float:
    best = record.get("best") or {}
    return float(best["loss"])


def _nonnegative_linear_fit(
    design: np.ndarray,
    losses: np.ndarray,
) -> tuple[float, np.ndarray] | None:
    best: tuple[float, np.ndarray] | None = None

    # Enumerate the active sets for E, A, and B. This small exact search
    # avoids adding SciPy only for non-negative least squares.
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


def _fit_at_exponents(
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
    return _nonnegative_linear_fit(design, losses)


def _search(
    n_scaled: np.ndarray,
    d_scaled: np.ndarray,
    losses: np.ndarray,
    alpha_values: np.ndarray,
    beta_values: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    best: tuple[float, float, float, np.ndarray] | None = None

    for alpha in alpha_values:
        for beta in beta_values:
            candidate = _fit_at_exponents(
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
        raise RuntimeError("No non-negative scaling-law candidate was found.")

    return best


def _log_r_squared(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    observed_log = np.log10(observed)
    predicted_log = np.log10(predicted)
    residual_sum = float(np.sum(np.square(observed_log - predicted_log)))
    total_sum = float(np.sum(np.square(observed_log - observed_log.mean())))
    return 1.0 - residual_sum / total_sum if total_sum > 0.0 else float("nan")


def _multiplicative_diagnostic(
    n_scaled: np.ndarray,
    d_scaled: np.ndarray,
    losses: np.ndarray,
) -> dict[str, float | str]:
    design = np.column_stack(
        (
            np.ones_like(losses),
            -np.log(n_scaled),
            -np.log(d_scaled),
        )
    )
    coefficients, *_ = np.linalg.lstsq(
        design,
        np.log(losses),
        rcond=None,
    )
    log_k, alpha, beta = (float(value) for value in coefficients)
    predictions = np.exp(design @ coefficients)

    return {
        "equation": (
            "L(N,D) = K*(N/1e6)^(-alpha)*(D/1e6)^(-beta)"
        ),
        "K": float(np.exp(log_k)),
        "alpha": alpha,
        "beta": beta,
        "log10_r_squared": _log_r_squared(losses, predictions),
        "warning": (
            "This multiplicative regression is only a directional diagnostic. "
            "It does not define an interior Chinchilla compute optimum."
        ),
    }


def fit_records(
    records: list[dict[str, Any]],
    *,
    min_loss: float = 1e-3,
) -> dict[str, Any]:
    selected = [
        record
        for record in records
        if record.get("status") == "complete"
        and _loss_from_record(record) >= min_loss
    ]

    if len(selected) < 8:
        raise ValueError(
            "At least eight complete, non-saturated observations are required."
        )

    parameters = np.asarray(
        [float(record["parameter_count"]) for record in selected],
        dtype=np.float64,
    )
    tokens = np.asarray(
        [float(record["sequence_tokens_seen"]) for record in selected],
        dtype=np.float64,
    )
    losses = np.asarray(
        [_loss_from_record(record) for record in selected],
        dtype=np.float64,
    )

    n_scaled = parameters / N_REFERENCE
    d_scaled = tokens / D_REFERENCE

    coarse = _search(
        n_scaled,
        d_scaled,
        losses,
        np.linspace(0.05, 5.0, 80),
        np.linspace(0.05, 5.0, 80),
    )
    _, coarse_alpha, coarse_beta, _ = coarse

    alpha_low = max(0.01, coarse_alpha - 0.25)
    alpha_high = coarse_alpha + 0.25
    beta_low = max(0.01, coarse_beta - 0.25)
    beta_high = coarse_beta + 0.25

    objective, alpha, beta, coefficients = _search(
        n_scaled,
        d_scaled,
        losses,
        np.linspace(alpha_low, alpha_high, 120),
        np.linspace(beta_low, beta_high, 120),
    )

    irreducible, model_coefficient, data_coefficient = (
        float(value) for value in coefficients
    )
    predictions = (
        irreducible
        + model_coefficient * np.power(n_scaled, -alpha)
        + data_coefficient * np.power(d_scaled, -beta)
    )
    r2_log = _log_r_squared(losses, predictions)
    log_rmse = objective**0.5

    boundary_hit = (
        alpha >= alpha_high - 1e-9
        or beta >= beta_high - 1e-9
        or alpha <= alpha_low + 1e-9
        or beta <= beta_low + 1e-9
    )
    floor_collapsed = irreducible == 0.0
    stable = (
        not boundary_hit
        and not floor_collapsed
        and r2_log >= 0.9
        and log_rmse <= 0.25
    )

    additive_candidate = {
        "equation": (
            "L(N,D) = E + A*(N/1e6)^(-alpha) "
            "+ B*(D/1e6)^(-beta)"
        ),
        "E": irreducible,
        "A": model_coefficient,
        "B": data_coefficient,
        "alpha": alpha,
        "beta": beta,
        "log10_rmse": log_rmse,
        "log10_r_squared": r2_log,
        "boundary_hit": boundary_hit,
        "floor_collapsed_to_zero": floor_collapsed,
    }

    result: dict[str, Any] = {
        "status": (
            "stable_pilot_fit"
            if stable
            else "pilot_does_not_identify_stable_chinchilla_exponents"
        ),
        "data_definition": (
            "D is the number of full sequence tokens processed. "
            "Supervised answer tokens are also recorded separately."
        ),
        "minimum_loss_included": min_loss,
        "observations_total": len(records),
        "observations_used": len(selected),
        "run_ids_used": [record["run_id"] for record in selected],
        "normalization": {
            "parameter_reference": N_REFERENCE,
            "token_reference": D_REFERENCE,
        },
        "additive_chinchilla_candidate": additive_candidate,
        "multiplicative_diagnostic": _multiplicative_diagnostic(
            n_scaled,
            d_scaled,
            losses,
        ),
        "conclusion": (
            "The one-seed pilot locates the transition and saturation regions, "
            "but it is not sufficient for a defensible compute-optimal law. "
            "A denser multi-seed experiment is required."
        ),
    }

    if stable:
        result["compute_optimal_parameter_exponent"] = beta / (alpha + beta)
        result["compute_optimal_data_exponent"] = alpha / (alpha + beta)

    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-loss", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    records = json.loads(args.results.read_text())
    result = fit_records(records, min_loss=args.min_loss)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print(json.dumps(result, indent=2))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
