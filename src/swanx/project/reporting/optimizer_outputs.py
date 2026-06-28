"""Optimizer-specific report outputs for YAML project runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..builder import BuiltProject
from ._shared import _status_dict, _write_array, _write_csv, _write_json
from .plots import _write_bayesian_plot_outputs, _write_least_squares_plot_outputs


def write_method_outputs(output: Path, method: str, result: Any, built: BuiltProject | None = None) -> list[str]:
    if result is None:
        return []
    notes: list[str] = []
    if method == "jax_least_squares":
        _write_least_squares_outputs(output / "optimizer" / "least_squares", result, built)
        notes.extend(_write_least_squares_plot_outputs(output / "plots", result, built))
    elif method == "jax_gradient":
        _write_gradient_outputs(output / "optimizer" / "gradient", result)
    elif method == "bayesian_optimization":
        _write_bayesian_outputs(output / "optimizer" / "bayesian", result)
        notes.extend(_write_bayesian_plot_outputs(output / "plots", result, built))
    return notes

def _write_least_squares_outputs(directory: Path, result: Any, built: BuiltProject | None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "status.json", _status_dict(result, objective_attr="final_cost"))
    _write_array(directory / "residual_vector.csv", getattr(result, "final_residuals", None), "residual")
    _write_array(directory / "jacobian.csv", getattr(result, "final_jacobian", None), "value")
    _write_array(directory / "covariance.csv", getattr(result, "covariance", None), "value")
    covariance = getattr(result, "covariance", None)
    if covariance is not None:
        covariance = np.asarray(covariance, dtype=float)
        sigma = np.sqrt(np.diag(covariance))
        denom = np.outer(sigma, sigma)
        correlation = np.divide(covariance, denom, out=np.zeros_like(covariance), where=denom != 0)
        _write_array(directory / "correlation.csv", correlation, "value")
        _write_csv(directory / "parameter_uncertainty.csv", _least_squares_uncertainty_rows(result, built, sigma))
    history = getattr(result, "history", ())
    _write_csv(directory / "convergence_history.csv", [
        ["iteration", "cost", "gradient_norm", "parameters_json"],
        *[
            [record.iteration, record.cost, record.gradient_norm, json.dumps(record.parameters, sort_keys=True)]
            for record in history
        ],
    ])
    active_bounds = [["parameter", "active_bound"]]
    if built is not None:
        best = getattr(result, "best_parameters", {})
        for parameter in built.spec.varying_parameters():
            value = float(best.get(parameter.name, parameter.value))
            bound = ""
            if parameter.lower is not None and np.isclose(value, parameter.lower):
                bound = "lower"
            elif parameter.upper is not None and np.isclose(value, parameter.upper):
                bound = "upper"
            active_bounds.append([parameter.name, bound])
    _write_csv(directory / "active_bounds.csv", active_bounds)

def _least_squares_uncertainty_rows(result: Any, built: BuiltProject | None, sigma: np.ndarray) -> list[list[Any]]:
    rows: list[list[Any]] = [["parameter", "best_value", "stderr", "ci95_low", "ci95_high", "lower", "upper"]]
    if built is None:
        for index, stderr in enumerate(sigma):
            best = np.nan
            rows.append([f"parameter_{index}", best, stderr, best, best, "", ""])
        return rows
    best_parameters = getattr(result, "best_parameters", {})
    for parameter, stderr in zip(built.spec.varying_parameters(), sigma):
        best_value = float(best_parameters.get(parameter.name, parameter.value))
        ci = 1.96 * float(stderr)
        rows.append([
            parameter.name,
            best_value,
            float(stderr),
            best_value - ci,
            best_value + ci,
            parameter.lower,
            parameter.upper,
        ])
    return rows

def _write_gradient_outputs(directory: Path, result: Any) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "status.json", _status_dict(result, objective_attr="best_loss"))
    history = getattr(result, "history", ())
    _write_csv(directory / "objective_history.csv", [
        ["iteration", "loss"],
        *[[record.iteration, record.loss] for record in history],
    ])
    _write_csv(directory / "parameter_history.csv", [
        ["iteration", "parameters_json"],
        *[[record.iteration, json.dumps(record.parameters, sort_keys=True)] for record in history],
    ])
    _write_csv(directory / "gradient_norm_history.csv", [
        ["iteration", "gradient_norm"],
        *[[record.iteration, record.gradient_norm] for record in history],
    ])
    _write_array(directory / "final_gradient.csv", getattr(result, "final_gradient", None), "gradient")

def _write_bayesian_outputs(directory: Path, result: Any) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "status.json", _status_dict(result, objective_attr="best_objective"))
    history = getattr(getattr(result, "history", None), "evaluations", ())
    best_objective = float("inf")
    best_parameters: dict[str, float] = {}
    evaluations = [["evaluation", "objective", "parameters_json"]]
    best_so_far = [["evaluation", "best_objective", "best_parameters_json"]]
    samples = [["evaluation", "parameters_json"]]
    for index, evaluation in enumerate(history, start=1):
        parameters_json = json.dumps(evaluation.parameters, sort_keys=True)
        objective = float(evaluation.objective)
        evaluations.append([index, objective, parameters_json])
        samples.append([index, parameters_json])
        if objective <= best_objective:
            best_objective = objective
            best_parameters = dict(evaluation.parameters)
        best_so_far.append([index, best_objective, json.dumps(best_parameters, sort_keys=True)])
    _write_csv(directory / "evaluations.csv", evaluations)
    _write_csv(directory / "best_so_far.csv", best_so_far)
    _write_csv(directory / "parameter_samples.csv", samples)
    _write_json(directory / "stage_summary.json", {"stages": []})
