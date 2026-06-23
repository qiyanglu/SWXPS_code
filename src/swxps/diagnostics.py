"""Local least-squares uncertainty and identifiability diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .fitting import FitParameter

if TYPE_CHECKING:
    from .jax_least_squares import JaxLeastSquaresOptimizationResult


@dataclass(frozen=True)
class ParameterDiagnostics:
    """Local parameter uncertainty and Jacobian identifiability summary."""

    names: tuple[str, ...]
    values: np.ndarray
    bounds: tuple[tuple[float | None, float | None], ...] | None
    residuals: np.ndarray
    jacobian: np.ndarray
    covariance: np.ndarray
    stderr: np.ndarray
    correlation: np.ndarray
    singular_values: np.ndarray
    condition_number: float
    dof: int
    residual_variance: float


def compute_parameter_diagnostics(
    values,
    names=None,
    bounds=None,
    residuals=None,
    jacobian=None,
    covariance=None,
    rcond=1.0e-12,
) -> ParameterDiagnostics:
    """Compute local covariance, correlation, and Jacobian diagnostics."""

    value_array = np.asarray(values, dtype=float)
    if value_array.ndim != 1:
        raise ValueError("values must be a one-dimensional array")
    parameter_count = value_array.size
    if not np.all(np.isfinite(value_array)):
        raise ValueError("values must be finite")
    if not np.isfinite(rcond) or rcond < 0:
        raise ValueError("rcond must be finite and non-negative")

    name_tuple = _normalize_names(names, parameter_count)
    bound_tuple = _normalize_bounds(bounds, parameter_count)
    residual_array = _normalize_residuals(residuals)
    jacobian_array = _normalize_jacobian(jacobian, parameter_count)
    if (
        residual_array is not None
        and jacobian_array is not None
        and jacobian_array.shape[0] != residual_array.size
    ):
        raise ValueError("jacobian shape must be (N, P) for N residuals and P values")

    if covariance is None:
        if residual_array is None or jacobian_array is None:
            raise ValueError(
                "residuals and jacobian are required when covariance is not provided"
            )
        dof = max(1, residual_array.size - parameter_count)
        residual_variance = float(np.dot(residual_array, residual_array) / dof)
        normal_matrix = jacobian_array.T @ jacobian_array
        covariance_array = residual_variance * np.linalg.pinv(
            normal_matrix,
            rcond=rcond,
        )
    else:
        covariance_array = np.asarray(covariance, dtype=float)
        if covariance_array.shape != (parameter_count, parameter_count):
            raise ValueError("covariance shape must be (P, P)")
        if residual_array is None:
            dof = 0
            residual_variance = float("nan")
        else:
            dof = max(1, residual_array.size - parameter_count)
            residual_variance = float(np.dot(residual_array, residual_array) / dof)

    diagonal = np.diag(covariance_array)
    stderr = np.sqrt(np.maximum(diagonal, 0.0))
    correlation = _correlation_from_covariance(covariance_array, stderr)
    singular_values = (
        np.array([], dtype=float)
        if jacobian_array is None
        else np.linalg.svd(jacobian_array, compute_uv=False)
    )
    condition_number = _condition_number(
        singular_values,
        float(rcond),
        None if jacobian_array is None else parameter_count,
    )

    stored_residuals = (
        np.array([], dtype=float) if residual_array is None else residual_array
    )
    stored_jacobian = (
        np.empty((0, parameter_count), dtype=float)
        if jacobian_array is None
        else jacobian_array
    )
    return ParameterDiagnostics(
        names=name_tuple,
        values=_readonly(value_array),
        bounds=bound_tuple,
        residuals=_readonly(stored_residuals),
        jacobian=_readonly(stored_jacobian),
        covariance=_readonly(covariance_array),
        stderr=_readonly(stderr),
        correlation=_readonly(correlation),
        singular_values=_readonly(singular_values),
        condition_number=condition_number,
        dof=dof,
        residual_variance=residual_variance,
    )


def diagnostics_from_least_squares_result(
    result: JaxLeastSquaresOptimizationResult,
    parameters: Sequence[FitParameter],
    rcond: float = 1.0e-12,
) -> ParameterDiagnostics:
    """Build diagnostics from a JAX least-squares result and declarations."""

    parameter_tuple = tuple(parameters)
    values = np.asarray(
        [result.best_parameters[parameter.name] for parameter in parameter_tuple],
        dtype=float,
    )
    return compute_parameter_diagnostics(
        values,
        names=tuple(parameter.name for parameter in parameter_tuple),
        bounds=tuple(
            (float(parameter.lower), float(parameter.upper))
            for parameter in parameter_tuple
        ),
        residuals=result.final_residuals,
        jacobian=result.final_jacobian,
        covariance=result.covariance,
        rcond=rcond,
    )


def plot_parameter_estimates(
    diagnostics: ParameterDiagnostics,
    ci: float | None = 0.95,
    ax=None,
    show_bounds: bool = True,
):
    """Plot parameter estimates, normal-approximation intervals, and bounds."""

    plt = _load_pyplot()
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(8.0, max(3.0, 0.48 * len(diagnostics.names) + 1.4))
        )
    else:
        fig = ax.figure
    multiplier = _ci_multiplier(ci)
    y = np.arange(len(diagnostics.names))

    if show_bounds and diagnostics.bounds is not None:
        for index, (lower, upper) in enumerate(diagnostics.bounds):
            if lower is not None and upper is not None:
                ax.hlines(
                    index,
                    lower,
                    upper,
                    color="tab:blue",
                    alpha=0.22,
                    linewidth=6.0,
                    zorder=1,
                )
            else:
                for bound in (lower, upper):
                    if bound is not None:
                        ax.plot(
                            bound,
                            index,
                            marker="|",
                            color="tab:blue",
                            markersize=12,
                            zorder=1,
                        )

    if multiplier is None:
        ax.plot(
            diagnostics.values,
            y,
            "o",
            color="black",
            label="best fit",
            zorder=3,
        )
    else:
        errors = multiplier * diagnostics.stderr
        errors = np.where(np.isfinite(errors), errors, 0.0)
        ax.errorbar(
            diagnostics.values,
            y,
            xerr=errors,
            fmt="o",
            color="black",
            ecolor="tab:orange",
            capsize=3,
            label=f"{int(round(100 * ci))}% CI",
            zorder=3,
        )
    ax.set_yticks(y, labels=diagnostics.names)
    ax.invert_yaxis()
    ax.set_xlabel("Parameter value")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_correlation_matrix(
    diagnostics: ParameterDiagnostics,
    ax=None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    annotate: bool = True,
):
    """Plot the parameter correlation matrix without a seaborn dependency."""

    plt = _load_pyplot()
    if ax is None:
        size = max(4.0, 0.55 * len(diagnostics.names) + 2.0)
        fig, ax = plt.subplots(figsize=(size, size))
    else:
        fig = ax.figure
    image = ax.imshow(
        diagnostics.correlation,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
    )
    indices = np.arange(len(diagnostics.names))
    ax.set_xticks(indices, labels=diagnostics.names, rotation=45, ha="right")
    ax.set_yticks(indices, labels=diagnostics.names)
    if annotate:
        midpoint = 0.5 * (vmin + vmax)
        for row in indices:
            for column in indices:
                value = diagnostics.correlation[row, column]
                if np.isfinite(value):
                    color = (
                        "white"
                        if abs(value - midpoint) > 0.25 * (vmax - vmin)
                        else "black"
                    )
                    ax.text(
                        column,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=8,
                    )
    ax.set_title("Parameter correlation")
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    return fig, ax


def plot_singular_values(diagnostics: ParameterDiagnostics, ax=None):
    """Plot available Jacobian singular values on a logarithmic scale."""

    if diagnostics.singular_values.size == 0:
        raise ValueError("Jacobian singular values are not available")
    plt = _load_pyplot()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
    else:
        fig = ax.figure
    indices = np.arange(1, diagnostics.singular_values.size + 1)
    plotted_values = np.where(diagnostics.singular_values > 0, diagnostics.singular_values, np.nan)
    ax.plot(indices, plotted_values, "o-", color="tab:blue")
    ax.set_yscale("log")
    ax.set_xlabel("Singular-value index")
    ax.set_ylabel("Singular value")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def _normalize_names(names, parameter_count: int) -> tuple[str, ...]:
    if names is None:
        return tuple(f"p{index}" for index in range(parameter_count))
    result = tuple(str(name) for name in names)
    if len(result) != parameter_count:
        raise ValueError("names must contain one entry per value")
    return result


def _normalize_bounds(bounds, parameter_count: int):
    if bounds is None:
        return None
    result = []
    for bound in bounds:
        if len(bound) != 2:
            raise ValueError("each bounds entry must contain lower and upper")
        lower = None if bound[0] is None else float(bound[0])
        upper = None if bound[1] is None else float(bound[1])
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("lower bounds must not exceed upper bounds")
        result.append((lower, upper))
    if len(result) != parameter_count:
        raise ValueError("bounds must contain one entry per value")
    return tuple(result)


def _normalize_residuals(residuals):
    if residuals is None:
        return None
    result = np.asarray(residuals, dtype=float)
    if result.ndim != 1:
        raise ValueError("residuals must be a one-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError("residuals must be finite")
    return result


def _normalize_jacobian(jacobian, parameter_count: int):
    if jacobian is None:
        return None
    result = np.asarray(jacobian, dtype=float)
    if result.ndim != 2 or result.shape[1] != parameter_count:
        raise ValueError("jacobian must be a two-dimensional array with P columns")
    if not np.all(np.isfinite(result)):
        raise ValueError("jacobian must be finite")
    return result


def _correlation_from_covariance(covariance: np.ndarray, stderr: np.ndarray):
    denominator = np.outer(stderr, stderr)
    correlation = np.full(covariance.shape, np.nan, dtype=float)
    valid = np.isfinite(covariance) & np.isfinite(denominator) & (denominator > 0)
    np.divide(covariance, denominator, out=correlation, where=valid)
    finite_diagonal = np.isfinite(np.diag(covariance)) & (np.diag(covariance) >= 0)
    diagonal_indices = np.arange(len(stderr))
    correlation[diagonal_indices[finite_diagonal], diagonal_indices[finite_diagonal]] = 1.0
    return correlation


def _condition_number(
    singular_values: np.ndarray,
    rcond: float,
    parameter_count: int | None,
) -> float:
    if singular_values.size == 0:
        return float("nan")
    largest = float(singular_values[0])
    if not np.isfinite(largest) or largest <= 0:
        return float("inf")
    if parameter_count is not None and singular_values.size < parameter_count:
        return float("inf")
    threshold = rcond * largest
    if np.any(~np.isfinite(singular_values)) or np.any(singular_values <= threshold):
        return float("inf")
    return largest / float(singular_values[-1])


def _ci_multiplier(ci: float | None) -> float | None:
    if ci is None:
        return None
    if np.isclose(ci, 0.68):
        return 1.0
    if np.isclose(ci, 0.95):
        return 1.96
    raise ValueError("ci must be None, 0.68, or 0.95")


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _load_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("matplotlib is required for parameter diagnostic plots") from error
    return plt
