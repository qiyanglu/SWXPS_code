"""Local least-squares uncertainty and identifiability diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
import warnings

import numpy as np

from ..fitting.core import FitParameter

if TYPE_CHECKING:
    from ..fitting.jax_least_squares import JaxLeastSquaresOptimizationResult


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
        dof = residual_array.size - parameter_count
        if dof <= 0:
            raise ValueError(
                "Jacobian-derived covariance requires N > P residual degrees of freedom"
            )
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
            dof = residual_array.size - parameter_count
            residual_variance = (
                float(np.dot(residual_array, residual_array) / dof)
                if dof > 0
                else float("nan")
            )

    covariance_array = _validate_and_symmetrize_covariance(
        covariance_array,
        parameter_count,
    )
    diagonal = np.diag(covariance_array)
    stderr = np.sqrt(diagonal)
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
        covariance=None,
        rcond=rcond,
    )



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


def _validate_and_symmetrize_covariance(
    covariance: np.ndarray,
    parameter_count: int,
) -> np.ndarray:
    """Validate covariance quality and return a symmetric PSD matrix."""

    result = np.asarray(covariance, dtype=float)
    if result.shape != (parameter_count, parameter_count):
        raise ValueError("covariance shape must be (P, P)")
    if not np.all(np.isfinite(result)):
        raise ValueError("covariance must contain only finite values")

    scale = max(1.0, float(np.max(np.abs(result))))
    symmetry_tolerance = 1.0e-10 * scale
    asymmetry = float(np.max(np.abs(result - result.T)))
    if asymmetry > symmetry_tolerance:
        warnings.warn(
            "covariance is not approximately symmetric; symmetrizing it before "
            "computing uncertainties and correlations",
            RuntimeWarning,
            stacklevel=3,
        )
    result = 0.5 * (result + result.T)

    diagonal = np.diag(result)
    diagonal_tolerance = 1.0e-12 * scale
    if np.any(diagonal < -diagonal_tolerance):
        raise ValueError("covariance contains materially negative variances")
    if np.any(diagonal < 0.0):
        warnings.warn(
            "covariance contains tiny negative variances; clipping them to zero",
            RuntimeWarning,
            stacklevel=3,
        )
        result = result.copy()
        indices = np.diag_indices_from(result)
        result[indices] = np.maximum(diagonal, 0.0)

    eigenvalues, eigenvectors = np.linalg.eigh(result)
    eigenvalue_tolerance = 1.0e-10 * scale * max(1, parameter_count)
    if eigenvalues[0] < -eigenvalue_tolerance:
        raise ValueError(
            "covariance is not positive semidefinite; refusing to compute "
            "misleading correlations"
        )
    if eigenvalues[0] < 0.0:
        warnings.warn(
            "covariance has tiny negative eigenvalues; projecting them to zero",
            RuntimeWarning,
            stacklevel=3,
        )
        result = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        result = 0.5 * (result + result.T)

    return result


def _correlation_from_covariance(covariance: np.ndarray, stderr: np.ndarray):
    denominator = np.outer(stderr, stderr)
    correlation = np.full(covariance.shape, np.nan, dtype=float)
    variances = np.diag(covariance)
    valid_variance = np.isfinite(variances) & (variances >= 0.0)
    valid = (
        np.isfinite(covariance)
        & np.isfinite(denominator)
        & (denominator > 0.0)
        & valid_variance[:, None]
        & valid_variance[None, :]
    )
    np.divide(covariance, denominator, out=correlation, where=valid)

    finite = np.isfinite(correlation)
    large_excursion = finite & (np.abs(correlation) > 1.0 + 1.0e-10)
    if np.any(large_excursion):
        raise ValueError(
            "covariance produced correlations materially outside [-1, 1]"
        )
    tiny_excursion = finite & (np.abs(correlation) > 1.0)
    correlation[tiny_excursion] = np.clip(
        correlation[tiny_excursion],
        -1.0,
        1.0,
    )

    diagonal_indices = np.arange(len(stderr))
    correlation[
        diagonal_indices[valid_variance],
        diagonal_indices[valid_variance],
    ] = 1.0
    if not np.allclose(correlation, correlation.T, equal_nan=True):
        raise ValueError("covariance produced a non-symmetric correlation matrix")
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



def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


__all__ = [
    "ParameterDiagnostics",
    "compute_parameter_diagnostics",
    "diagnostics_from_least_squares_result",
]