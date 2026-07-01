"""Scaled-Jacobian identifiability diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IdentifiabilitySettings:
    weak_modes: int = 5
    active_bound_tol: float = 0.02
    low_sensitivity_threshold: float = 0.05
    high_uncertainty_threshold: float = 0.50
    high_correlation_threshold: float = 0.90
    high_weak_participation_threshold: float = 0.50

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object] | None) -> "IdentifiabilitySettings":
        values = {} if raw is None else dict(raw)
        values.pop("enabled", None)
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError("unknown identifiability option(s): " + ", ".join(unknown))
        defaults = cls()
        return cls(
            weak_modes=int(values.get("weak_modes", defaults.weak_modes)),
            active_bound_tol=float(values.get("active_bound_tol", defaults.active_bound_tol)),
            low_sensitivity_threshold=float(
                values.get("low_sensitivity_threshold", defaults.low_sensitivity_threshold)
            ),
            high_uncertainty_threshold=float(
                values.get("high_uncertainty_threshold", defaults.high_uncertainty_threshold)
            ),
            high_correlation_threshold=float(
                values.get("high_correlation_threshold", defaults.high_correlation_threshold)
            ),
            high_weak_participation_threshold=float(
                values.get(
                    "high_weak_participation_threshold",
                    defaults.high_weak_participation_threshold,
                )
            ),
        )


@dataclass(frozen=True)
class IdentifiabilityParameter:
    name: str
    initial: float
    lower: float
    upper: float
    best: float

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def scaled_position(self) -> float:
        return (self.best - self.lower) / self.width


@dataclass(frozen=True)
class IdentifiabilityAnalysis:
    parameters: tuple[IdentifiabilityParameter, ...]
    residuals: np.ndarray
    scaled_jacobian: np.ndarray
    scaled_sensitivity_norm: np.ndarray
    scaled_sensitivity_rms: np.ndarray
    relative_sensitivity: np.ndarray
    scaled_gradient: np.ndarray
    singular_values: np.ndarray
    right_singular_vectors: np.ndarray
    condition_number: float
    weak_mode_count: int
    weak_mode_participation: np.ndarray
    stderr: np.ndarray
    scaled_stderr: np.ndarray
    correlation: np.ndarray | None
    max_abs_correlation: np.ndarray
    active_bounds: tuple[str, ...]
    suggestions: tuple[str, ...]
    dataset_sensitivity_rows: tuple[tuple[object, ...], ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)


def analyze_identifiability(
    parameters: Sequence[IdentifiabilityParameter],
    residuals,
    jacobian,
    *,
    covariance=None,
    correlation=None,
    dataset_labels: Sequence[str] | None = None,
    settings: IdentifiabilitySettings | None = None,
) -> IdentifiabilityAnalysis:
    """Analyze local fit identifiability from a residual vector and Jacobian."""

    settings = IdentifiabilitySettings() if settings is None else settings
    parameter_tuple = tuple(parameters)
    parameter_count = len(parameter_tuple)
    residual_array = np.asarray(residuals, dtype=float)
    jacobian_array = np.asarray(jacobian, dtype=float)
    if residual_array.ndim != 1:
        raise ValueError("residuals must be a one-dimensional array")
    if jacobian_array.ndim != 2 or jacobian_array.shape[1] != parameter_count:
        raise ValueError("jacobian must have shape (N, P)")
    if jacobian_array.shape[0] != residual_array.size:
        raise ValueError("jacobian row count must match residual count")
    if not np.all(np.isfinite(residual_array)) or not np.all(np.isfinite(jacobian_array)):
        raise ValueError("residuals and jacobian must be finite")
    widths = np.asarray([parameter.width for parameter in parameter_tuple], dtype=float)
    if np.any(~np.isfinite(widths)) or np.any(widths <= 0.0):
        raise ValueError("all parameter ranges must be finite and positive")

    scaled_jacobian = jacobian_array * widths[np.newaxis, :]
    sensitivity = np.linalg.norm(scaled_jacobian, axis=0)
    sensitivity_rms = sensitivity / np.sqrt(max(residual_array.size, 1))
    max_sensitivity = float(np.max(sensitivity)) if sensitivity.size else 1.0
    relative_sensitivity = sensitivity / max(max_sensitivity, np.finfo(float).tiny)
    scaled_gradient = scaled_jacobian.T @ residual_array

    singular_values, vt = _svd(scaled_jacobian)
    weak_mode_count = min(max(int(settings.weak_modes), 0), vt.shape[0])
    weak_participation = _weak_mode_participation(vt, weak_mode_count, parameter_count)

    stderr = _stderr(covariance, parameter_count)
    scaled_stderr = np.divide(stderr, widths, out=np.full(parameter_count, np.nan), where=widths > 0)
    correlation_array = _correlation(correlation, covariance, parameter_count)
    max_abs_corr = _max_abs_correlations(correlation_array, parameter_count)
    active_bounds = tuple(
        _classify_bound(parameter.scaled_position, settings.active_bound_tol)
        for parameter in parameter_tuple
    )
    suggestions = tuple(
        _suggest_actions(
            relative_sensitivity=relative_sensitivity,
            scaled_stderr=scaled_stderr,
            max_abs_correlation=max_abs_corr,
            weak_participation=weak_participation,
            active_bounds=active_bounds,
            settings=settings,
        )
    )
    dataset_rows = _dataset_sensitivity_rows(
        dataset_labels,
        scaled_jacobian,
        sensitivity,
        tuple(parameter.name for parameter in parameter_tuple),
    )

    return IdentifiabilityAnalysis(
        parameters=parameter_tuple,
        residuals=_readonly(residual_array),
        scaled_jacobian=_readonly(scaled_jacobian),
        scaled_sensitivity_norm=_readonly(sensitivity),
        scaled_sensitivity_rms=_readonly(sensitivity_rms),
        relative_sensitivity=_readonly(relative_sensitivity),
        scaled_gradient=_readonly(scaled_gradient),
        singular_values=_readonly(singular_values),
        right_singular_vectors=_readonly(vt),
        condition_number=_condition_number(singular_values),
        weak_mode_count=weak_mode_count,
        weak_mode_participation=_readonly(weak_participation),
        stderr=_readonly(stderr),
        scaled_stderr=_readonly(scaled_stderr),
        correlation=None if correlation_array is None else _readonly(correlation_array),
        max_abs_correlation=_readonly(max_abs_corr),
        active_bounds=active_bounds,
        suggestions=suggestions,
        dataset_sensitivity_rows=tuple(dataset_rows),
    )


def _svd(scaled_jacobian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if scaled_jacobian.size == 0:
        return np.array([], dtype=float), np.empty((0, scaled_jacobian.shape[1]), dtype=float)
    _u, singular_values, vt = np.linalg.svd(scaled_jacobian, full_matrices=False)
    return singular_values, vt


def _condition_number(singular_values: np.ndarray) -> float:
    if singular_values.size == 0:
        return float("nan")
    if singular_values[-1] <= 0.0:
        return float("inf")
    return float(singular_values[0] / singular_values[-1])


def _weak_mode_participation(
    vt: np.ndarray,
    weak_mode_count: int,
    parameter_count: int,
) -> np.ndarray:
    if weak_mode_count <= 0:
        return np.zeros(parameter_count, dtype=float)
    weak_vt = vt[-weak_mode_count:, :]
    return np.sqrt(np.sum(weak_vt**2, axis=0))


def _stderr(covariance, parameter_count: int) -> np.ndarray:
    if covariance is None:
        return np.full(parameter_count, np.nan)
    covariance_array = np.asarray(covariance, dtype=float)
    if covariance_array.shape != (parameter_count, parameter_count):
        return np.full(parameter_count, np.nan)
    return np.sqrt(np.maximum(np.diag(covariance_array), 0.0))


def _correlation(correlation, covariance, parameter_count: int) -> np.ndarray | None:
    if correlation is not None:
        correlation_array = np.asarray(correlation, dtype=float)
        return correlation_array if correlation_array.shape == (parameter_count, parameter_count) else None
    if covariance is None:
        return None
    covariance_array = np.asarray(covariance, dtype=float)
    if covariance_array.shape != (parameter_count, parameter_count):
        return None
    sigma = np.sqrt(np.maximum(np.diag(covariance_array), 0.0))
    denominator = np.outer(sigma, sigma)
    return np.divide(
        covariance_array,
        denominator,
        out=np.full_like(covariance_array, np.nan),
        where=denominator > 0,
    )


def _max_abs_correlations(correlation: np.ndarray | None, parameter_count: int) -> np.ndarray:
    if correlation is None:
        return np.full(parameter_count, np.nan)
    result = np.full(parameter_count, np.nan)
    for index in range(parameter_count):
        values = np.delete(np.abs(correlation[index]), index)
        finite = values[np.isfinite(values)]
        result[index] = float(np.max(finite)) if finite.size else np.nan
    return result


def _classify_bound(scaled_position: float, active_tol: float) -> str:
    if scaled_position <= active_tol:
        return "near_lower"
    if scaled_position >= 1.0 - active_tol:
        return "near_upper"
    return ""


def _suggest_actions(
    *,
    relative_sensitivity: np.ndarray,
    scaled_stderr: np.ndarray,
    max_abs_correlation: np.ndarray,
    weak_participation: np.ndarray,
    active_bounds: Sequence[str],
    settings: IdentifiabilitySettings,
) -> list[str]:
    suggestions = []
    for index in range(relative_sensitivity.size):
        flags = []
        if relative_sensitivity[index] < settings.low_sensitivity_threshold:
            flags.append("low_sensitivity")
        if (
            np.isfinite(scaled_stderr[index])
            and scaled_stderr[index] > settings.high_uncertainty_threshold
        ):
            flags.append("high_uncertainty")
        if (
            np.isfinite(max_abs_correlation[index])
            and max_abs_correlation[index] > settings.high_correlation_threshold
        ):
            flags.append("high_correlation")
        if weak_participation[index] > settings.high_weak_participation_threshold:
            flags.append("weak_svd_mode")
        if active_bounds[index]:
            flags.append(active_bounds[index])

        if "low_sensitivity" in flags and "weak_svd_mode" in flags:
            action = "fix_or_profile_candidate"
        elif "high_correlation" in flags and "weak_svd_mode" in flags:
            action = "tie_or_reparameterize_candidate"
        elif active_bounds[index]:
            action = "review_bound_or_model"
        elif "high_uncertainty" in flags or "weak_svd_mode" in flags:
            action = "needs_profile_check"
        else:
            action = "keep_for_now"
        suggestions.append("; ".join([action, *flags]))
    return suggestions


def _dataset_sensitivity_rows(
    dataset_labels: Sequence[str] | None,
    scaled_jacobian: np.ndarray,
    total_sensitivity: np.ndarray,
    names: tuple[str, ...],
) -> list[tuple[object, ...]]:
    header = (
        "dataset",
        "residual_count",
        "parameter",
        "scaled_sensitivity_norm",
        "fraction_of_parameter_sensitivity",
    )
    if dataset_labels is None:
        return [header, ("note", "", "dataset labels unavailable", "", "")]
    labels = tuple(str(label) for label in dataset_labels)
    if len(labels) != scaled_jacobian.shape[0]:
        return [header, ("note", "", "dataset label count does not match residual vector", "", "")]
    rows = [header]
    for dataset in dict.fromkeys(labels):
        mask = np.asarray([label == dataset for label in labels], dtype=bool)
        block_sensitivity = np.linalg.norm(scaled_jacobian[mask, :], axis=0)
        fraction = np.divide(
            block_sensitivity,
            total_sensitivity,
            out=np.zeros_like(block_sensitivity),
            where=total_sensitivity > 0,
        )
        for index, name in enumerate(names):
            rows.append((dataset, int(mask.sum()), name, block_sensitivity[index], fraction[index]))
    return rows


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


__all__ = [
    "IdentifiabilityAnalysis",
    "IdentifiabilityParameter",
    "IdentifiabilitySettings",
    "analyze_identifiability",
]
