"""Experimental data preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class BackgroundCorrection:
    """Result of a polynomial edge-background subtraction."""

    x: np.ndarray
    raw: np.ndarray
    background: np.ndarray
    corrected: np.ndarray
    edge_mask: np.ndarray
    coefficients: np.ndarray

    @property
    def normalized(self) -> np.ndarray:
        """Return the background-normalized curve, with edge points near one."""

        return normalize_by_background(self.raw, self.background)


def subtract_edge_polynomial_background(
    x: np.ndarray,
    values: np.ndarray,
    edge_fraction: float = 0.10,
    order: int = 2,
) -> BackgroundCorrection:
    """Subtract a polynomial background fitted to both curve edges.

    `edge_fraction` may be supplied either as a fraction, such as `0.10`,
    or as a percentage, such as `10`, meaning 10 percent at each edge.
    """

    x_array = np.asarray(x, dtype=float)
    value_array = np.asarray(values, dtype=float)
    _validate_curve(x_array, value_array)
    fraction = _normalize_edge_fraction(edge_fraction)
    if order < 0:
        raise ValueError("background polynomial order must be non-negative")

    edge_count = max(1, int(np.ceil(fraction * len(value_array))))
    if 2 * edge_count > len(value_array):
        raise ValueError("edge_fraction selects more than the full curve")
    edge_mask = np.zeros(len(value_array), dtype=bool)
    edge_mask[:edge_count] = True
    edge_mask[-edge_count:] = True
    if np.count_nonzero(edge_mask) <= order:
        raise ValueError("not enough edge points for requested polynomial order")

    coefficients = np.polyfit(x_array[edge_mask], value_array[edge_mask], order)
    background = np.polyval(coefficients, x_array)
    return BackgroundCorrection(
        x=x_array,
        raw=value_array,
        background=background,
        corrected=value_array - background,
        edge_mask=edge_mask,
        coefficients=coefficients,
    )


def normalize_by_mean(values: np.ndarray) -> np.ndarray:
    """Normalize a curve by its arithmetic mean."""

    value_array = np.asarray(values, dtype=float)
    mean = float(np.mean(value_array))
    if not np.isfinite(mean) or mean == 0.0:
        raise ValueError("cannot normalize by a zero or non-finite mean")
    return value_array / mean


def normalize_by_background(values: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Normalize a curve by a fitted background.

    This is equivalent to `1 + (values - background) / background`, so points
    that follow the fitted background stay near one after normalization.
    """

    value_array = np.asarray(values, dtype=float)
    background_array = np.asarray(background, dtype=float)
    if value_array.shape != background_array.shape:
        raise ValueError("values and background must have the same shape")
    if np.any(~np.isfinite(background_array)) or np.any(background_array == 0.0):
        raise ValueError("background must be finite and non-zero")
    return value_array / background_array


def normalize_rocking_curve(
    x: np.ndarray,
    values: np.ndarray,
    mode: Literal["mean", "edge_polynomial"] = "mean",
    offpeak_mask: np.ndarray | None = None,
    edge_fraction: float = 0.10,
    polynomial_order: int = 2,
) -> tuple[np.ndarray, float | np.ndarray]:
    """Normalize a raw rocking curve by a scalar mean or edge polynomial."""

    x_array = np.asarray(x, dtype=float)
    value_array = np.asarray(values, dtype=float)
    if x_array.shape != value_array.shape or x_array.ndim != 1:
        raise ValueError("x and values must be matching one-dimensional arrays")
    if not np.all(np.isfinite(x_array)) or not np.all(np.isfinite(value_array)):
        raise ValueError("x and values must be finite")
    if mode == "edge_polynomial":
        _validate_curve(x_array, value_array)
        correction = subtract_edge_polynomial_background(
            x_array,
            value_array,
            edge_fraction=edge_fraction,
            order=polynomial_order,
        )
        return correction.normalized, correction.background
    if mode != "mean":
        raise ValueError("mode must be 'mean' or 'edge_polynomial'")

    mask = (
        np.ones(x_array.shape, dtype=bool)
        if offpeak_mask is None
        else np.asarray(offpeak_mask, dtype=bool)
    )
    if mask.shape != x_array.shape:
        raise ValueError("offpeak_mask must match rocking-curve angles")
    if not np.any(mask):
        raise ValueError("offpeak_mask must select at least one angle")
    normalization = float(np.mean(value_array[mask]))
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("normalization must be positive and finite")
    return value_array / normalization, normalization


def _normalize_edge_fraction(edge_fraction: float) -> float:
    fraction = float(edge_fraction)
    if fraction > 1.0:
        fraction /= 100.0
    if not 0.0 < fraction <= 0.5:
        raise ValueError("edge_fraction must select between 0 and 50 percent per edge")
    return fraction


def _validate_curve(x: np.ndarray, values: np.ndarray) -> None:
    if x.shape != values.shape:
        raise ValueError("x and values must have the same shape")
    if x.ndim != 1:
        raise ValueError("x and values must be one-dimensional")
    if len(x) < 3:
        raise ValueError("at least three data points are required")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(values)):
        raise ValueError("x and values must be finite")
