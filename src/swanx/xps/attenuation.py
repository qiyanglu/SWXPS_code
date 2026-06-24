"""Electron attenuation for standing-wave XPS calculations."""

from __future__ import annotations

import numpy as np


def attenuation_factor(
    depth: np.ndarray,
    attenuation_length: np.ndarray,
    emission_angle_deg: float = 0.0,
) -> np.ndarray:
    """Return electron attenuation from each depth to the surface."""

    depth = np.asarray(depth, dtype=float)
    attenuation_length = np.asarray(attenuation_length, dtype=float)
    if depth.shape != attenuation_length.shape:
        raise ValueError("depth and attenuation_length must have the same shape")
    if np.any(attenuation_length <= 0):
        raise ValueError("attenuation_length values must be positive")

    cos_alpha = np.cos(np.deg2rad(emission_angle_deg))
    if cos_alpha <= 0:
        raise ValueError("emission_angle_deg must be less than 90 degrees")

    attenuation_coefficient = 1.0 / (attenuation_length * cos_alpha)
    optical_depth = _cumulative_trapezoid(depth, attenuation_coefficient)
    return np.exp(-optical_depth)


def _cumulative_trapezoid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    result = np.zeros_like(x, dtype=float)
    if len(x) < 2:
        return result
    dx = np.diff(x)
    areas = 0.5 * (y[:-1] + y[1:]) * dx
    result[1:] = np.cumsum(areas)
    return result


__all__ = ["attenuation_factor"]
