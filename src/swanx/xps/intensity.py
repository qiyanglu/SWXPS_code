"""XPS intensity integration and roughness-sampled layer properties."""

from __future__ import annotations

from collections.abc import Sequence
from math import erf, sqrt
from typing import Literal

import numpy as np

from ..layers import Layer
from ..optics.fields import FieldProfile
from .attenuation import attenuation_factor


def integrate_xps_intensity(
    profile: FieldProfile,
    concentration: np.ndarray,
    attenuation_length: np.ndarray,
    emission_angle_deg: float = 0.0,
) -> float:
    """Integrate concentration, field intensity, and attenuation over depth."""

    concentration = np.asarray(concentration, dtype=float)
    attenuation_length = np.asarray(attenuation_length, dtype=float)
    if concentration.shape != profile.depth.shape:
        raise ValueError("concentration must match profile.depth shape")
    if attenuation_length.shape != profile.depth.shape:
        raise ValueError("attenuation_length must match profile.depth shape")

    attenuation = attenuation_factor(
        profile.depth,
        attenuation_length,
        emission_angle_deg=emission_angle_deg,
    )
    integrand = concentration * profile.intensity * attenuation
    return float(_trapezoid(integrand, profile.depth))


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(y, x))


def nominal_layer_index_at_depth(
    layers: Sequence[Layer],
    depth: np.ndarray,
) -> np.ndarray:
    """Map sampled depths to nominal layer indices."""

    depth = np.asarray(depth, dtype=float)
    finite_thicknesses = [layer.thickness for layer in layers[1:-1]]
    boundaries = np.cumsum(finite_thicknesses)
    index = np.searchsorted(boundaries, depth, side="right") + 1
    return np.clip(index, 1, len(layers) - 2)


def graded_layer_property_at_depth(
    layers: Sequence[Layer],
    values_by_layer: Sequence[float],
    depth: np.ndarray,
    profile: Literal["erf", "linear"] = "erf",
    erf_truncation_factor: float = 4.0,
    linear_width_factor: float = sqrt(3.0),
) -> np.ndarray:
    """Return a layer property sampled with selected roughness grading."""

    values_by_layer = np.asarray(values_by_layer, dtype=float)
    depth = np.asarray(depth, dtype=float)
    if len(values_by_layer) != len(layers):
        raise ValueError("values_by_layer must have one value per layer")
    if erf_truncation_factor <= 0:
        raise ValueError("erf_truncation_factor must be positive")
    if linear_width_factor <= 0:
        raise ValueError("linear_width_factor must be positive")
    if profile not in {"erf", "linear"}:
        raise ValueError("profile must be 'erf' or 'linear'")

    nominal_index = nominal_layer_index_at_depth(layers, depth)
    values = values_by_layer[nominal_index].copy()
    boundaries = _nominal_boundaries(layers)

    for interface_index, boundary in enumerate(boundaries):
        sigma = layers[interface_index + 1].roughness
        if sigma <= 0:
            continue
        active_width_factor = (
            erf_truncation_factor if profile == "erf" else linear_width_factor
        )
        mask = np.abs(depth - boundary) <= active_width_factor * sigma
        mask &= (nominal_index == interface_index) | (
            nominal_index == interface_index + 1
        )
        if not np.any(mask):
            continue
        fraction = _roughness_fraction(
            depth[mask] - boundary,
            sigma,
            profile,
            linear_width_factor,
        )
        above = values_by_layer[interface_index]
        below = values_by_layer[interface_index + 1]
        values[mask] = (1.0 - fraction) * above + fraction * below

    return values


def _nominal_boundaries(layers: Sequence[Layer]) -> np.ndarray:
    boundaries = [0.0]
    current_depth = 0.0
    for layer in layers[1:-1]:
        current_depth += layer.thickness
        boundaries.append(current_depth)
    return np.asarray(boundaries, dtype=float)


def _roughness_fraction(
    distance: np.ndarray,
    sigma: float,
    profile: Literal["erf", "linear"],
    linear_width_factor: float,
) -> np.ndarray:
    if profile == "erf":
        return 0.5 * (
            1.0
            + np.array(
                [
                    erf(value / (sqrt(2.0) * sigma))
                    for value in np.asarray(distance, dtype=float)
                ]
            )
        )

    half_width = linear_width_factor * sigma
    return np.clip(
        0.5 + 0.5 * np.asarray(distance, dtype=float) / half_width,
        0.0,
        1.0,
    )


__all__ = [
    "graded_layer_property_at_depth",
    "integrate_xps_intensity",
    "nominal_layer_index_at_depth",
]

