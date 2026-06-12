"""Standing-wave XPS rocking-curve calculations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import erf, sqrt

import numpy as np

from .fields import FieldProfile, transfer_matrix_electric_field_profile
from .layers import Layer


@dataclass(frozen=True)
class RockingCurve:
    """A normalized XPS rocking curve."""

    angle: np.ndarray
    intensity: np.ndarray
    raw_intensity: np.ndarray
    normalization: float


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
    return float(np.trapz(integrand, profile.depth))


def normalized_rocking_curve(
    angles: np.ndarray,
    energy_ev: float,
    layers: Sequence[Layer],
    concentration_by_layer: Sequence[float],
    imfp_by_layer: Sequence[float],
    emission_angle_deg: float = 0.0,
    field_step: float = 1.0,
    roughness_step: float = 1.0,
    offpeak_mask: np.ndarray | None = None,
) -> RockingCurve:
    """Return a normalized SW-XPS rocking curve with constant cross section."""

    angles = np.asarray(angles, dtype=float)
    concentration_by_layer = np.asarray(concentration_by_layer, dtype=float)
    imfp_by_layer = np.asarray(imfp_by_layer, dtype=float)

    if len(concentration_by_layer) != len(layers):
        raise ValueError("concentration_by_layer must have one value per layer")
    if len(imfp_by_layer) != len(layers):
        raise ValueError("imfp_by_layer must have one value per layer")

    raw = []
    for angle in angles:
        profile = transfer_matrix_electric_field_profile(
            angle_deg=float(angle),
            energy_ev=energy_ev,
            layers=layers,
            step=field_step,
            roughness_step=roughness_step,
        )
        concentration = graded_layer_property_at_depth(
            layers,
            concentration_by_layer,
            profile.depth,
        )
        attenuation_coefficient = graded_layer_property_at_depth(
            layers,
            1.0 / imfp_by_layer,
            profile.depth,
        )
        attenuation_length = 1.0 / attenuation_coefficient
        raw.append(
            integrate_xps_intensity(
                profile,
                concentration,
                attenuation_length,
                emission_angle_deg=emission_angle_deg,
            )
        )

    raw_intensity = np.asarray(raw, dtype=float)
    if offpeak_mask is None:
        offpeak_mask = np.ones(angles.shape, dtype=bool)
    else:
        offpeak_mask = np.asarray(offpeak_mask, dtype=bool)
    if offpeak_mask.shape != angles.shape:
        raise ValueError("offpeak_mask must match angles shape")
    if not np.any(offpeak_mask):
        raise ValueError("offpeak_mask must select at least one angle")

    normalization = float(np.mean(raw_intensity[offpeak_mask]))
    if normalization <= 0:
        raise ValueError("normalization must be positive")

    return RockingCurve(
        angle=angles,
        intensity=raw_intensity / normalization,
        raw_intensity=raw_intensity,
        normalization=normalization,
    )


def nominal_layer_index_at_depth(layers: Sequence[Layer], depth: np.ndarray) -> np.ndarray:
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
    width_factor: float = 4.0,
) -> np.ndarray:
    """Return a layer property sampled with error-function roughness grading."""

    values_by_layer = np.asarray(values_by_layer, dtype=float)
    depth = np.asarray(depth, dtype=float)
    if len(values_by_layer) != len(layers):
        raise ValueError("values_by_layer must have one value per layer")
    if width_factor <= 0:
        raise ValueError("width_factor must be positive")

    nominal_index = nominal_layer_index_at_depth(layers, depth)
    values = values_by_layer[nominal_index].copy()
    boundaries = _nominal_boundaries(layers)

    for interface_index, boundary in enumerate(boundaries):
        sigma = layers[interface_index + 1].roughness
        if sigma <= 0:
            continue
        mask = np.abs(depth - boundary) <= width_factor * sigma
        if not np.any(mask):
            continue
        fraction = 0.5 * (
            1.0
            + np.array(
                [
                    erf((z - boundary) / (sqrt(2.0) * sigma))
                    for z in depth[mask]
                ]
            )
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


def _cumulative_trapezoid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    result = np.zeros_like(x, dtype=float)
    if len(x) < 2:
        return result
    dx = np.diff(x)
    areas = 0.5 * (y[:-1] + y[1:]) * dx
    result[1:] = np.cumsum(areas)
    return result
