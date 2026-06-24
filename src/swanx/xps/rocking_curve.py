"""Normalized standing-wave XPS rocking-curve calculations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from typing import Literal

import numpy as np

from ..layers import Layer
from ..optics.fields import transfer_matrix_electric_field_profile
from .intensity import graded_layer_property_at_depth, integrate_xps_intensity


@dataclass(frozen=True)
class RockingCurve:
    """A normalized XPS rocking curve."""

    angle: np.ndarray
    intensity: np.ndarray
    raw_intensity: np.ndarray
    normalization: float | np.ndarray


def normalized_rocking_curve(
    angles: np.ndarray,
    energy_ev: float,
    layers: Sequence[Layer],
    concentration_by_layer: Sequence[float],
    imfp_by_layer: Sequence[float],
    emission_angle_deg: float = 0.0,
    field_step: float = 1.0,
    roughness_step: float | Sequence[float] = 1.0,
    roughness_profile: Literal["erf", "linear"] = "erf",
    erf_truncation_factor: float = 4.0,
    linear_width_factor: float = sqrt(3.0),
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
            roughness_profile=roughness_profile,
            erf_truncation_factor=erf_truncation_factor,
            linear_width_factor=linear_width_factor,
        )
        concentration = graded_layer_property_at_depth(
            layers,
            concentration_by_layer,
            profile.depth,
            profile=roughness_profile,
            erf_truncation_factor=erf_truncation_factor,
            linear_width_factor=linear_width_factor,
        )
        attenuation_coefficient = graded_layer_property_at_depth(
            layers,
            1.0 / imfp_by_layer,
            profile.depth,
            profile=roughness_profile,
            erf_truncation_factor=erf_truncation_factor,
            linear_width_factor=linear_width_factor,
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


__all__ = ["RockingCurve", "normalized_rocking_curve"]
