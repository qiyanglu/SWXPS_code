"""Cell-centered attenuation and XPS integration on a unified layer grid."""

from __future__ import annotations

import numpy as np

from ..stack.slicing import LayerGrid


def cell_centered_attenuation(
    widths: np.ndarray,
    attenuation_length: np.ndarray,
    emission_angle_deg: float = 0.0,
) -> np.ndarray:
    """Return attenuation from each cell center to the sample surface."""

    widths = np.asarray(widths, dtype=float)
    attenuation_length = np.asarray(attenuation_length, dtype=float)
    if widths.shape != attenuation_length.shape:
        raise ValueError("widths and attenuation_length must have the same shape")
    if np.any(widths <= 0) or np.any(attenuation_length <= 0):
        raise ValueError("widths and attenuation_length values must be positive")
    cos_alpha = np.cos(np.deg2rad(emission_angle_deg))
    if cos_alpha <= 0:
        raise ValueError("emission_angle_deg must be less than 90 degrees")

    cell_optical_depth = widths / (attenuation_length * cos_alpha)
    optical_depth_to_center = (
        np.cumsum(cell_optical_depth) - 0.5 * cell_optical_depth
    )
    return np.exp(-optical_depth_to_center)


def integrate_xps_on_grid(
    field_intensity: np.ndarray,
    grid: LayerGrid,
    concentration: np.ndarray,
    attenuation_length: np.ndarray,
    emission_angle_deg: float = 0.0,
) -> np.ndarray:
    """Integrate one or many angle-dependent fields with midpoint quadrature."""

    field_intensity = np.asarray(field_intensity, dtype=float)
    concentration = np.asarray(concentration, dtype=float)
    attenuation_length = np.asarray(attenuation_length, dtype=float)
    if concentration.shape != grid.centers.shape:
        raise ValueError("concentration must contain one value per grid cell")
    if attenuation_length.shape != grid.centers.shape:
        raise ValueError("attenuation_length must contain one value per grid cell")
    if field_intensity.shape[0] != len(grid.centers):
        raise ValueError("field_intensity first dimension must match grid cells")

    attenuation = cell_centered_attenuation(
        grid.widths,
        attenuation_length,
        emission_angle_deg=emission_angle_deg,
    )
    weights = concentration * attenuation * grid.widths
    if field_intensity.ndim == 1:
        return np.asarray(np.sum(field_intensity * weights))
    if field_intensity.ndim != 2:
        raise ValueError("field_intensity must be one- or two-dimensional")
    return np.sum(field_intensity * weights[:, np.newaxis], axis=0)


__all__ = ["cell_centered_attenuation", "integrate_xps_on_grid"]
