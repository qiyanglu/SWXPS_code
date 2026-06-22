"""Physics helpers evaluated on one shared cell-centered layer grid."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from typing import Literal

import numpy as np

from .fields import (
    _transfer_matrix_field_amplitudes_sharp_batched,
    FieldProfile,
)
from .layers import Layer
from .reflectivity import energy_to_wavelength, kz_in_layers
from .slicing import (
    FixedLayerGridPlan,
    LayerGrid,
    LayerSlicingPolicy,
    adaptive_layer_grid,
    fixed_layer_grid,
)
from .xps import graded_layer_property_at_depth

SlicingSpecification = LayerSlicingPolicy | FixedLayerGridPlan


def materialize_layer_grid(
    layers: Sequence[Layer],
    slicing: SlicingSpecification,
) -> LayerGrid:
    """Return an adaptive or fixed-capacity grid for the supplied stack."""

    if isinstance(slicing, LayerSlicingPolicy):
        return adaptive_layer_grid(layers, slicing)
    if isinstance(slicing, FixedLayerGridPlan):
        return fixed_layer_grid(layers, slicing)
    raise TypeError("slicing must be a LayerSlicingPolicy or FixedLayerGridPlan")


def effective_layers_from_grid(
    nominal_layers: Sequence[Layer],
    grid: LayerGrid,
    profile: Literal["erf", "linear"] = "erf",
    erf_truncation_factor: float = 4.0,
    linear_width_factor: float = sqrt(3.0),
) -> list[Layer]:
    """Return one graded sharp optical layer per grid cell."""

    deltas = graded_layer_property_at_depth(
        nominal_layers,
        [layer.delta for layer in nominal_layers],
        grid.centers,
        profile=profile,
        erf_truncation_factor=erf_truncation_factor,
        linear_width_factor=linear_width_factor,
    )
    betas = graded_layer_property_at_depth(
        nominal_layers,
        [layer.beta for layer in nominal_layers],
        grid.centers,
        profile=profile,
        erf_truncation_factor=erf_truncation_factor,
        linear_width_factor=linear_width_factor,
    )
    cells = [
        Layer(float(width), float(delta), float(beta), roughness=0.0)
        for width, delta, beta in zip(grid.widths, deltas, betas)
    ]
    first = nominal_layers[0]
    last = nominal_layers[-1]
    return [
        Layer(0.0, first.delta, first.beta, roughness=0.0),
        *cells,
        Layer(0.0, last.delta, last.beta, roughness=0.0),
    ]


def reflectivity_on_grid(
    angles: np.ndarray,
    energy_ev: float,
    effective_layers: Sequence[Layer],
) -> np.ndarray:
    """Return NumPy transfer-matrix reflectivity for a materialized grid."""

    downward, upward = _transfer_matrix_field_amplitudes_sharp_batched(
        np.asarray(angles, dtype=float),
        energy_ev,
        effective_layers,
    )
    return np.abs(upward[0] / downward[0]) ** 2


def field_profiles_on_grid(
    angles: np.ndarray,
    energy_ev: float,
    effective_layers: Sequence[Layer],
    grid: LayerGrid,
) -> tuple[FieldProfile, ...]:
    """Evaluate fields at the same centers used by the effective optical cells."""

    angles = np.asarray(angles, dtype=float)
    downward, upward = _transfer_matrix_field_amplitudes_sharp_batched(
        angles,
        energy_ev,
        effective_layers,
    )
    wavelength = energy_to_wavelength(energy_ev)
    kz = kz_in_layers(angles, wavelength, [layer.n for layer in effective_layers])
    sampled_layers = grid.effective_layer_index
    local_depth = 0.5 * grid.widths
    phase = kz[sampled_layers] * local_depth[:, np.newaxis]
    electric_field = (
        downward[sampled_layers] * np.exp(1j * phase)
        + upward[sampled_layers] * np.exp(-1j * phase)
    )
    intensity = np.abs(electric_field) ** 2
    return tuple(
        FieldProfile(
            depth=grid.centers,
            electric_field=electric_field[:, angle_index],
            intensity=intensity[:, angle_index],
            layer_index=grid.effective_layer_index,
        )
        for angle_index in range(angles.size)
    )


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
