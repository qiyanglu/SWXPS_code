"""Physics helpers evaluated on one shared cell-centered layer grid."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from typing import Literal

import numpy as np

from .fields import (
    _field_intensity,
    _graded_delta_beta_at_depth,
    _nominal_boundaries,
    _transfer_matrix_field_amplitudes_sharp_batched,
    FieldProfile,
)
from ..layers import Layer
from ..polarization import Polarization, polarization_weights
from .parratt import energy_to_wavelength, kz_in_layers
from ..stack.slicing import (
    FixedLayerGridPlan,
    LayerGrid,
    LayerSlicingPolicy,
    adaptive_layer_grid,
    fixed_layer_grid,
)
from ..xps.grid import cell_centered_attenuation, integrate_xps_on_grid
from ..xps.intensity import graded_layer_property_at_depth

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

    boundaries = _nominal_boundaries(nominal_layers)
    graded_constants = [
        _graded_delta_beta_at_depth(
            float(center),
            nominal_layers,
            boundaries,
            profile,
            erf_truncation_factor,
            linear_width_factor,
        )
        for center in grid.centers
    ]
    deltas = np.asarray([values[0] for values in graded_constants], dtype=float)
    betas = np.asarray([values[1] for values in graded_constants], dtype=float)
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
    polarization: Polarization = "s",
) -> np.ndarray:
    """Return NumPy transfer-matrix reflectivity for a materialized grid."""

    s_weight, p_weight = polarization_weights(polarization)
    if s_weight and p_weight:
        return (
            s_weight
            * reflectivity_on_grid(
                angles,
                energy_ev,
                effective_layers,
                polarization="s",
            )
            + p_weight
            * reflectivity_on_grid(
                angles,
                energy_ev,
                effective_layers,
                polarization="p",
            )
        )
    pure_polarization = "s" if s_weight else "p"
    downward, upward = _transfer_matrix_field_amplitudes_sharp_batched(
        np.asarray(angles, dtype=float),
        energy_ev,
        effective_layers,
        polarization=pure_polarization,
    )
    return np.abs(upward[0] / downward[0]) ** 2


def field_profiles_on_grid(
    angles: np.ndarray,
    energy_ev: float,
    effective_layers: Sequence[Layer],
    grid: LayerGrid,
    polarization: Polarization = "s",
) -> tuple[FieldProfile, ...]:
    """Evaluate fields at the same centers used by the effective optical cells."""

    angles = np.asarray(angles, dtype=float)
    s_weight, p_weight = polarization_weights(polarization)
    if s_weight and p_weight:
        s_profiles = field_profiles_on_grid(
            angles,
            energy_ev,
            effective_layers,
            grid,
            polarization="s",
        )
        p_profiles = field_profiles_on_grid(
            angles,
            energy_ev,
            effective_layers,
            grid,
            polarization="p",
        )
        return tuple(
            FieldProfile(
                depth=s_profile.depth,
                electric_field=s_profile.electric_field,
                intensity=s_weight * s_profile.intensity + p_weight * p_profile.intensity,
                layer_index=s_profile.layer_index,
            )
            for s_profile, p_profile in zip(s_profiles, p_profiles)
        )
    pure_polarization = "s" if s_weight else "p"
    downward, upward = _transfer_matrix_field_amplitudes_sharp_batched(
        angles,
        energy_ev,
        effective_layers,
        polarization=pure_polarization,
    )
    wavelength = energy_to_wavelength(energy_ev)
    kz = kz_in_layers(angles, wavelength, [layer.n for layer in effective_layers])
    sampled_layers = grid.effective_layer_index
    local_depth = 0.5 * grid.widths
    phase = kz[sampled_layers] * local_depth[:, np.newaxis]
    down_field = downward[sampled_layers] * np.exp(1j * phase)
    up_field = upward[sampled_layers] * np.exp(-1j * phase)
    electric_field = down_field + up_field
    k0 = 2.0 * np.pi / wavelength
    n_sampled = np.asarray([layer.n for layer in effective_layers], dtype=complex)[
        sampled_layers,
        np.newaxis,
    ]
    intensity = _field_intensity(
        down_field,
        up_field,
        kz[sampled_layers],
        n_sampled,
        k0,
        pure_polarization,
    )
    return tuple(
        FieldProfile(
            depth=grid.centers,
            electric_field=electric_field[:, angle_index],
            intensity=intensity[:, angle_index],
            layer_index=grid.effective_layer_index,
        )
        for angle_index in range(angles.size)
    )


_UNIFIED_SIMULATION_EXPORTS = {
    "simulate_reflectivity_unified",
    "simulate_rocking_curves_unified",
}


def __getattr__(name: str):
    if name not in _UNIFIED_SIMULATION_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .. import simulation_unified

    value = getattr(simulation_unified, name)
    globals()[name] = value
    return value


__all__ = [
    "cell_centered_attenuation",
    "effective_layers_from_grid",
    "field_profiles_on_grid",
    "integrate_xps_on_grid",
    "materialize_layer_grid",
    "reflectivity_on_grid",
    "simulate_reflectivity_unified",
    "simulate_rocking_curves_unified",
]
