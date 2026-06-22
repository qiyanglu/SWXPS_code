"""High-level NumPy/JAX simulations on one unified layer grid."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .preprocessing import normalize_rocking_curve
from .simulation import (
    CoreLevelResult,
    ReflectivityRequest,
    ReflectivityResult,
    RockingCurveRequest,
    RockingCurveResult,
    _apply_emitting_layer_filter,
    _values_by_material,
)
from .unified_grid import (
    effective_layers_from_grid,
    field_profiles_on_grid,
    integrate_xps_on_grid,
    materialize_layer_grid,
    reflectivity_on_grid,
)
from .xps import RockingCurve, graded_layer_property_at_depth


def simulate_reflectivity_unified(
    request: ReflectivityRequest,
    backend: Literal["numpy", "jax"] = "numpy",
) -> ReflectivityResult:
    """Simulate reflectivity using the request's shared slicing specification."""

    if request.slicing is None:
        raise ValueError("unified simulation requires request.slicing")
    angles = np.asarray(request.angles, dtype=float)
    calculation_angle = angles + request.angle_offset
    nominal_layers = request.stack.optical_layers
    grid = materialize_layer_grid(nominal_layers, request.slicing)
    effective_layers = effective_layers_from_grid(
        nominal_layers,
        grid,
        profile=request.roughness_profile,
        erf_truncation_factor=request.erf_truncation_factor,
        linear_width_factor=request.linear_width_factor,
    )
    if backend == "numpy":
        reflectivity = reflectivity_on_grid(
            calculation_angle,
            request.energy_ev,
            effective_layers,
        )
    elif backend == "jax":
        from .reflectivity_jax import (
            jitted_transfer_matrix_reflectivity,
            layer_arrays_from_layers,
        )

        thicknesses, deltas, betas, _ = layer_arrays_from_layers(effective_layers)
        reflectivity = np.asarray(
            jitted_transfer_matrix_reflectivity(
                calculation_angle,
                request.energy_ev,
                thicknesses,
                deltas,
                betas,
            ),
            dtype=float,
        )
    else:
        raise ValueError("backend must be 'numpy' or 'jax'")
    return ReflectivityResult(
        angle=angles,
        calculation_angle=calculation_angle,
        reflectivity=np.asarray(reflectivity, dtype=float),
    )


def simulate_rocking_curves_unified(
    request: RockingCurveRequest,
    backend: Literal["numpy", "jax"] = "numpy",
) -> RockingCurveResult:
    """Simulate normalized RCs with one grid for optics, fields, and XPS."""

    if request.slicing is None:
        raise ValueError("unified simulation requires request.slicing")
    angles = np.asarray(request.angles, dtype=float)
    calculation_angle = angles + request.angle_offset
    nominal_layers = request.stack.optical_layers
    grid = materialize_layer_grid(nominal_layers, request.slicing)
    effective_layers = effective_layers_from_grid(
        nominal_layers,
        grid,
        profile=request.roughness_profile,
        erf_truncation_factor=request.erf_truncation_factor,
        linear_width_factor=request.linear_width_factor,
    )

    if backend == "numpy":
        profiles = field_profiles_on_grid(
            calculation_angle,
            request.photon_energy_ev,
            effective_layers,
            grid,
        )
        field_intensity = np.column_stack(
            [profile.intensity for profile in profiles]
        )
    elif backend == "jax":
        from .reflectivity_jax import (
            jitted_transfer_matrix_field_intensity,
            layer_arrays_from_layers,
        )

        thicknesses, deltas, betas, _ = layer_arrays_from_layers(effective_layers)
        field_intensity = np.asarray(
            jitted_transfer_matrix_field_intensity(
                calculation_angle,
                request.photon_energy_ev,
                thicknesses,
                deltas,
                betas,
                grid.centers,
                grid.effective_layer_index,
            ),
            dtype=float,
        )
    else:
        raise ValueError("backend must be 'numpy' or 'jax'")

    results = tuple(
        _simulate_core_on_grid(request, core_level, grid, field_intensity)
        for core_level in request.core_levels
    )
    return RockingCurveResult(
        angle=angles,
        calculation_angle=calculation_angle,
        core_levels=results,
    )


def _simulate_core_on_grid(request, core_level, grid, field_intensity):
    kinetic_energy_ev = request.photon_energy_ev - core_level.binding_energy_ev
    if kinetic_energy_ev <= 0:
        raise ValueError("core-level kinetic energy must be positive")

    materials = request.stack.materials
    concentration_by_layer = _values_by_material(
        materials,
        core_level.concentration_by_material,
        default=0.0,
    )
    if core_level.emitting_layer_indices is not None:
        concentration_by_layer = _apply_emitting_layer_filter(
            concentration_by_layer,
            core_level.emitting_layer_indices,
        )
    imfp_by_layer = _values_by_material(
        materials,
        core_level.imfp_by_material,
        default=None,
    )
    nominal_layers = request.stack.optical_layers
    concentration = graded_layer_property_at_depth(
        nominal_layers,
        concentration_by_layer,
        grid.centers,
        profile=request.roughness_profile,
        erf_truncation_factor=request.erf_truncation_factor,
        linear_width_factor=request.linear_width_factor,
    )
    attenuation_coefficient = graded_layer_property_at_depth(
        nominal_layers,
        1.0 / np.asarray(imfp_by_layer, dtype=float),
        grid.centers,
        profile=request.roughness_profile,
        erf_truncation_factor=request.erf_truncation_factor,
        linear_width_factor=request.linear_width_factor,
    )
    raw = integrate_xps_on_grid(
        field_intensity,
        grid,
        concentration,
        1.0 / attenuation_coefficient,
        emission_angle_deg=core_level.emission_angle_deg,
    )
    normalized, normalization = normalize_rocking_curve(
        np.asarray(request.angles, dtype=float),
        raw,
        mode=request.normalization_mode,
        offpeak_mask=request.offpeak_mask,
        edge_fraction=request.normalization_edge_fraction,
        polynomial_order=request.normalization_polynomial_order,
    )
    return CoreLevelResult(
        name=core_level.name,
        binding_energy_ev=core_level.binding_energy_ev,
        kinetic_energy_ev=kinetic_energy_ev,
        curve=RockingCurve(
            angle=np.asarray(request.angles, dtype=float),
            intensity=normalized,
            raw_intensity=raw,
            normalization=normalization,
        ),
    )


