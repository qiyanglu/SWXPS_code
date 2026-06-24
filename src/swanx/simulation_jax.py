"""Optional JAX-backed high-level simulations.

These entry points mirror ``simulation.py`` result types while keeping the
existing NumPy backend untouched. Roughness discretization, concentration
grading, and emitting-layer selection are intentionally reused from the NumPy
helpers before fixed-shape arrays enter JIT.
"""

from __future__ import annotations

import numpy as np

from .optics.fields import depth_grid, effective_layers_with_roughness
from .preprocessing import normalize_rocking_curve
from .reflectivity_jax import (
    jitted_normalized_rocking_curve_from_field,
    jitted_transfer_matrix_field_intensity,
    jitted_transfer_matrix_reflectivity,
    layer_arrays_from_layers,
)
from .simulation import (
    CoreLevelRequest,
    CoreLevelResult,
    ReflectivityRequest,
    ReflectivityResult,
    RockingCurveRequest,
    RockingCurveResult,
)
from ._xps import RockingCurve, graded_layer_property_at_depth


def simulate_reflectivity_jax(request: ReflectivityRequest) -> ReflectivityResult:
    """Simulate reflectivity with the optional JAX transfer-matrix backend."""

    if request.slicing is not None:
        from .simulation_unified import simulate_reflectivity_unified

        return simulate_reflectivity_unified(request, backend="jax")

    angles = np.asarray(request.angles, dtype=float)
    calculation_angle = angles + request.angle_offset
    effective_layers = effective_layers_with_roughness(
        request.stack.optical_layers,
        step=request.roughness_step,
        profile=request.roughness_profile,
        erf_truncation_factor=request.erf_truncation_factor,
        linear_width_factor=request.linear_width_factor,
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
    return ReflectivityResult(
        angle=angles,
        calculation_angle=calculation_angle,
        reflectivity=reflectivity,
    )


def simulate_rocking_curves_jax(request: RockingCurveRequest) -> RockingCurveResult:
    """Simulate normalized SW-XPS RCs with the optional JAX backend."""

    if request.slicing is not None:
        from .simulation_unified import simulate_rocking_curves_unified

        return simulate_rocking_curves_unified(request, backend="jax")

    angles = np.asarray(request.angles, dtype=float)
    calculation_angle = angles + request.angle_offset
    nominal_layers = request.stack.optical_layers
    effective_layers = effective_layers_with_roughness(
        nominal_layers,
        step=request.roughness_step,
        profile=request.roughness_profile,
        erf_truncation_factor=request.erf_truncation_factor,
        linear_width_factor=request.linear_width_factor,
    )
    thicknesses, deltas, betas, _ = layer_arrays_from_layers(effective_layers)
    depth, layer_index = depth_grid(effective_layers, request.field_step)

    if depth.size == 0:
        field_intensity = np.zeros((0, angles.size), dtype=float)
    else:
        field_intensity = jitted_transfer_matrix_field_intensity(
            calculation_angle,
            request.photon_energy_ev,
            thicknesses,
            deltas,
            betas,
            depth,
            layer_index,
        )

    results = tuple(
        _simulate_core_from_jax_field(
            request,
            core_level,
            depth,
            field_intensity,
        )
        for core_level in request.core_levels
    )
    return RockingCurveResult(
        angle=angles,
        calculation_angle=calculation_angle,
        core_levels=results,
    )


def _simulate_core_from_jax_field(
    request: RockingCurveRequest,
    core_level: CoreLevelRequest,
    depth: np.ndarray,
    field_intensity,
) -> CoreLevelResult:
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

    if depth.size == 0:
        raw_intensity = np.zeros(np.asarray(request.angles).shape, dtype=float)
        normalization = 0.0
        normalized = raw_intensity
    else:
        nominal_layers = request.stack.optical_layers
        concentration = graded_layer_property_at_depth(
            nominal_layers,
            concentration_by_layer,
            depth,
            profile=request.roughness_profile,
            erf_truncation_factor=request.erf_truncation_factor,
            linear_width_factor=request.linear_width_factor,
        )
        attenuation_coefficient = graded_layer_property_at_depth(
            nominal_layers,
            1.0 / np.asarray(imfp_by_layer, dtype=float),
            depth,
            profile=request.roughness_profile,
            erf_truncation_factor=request.erf_truncation_factor,
            linear_width_factor=request.linear_width_factor,
        )
        attenuation_length = 1.0 / attenuation_coefficient
        offpeak_mask = _offpeak_mask(request)
        normalized_jax, raw_jax, normalization_jax = (
            jitted_normalized_rocking_curve_from_field(
                field_intensity,
                depth,
                concentration,
                attenuation_length,
                core_level.emission_angle_deg,
                offpeak_mask,
            )
        )
        raw_intensity = np.asarray(raw_jax, dtype=float)
        if request.normalization_mode == "mean":
            normalized = np.asarray(normalized_jax, dtype=float)
            normalization = float(normalization_jax)
        else:
            normalized, normalization = normalize_rocking_curve(
                np.asarray(request.angles, dtype=float),
                raw_intensity,
                mode=request.normalization_mode,
                offpeak_mask=request.offpeak_mask,
                edge_fraction=request.normalization_edge_fraction,
                polynomial_order=request.normalization_polynomial_order,
            )

    if np.any(np.asarray(normalization) <= 0):
        raise ValueError("normalization must be positive")

    return CoreLevelResult(
        name=core_level.name,
        binding_energy_ev=core_level.binding_energy_ev,
        kinetic_energy_ev=kinetic_energy_ev,
        curve=RockingCurve(
            angle=np.asarray(request.angles, dtype=float),
            intensity=normalized,
            raw_intensity=raw_intensity,
            normalization=normalization,
        ),
    )


def _values_by_material(
    materials: list[str],
    values: dict[str, float],
    default: float | None,
) -> list[float]:
    output = []
    for material in materials:
        if material in values:
            output.append(float(values[material]))
        elif default is None:
            raise ValueError(f"missing value for material {material!r}")
        else:
            output.append(float(default))
    return output


def _apply_emitting_layer_filter(
    concentration_by_layer: list[float],
    emitting_layer_indices: tuple[int, ...],
) -> list[float]:
    if not emitting_layer_indices:
        raise ValueError("emitting_layer_indices must not be empty")
    layer_count = len(concentration_by_layer)
    selected = set()
    for index in emitting_layer_indices:
        if index < 0 or index >= layer_count:
            raise ValueError("emitting_layer_indices contains an index outside the stack")
        selected.add(int(index))
    return [
        concentration if index in selected else 0.0
        for index, concentration in enumerate(concentration_by_layer)
    ]


def _offpeak_mask(request: RockingCurveRequest) -> np.ndarray:
    angles = np.asarray(request.angles, dtype=float)
    if request.offpeak_mask is None:
        return np.ones(angles.shape, dtype=bool)
    offpeak_mask = np.asarray(request.offpeak_mask, dtype=bool)
    if offpeak_mask.shape != angles.shape:
        raise ValueError("offpeak_mask must match angles shape")
    if not np.any(offpeak_mask):
        raise ValueError("offpeak_mask must select at least one angle")
    return offpeak_mask
