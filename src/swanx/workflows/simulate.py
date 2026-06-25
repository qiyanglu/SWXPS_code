"""High-level simulation API for reflectivity and SW-XPS rocking curves."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import sqrt
from typing import Literal

import numpy as np

from ..optics.fields import (
    FieldProfile,
    transfer_matrix_electric_field_profiles,
    transfer_matrix_reflectivity_array,
)
from ..polarization import Polarization, polarization_weights
from ..preprocessing import normalize_rocking_curve
from ..stack.model import SimulationStack
from ..stack.slicing import FixedLayerGridPlan, LayerSlicingPolicy
from ..xps.intensity import graded_layer_property_at_depth, integrate_xps_intensity
from ..xps.rocking_curve import RockingCurve
from ..xps.utils import _apply_emitting_layer_filter, _values_by_material


def _is_default_legacy_step(value: float | Sequence[float]) -> bool:
    """Return whether a legacy step argument was left at its scalar default."""

    values = np.asarray(value)
    return values.ndim == 0 and float(values) == 1.0


@dataclass(frozen=True)
class ReflectivityRequest:
    """Input parameters for a reflectivity simulation."""

    angles: np.ndarray
    energy_ev: float
    stack: SimulationStack
    angle_offset: float = 0.0
    roughness_step: float | Sequence[float] = 1.0
    roughness_profile: Literal["erf", "linear"] = "erf"
    erf_truncation_factor: float = 4.0
    linear_width_factor: float = sqrt(3.0)
    polarization: Polarization = "s"
    slicing: LayerSlicingPolicy | FixedLayerGridPlan | None = field(
        default_factory=LayerSlicingPolicy
    )

    def __post_init__(self) -> None:
        polarization_weights(self.polarization)
        if self.slicing is not None and not _is_default_legacy_step(
            self.roughness_step
        ):
            raise ValueError(
                "roughness_step is only used by the legacy path; "
                "set slicing=None or remove roughness_step"
            )


@dataclass(frozen=True)
class ReflectivityResult:
    """Output from a reflectivity simulation."""

    angle: np.ndarray
    calculation_angle: np.ndarray
    reflectivity: np.ndarray


@dataclass(frozen=True)
class CoreLevelRequest:
    """Input parameters for one normalized SW-XPS core-level RC.

    ``imfp_by_material`` must contain electron IMFP values in Angstrom at the
    photoelectron kinetic energy for this core level. Use
    ``swanx.io.read_imfp``, ``swanx.io.core_level_from_tables``, or
    ``swanx.io.core_levels_from_specs`` to construct this dictionary from IMFP
    files.

    Use `emitting_layer_indices` to select the stack layers that emit this
    core level. Leaving it as `None` keeps all layers with matching material
    labels active for backward-compatible material-level simulations.
    """

    name: str
    binding_energy_ev: float
    concentration_by_material: dict[str, float]
    imfp_by_material: dict[str, float]
    emission_angle_deg: float = 0.0
    emitting_layer_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class RockingCurveRequest:
    """Input parameters for one or more normalized SW-XPS RCs.

    ``RockingCurveRequest`` assumes the stack already contains optical
    constants at ``photon_energy_ev``, and each ``CoreLevelRequest`` already
    contains IMFP values at its own kinetic energy. It does not read OPC or
    IMFP files directly.
    """

    angles: np.ndarray
    photon_energy_ev: float
    stack: SimulationStack
    core_levels: tuple[CoreLevelRequest, ...]
    angle_offset: float = 0.0
    field_step: float = 1.0
    roughness_step: float | Sequence[float] = 1.0
    roughness_profile: Literal["erf", "linear"] = "erf"
    erf_truncation_factor: float = 4.0
    linear_width_factor: float = sqrt(3.0)
    polarization: Polarization = "s"
    offpeak_mask: np.ndarray | None = None
    normalization_mode: Literal["mean", "edge_polynomial"] = "mean"
    normalization_edge_fraction: float = 0.10
    normalization_polynomial_order: int = 2
    slicing: LayerSlicingPolicy | FixedLayerGridPlan | None = field(
        default_factory=LayerSlicingPolicy
    )

    def __post_init__(self) -> None:
        polarization_weights(self.polarization)
        if self.slicing is None:
            return
        if not _is_default_legacy_step(self.field_step):
            raise ValueError(
                "field_step is only used by the legacy path; "
                "set slicing=None or remove field_step"
            )
        if not _is_default_legacy_step(self.roughness_step):
            raise ValueError(
                "roughness_step is only used by the legacy path; "
                "set slicing=None or remove roughness_step"
            )


@dataclass(frozen=True)
class CoreLevelResult:
    """Output for one normalized SW-XPS core-level RC."""

    name: str
    binding_energy_ev: float
    kinetic_energy_ev: float
    curve: RockingCurve


@dataclass(frozen=True)
class RockingCurveResult:
    """Output from a multi-core-level SW-XPS RC simulation."""

    angle: np.ndarray
    calculation_angle: np.ndarray
    core_levels: tuple[CoreLevelResult, ...]


def simulate_reflectivity(request: ReflectivityRequest) -> ReflectivityResult:
    """Simulate reflectivity with explicit angle-offset handling."""

    if request.slicing is not None:
        from ..simulation_unified import simulate_reflectivity_unified

        return simulate_reflectivity_unified(request, backend="numpy")

    angles = np.asarray(request.angles, dtype=float)
    calculation_angle = angles + request.angle_offset
    layers = request.stack.optical_layers
    reflectivity = transfer_matrix_reflectivity_array(
        calculation_angle,
        request.energy_ev,
        layers,
        roughness_step=request.roughness_step,
        roughness_profile=request.roughness_profile,
        erf_truncation_factor=request.erf_truncation_factor,
        linear_width_factor=request.linear_width_factor,
        polarization=request.polarization,
    ).astype(
        float,
        copy=False,
    )
    return ReflectivityResult(
        angle=angles,
        calculation_angle=calculation_angle,
        reflectivity=reflectivity,
    )


def simulate_rocking_curve(
    request: RockingCurveRequest,
    core_level: CoreLevelRequest,
) -> CoreLevelResult:
    """Simulate one normalized SW-XPS rocking curve."""

    one_core_request = RockingCurveRequest(
        angles=request.angles,
        photon_energy_ev=request.photon_energy_ev,
        stack=request.stack,
        core_levels=(core_level,),
        angle_offset=request.angle_offset,
        field_step=request.field_step,
        roughness_step=request.roughness_step,
        roughness_profile=request.roughness_profile,
        erf_truncation_factor=request.erf_truncation_factor,
        linear_width_factor=request.linear_width_factor,
        polarization=request.polarization,
        offpeak_mask=request.offpeak_mask,
        normalization_mode=request.normalization_mode,
        normalization_edge_fraction=request.normalization_edge_fraction,
        normalization_polynomial_order=request.normalization_polynomial_order,
        slicing=request.slicing,
    )
    return simulate_rocking_curves(one_core_request).core_levels[0]


def simulate_rocking_curves(request: RockingCurveRequest) -> RockingCurveResult:
    """Simulate all requested normalized SW-XPS rocking curves.

    Electric-field profiles are computed once per angle and reused for every
    requested core level. This is the preferred entry point for fitting.
    """

    if request.slicing is not None:
        from ..simulation_unified import simulate_rocking_curves_unified

        return simulate_rocking_curves_unified(request, backend="numpy")

    angles = np.asarray(request.angles, dtype=float)
    calculation_angle = angles + request.angle_offset
    layers = request.stack.optical_layers
    profiles = transfer_matrix_electric_field_profiles(
        calculation_angle,
        request.photon_energy_ev,
        layers,
        step=request.field_step,
        roughness_step=request.roughness_step,
        roughness_profile=request.roughness_profile,
        erf_truncation_factor=request.erf_truncation_factor,
        linear_width_factor=request.linear_width_factor,
        polarization=request.polarization,
    )
    results = tuple(
        _simulate_core_from_profiles(
            request,
            core_level,
            profiles,
        )
        for core_level in request.core_levels
    )
    return RockingCurveResult(
        angle=angles,
        calculation_angle=calculation_angle,
        core_levels=results,
    )


def _simulate_core_from_profiles(
    request: RockingCurveRequest,
    core_level: CoreLevelRequest,
    profiles: Sequence[FieldProfile],
) -> CoreLevelResult:
    angles = np.asarray(request.angles, dtype=float)
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

    layers = request.stack.optical_layers
    if profiles:
        concentration = graded_layer_property_at_depth(
            layers,
            concentration_by_layer,
            profiles[0].depth,
            profile=request.roughness_profile,
            erf_truncation_factor=request.erf_truncation_factor,
            linear_width_factor=request.linear_width_factor,
        )
        attenuation_coefficient = graded_layer_property_at_depth(
            layers,
            1.0 / np.asarray(imfp_by_layer, dtype=float),
            profiles[0].depth,
            profile=request.roughness_profile,
            erf_truncation_factor=request.erf_truncation_factor,
            linear_width_factor=request.linear_width_factor,
        )
        attenuation_length = 1.0 / attenuation_coefficient
    else:
        concentration = np.array([], dtype=float)
        attenuation_length = np.array([], dtype=float)

    raw_intensity = np.fromiter(
        (
            integrate_xps_intensity(
                profile,
                concentration,
                attenuation_length,
                emission_angle_deg=core_level.emission_angle_deg,
            )
            for profile in profiles
        ),
        dtype=float,
        count=len(profiles),
    )

    normalized, normalization = normalize_rocking_curve(
        angles,
        raw_intensity,
        mode=request.normalization_mode,
        offpeak_mask=request.offpeak_mask,
        edge_fraction=request.normalization_edge_fraction,
        polynomial_order=request.normalization_polynomial_order,
    )

    curve = RockingCurve(
        angle=angles,
        intensity=normalized,
        raw_intensity=raw_intensity,
        normalization=normalization,
    )
    return CoreLevelResult(
        name=core_level.name,
        binding_energy_ev=core_level.binding_energy_ev,
        kinetic_energy_ev=kinetic_energy_ev,
        curve=curve,
    )


__all__ = [
    "CoreLevelRequest",
    "CoreLevelResult",
    "ReflectivityRequest",
    "ReflectivityResult",
    "RockingCurveRequest",
    "RockingCurveResult",
    "simulate_reflectivity",
    "simulate_rocking_curve",
    "simulate_rocking_curves",
]
