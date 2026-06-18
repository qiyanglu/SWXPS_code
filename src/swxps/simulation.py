"""High-level simulation API for reflectivity and SW-XPS rocking curves."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from typing import Literal

import numpy as np

from .fields import (
    FieldProfile,
    transfer_matrix_electric_field_profiles,
    transfer_matrix_reflectivity_array,
)
from .layers import Layer
from .xps import RockingCurve, graded_layer_property_at_depth, integrate_xps_intensity


@dataclass(frozen=True)
class StackLayer:
    """A material-labeled layer used by high-level simulations."""

    material: str
    thickness: float
    delta: float = 0.0
    beta: float = 0.0
    roughness: float = 0.0

    def to_layer(self) -> Layer:
        """Return the optical Layer representation."""

        return Layer(
            thickness=self.thickness,
            delta=self.delta,
            beta=self.beta,
            roughness=self.roughness,
        )


@dataclass(frozen=True)
class SimulationStack:
    """A material-labeled stack from vacuum to substrate."""

    layers: tuple[StackLayer, ...]

    def __post_init__(self) -> None:
        if len(self.layers) < 2:
            raise ValueError("a simulation stack requires at least two layers")

    @property
    def optical_layers(self) -> list[Layer]:
        """Return low-level optical layers."""

        return [layer.to_layer() for layer in self.layers]

    @property
    def materials(self) -> list[str]:
        """Return material labels for each layer."""

        return [layer.material for layer in self.layers]


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


@dataclass(frozen=True)
class ReflectivityResult:
    """Output from a reflectivity simulation."""

    angle: np.ndarray
    calculation_angle: np.ndarray
    reflectivity: np.ndarray


@dataclass(frozen=True)
class CoreLevelRequest:
    """Input parameters for one normalized SW-XPS core-level RC.

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
    """Input parameters for one or more normalized SW-XPS RCs."""

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
    offpeak_mask: np.ndarray | None = None


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
        offpeak_mask=request.offpeak_mask,
    )
    return simulate_rocking_curves(one_core_request).core_levels[0]


def simulate_rocking_curves(request: RockingCurveRequest) -> RockingCurveResult:
    """Simulate all requested normalized SW-XPS rocking curves.

    Electric-field profiles are computed once per angle and reused for every
    requested core level. This is the preferred entry point for fitting.
    """

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

    if request.offpeak_mask is None:
        offpeak_mask = np.ones(angles.shape, dtype=bool)
    else:
        offpeak_mask = np.asarray(request.offpeak_mask, dtype=bool)
    if offpeak_mask.shape != angles.shape:
        raise ValueError("offpeak_mask must match angles shape")
    if not np.any(offpeak_mask):
        raise ValueError("offpeak_mask must select at least one angle")

    normalization = float(np.mean(raw_intensity[offpeak_mask]))
    if normalization <= 0:
        raise ValueError("normalization must be positive")

    curve = RockingCurve(
        angle=angles,
        intensity=raw_intensity / normalization,
        raw_intensity=raw_intensity,
        normalization=normalization,
    )
    return CoreLevelResult(
        name=core_level.name,
        binding_energy_ev=core_level.binding_energy_ev,
        kinetic_energy_ev=kinetic_energy_ev,
        curve=curve,
    )


def stack_from_layers(materials: Sequence[str], layers: Sequence[Layer]) -> SimulationStack:
    """Create a material-labeled stack from existing Layer objects."""

    if len(materials) != len(layers):
        raise ValueError("materials and layers must have the same length")
    return SimulationStack(
        tuple(
            StackLayer(
                material=material,
                thickness=layer.thickness,
                delta=layer.delta,
                beta=layer.beta,
                roughness=layer.roughness,
            )
            for material, layer in zip(materials, layers)
        )
    )


def _values_by_material(
    materials: Sequence[str],
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
    concentration_by_layer: Sequence[float],
    emitting_layer_indices: Sequence[int],
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
        float(concentration) if index in selected else 0.0
        for index, concentration in enumerate(concentration_by_layer)
    ]
