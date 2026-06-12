"""Depth-dependent electric fields for multilayer stacks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import erf, sqrt

import numpy as np

from .layers import Layer
from .reflectivity import (
    apply_roughness,
    energy_to_wavelength,
    fresnel_r_s,
    kz_in_layers,
)


@dataclass(frozen=True)
class FieldProfile:
    """Electric field sampled on a depth grid."""

    depth: np.ndarray
    electric_field: np.ndarray
    intensity: np.ndarray
    layer_index: np.ndarray


def parratt_reflection_amplitudes(
    angle_deg: float,
    energy_ev: float,
    layers: Sequence[Layer],
) -> np.ndarray:
    """Return Parratt reflection amplitudes at the top of each layer."""

    _validate_field_inputs(angle_deg, layers)

    wavelength = energy_to_wavelength(energy_ev)
    kz = kz_in_layers(angle_deg, wavelength, [layer.n for layer in layers])
    thicknesses = np.asarray([layer.thickness for layer in layers], dtype=float)
    roughnesses = np.asarray([layer.roughness for layer in layers], dtype=float)

    amplitudes = np.zeros(len(layers), dtype=complex)
    for j in range(len(layers) - 2, -1, -1):
        r_interface = fresnel_r_s(kz[j], kz[j + 1])
        r_interface = apply_roughness(
            r_interface,
            kz[j],
            kz[j + 1],
            roughnesses[j + 1],
        )
        phase = np.exp(2j * kz[j + 1] * thicknesses[j + 1])
        amplitudes[j] = (r_interface + amplitudes[j + 1] * phase) / (
            1.0 + r_interface * amplitudes[j + 1] * phase
        )

    return amplitudes


def layer_field_amplitudes(
    angle_deg: float,
    energy_ev: float,
    layers: Sequence[Layer],
) -> tuple[np.ndarray, np.ndarray]:
    """Return downward and upward field amplitudes at each layer top."""

    reflection = parratt_reflection_amplitudes(angle_deg, energy_ev, layers)
    wavelength = energy_to_wavelength(energy_ev)
    kz = kz_in_layers(angle_deg, wavelength, [layer.n for layer in layers])
    thicknesses = np.asarray([layer.thickness for layer in layers], dtype=float)

    downward = np.zeros(len(layers), dtype=complex)
    upward = np.zeros(len(layers), dtype=complex)
    downward[0] = 1.0 + 0.0j
    upward[0] = reflection[0]

    for j in range(len(layers) - 1):
        bottom_field = (
            downward[j] * np.exp(1j * kz[j] * thicknesses[j])
            + upward[j] * np.exp(-1j * kz[j] * thicknesses[j])
        )
        downward[j + 1] = bottom_field / (1.0 + reflection[j + 1])
        upward[j + 1] = reflection[j + 1] * downward[j + 1]

    return downward, upward


def depth_grid(layers: Sequence[Layer], step: float) -> tuple[np.ndarray, np.ndarray]:
    """Return depths and layer indices for all finite layers."""

    if step <= 0:
        raise ValueError("step must be positive")

    depths: list[np.ndarray] = []
    layer_indices: list[np.ndarray] = []
    current_depth = 0.0

    for index in range(1, len(layers) - 1):
        thickness = layers[index].thickness
        n_points = max(2, int(np.ceil(thickness / step)) + 1)
        local_depth = np.linspace(0.0, thickness, n_points, endpoint=True)
        if depths:
            local_depth = local_depth[1:]
        global_depth = current_depth + local_depth
        depths.append(global_depth)
        layer_indices.append(np.full(global_depth.shape, index, dtype=int))
        current_depth += thickness

    if not depths:
        return np.array([], dtype=float), np.array([], dtype=int)

    return np.concatenate(depths), np.concatenate(layer_indices)


def electric_field_profile(
    angle_deg: float,
    energy_ev: float,
    layers: Sequence[Layer],
    step: float = 1.0,
) -> FieldProfile:
    """Return complex electric field and intensity through finite layers."""

    _validate_field_inputs(angle_deg, layers)
    depths, layer_indices = depth_grid(layers, step)
    if depths.size == 0:
        return FieldProfile(
            depth=depths,
            electric_field=np.array([], dtype=complex),
            intensity=np.array([], dtype=float),
            layer_index=layer_indices,
        )

    downward, upward = layer_field_amplitudes(angle_deg, energy_ev, layers)
    wavelength = energy_to_wavelength(energy_ev)
    kz = kz_in_layers(angle_deg, wavelength, [layer.n for layer in layers])
    starts = _finite_layer_start_depths(layers)

    electric_field = np.empty(depths.shape, dtype=complex)
    for index in range(1, len(layers) - 1):
        mask = layer_indices == index
        local_depth = depths[mask] - starts[index]
        electric_field[mask] = (
            downward[index] * np.exp(1j * kz[index] * local_depth)
            + upward[index] * np.exp(-1j * kz[index] * local_depth)
        )

    intensity = np.abs(electric_field) ** 2
    return FieldProfile(
        depth=depths,
        electric_field=electric_field,
        intensity=intensity,
        layer_index=layer_indices,
    )


def transfer_matrix_reflection_amplitude(
    angle_deg: float,
    energy_ev: float,
    layers: Sequence[Layer],
    roughness_step: float = 0.5,
    width_factor: float = 4.0,
) -> complex:
    """Return the reflection amplitude from transfer matrices.

    If any layer roughness is nonzero, rough interfaces are approximated by a
    graded stack of thin effective layers.
    """

    downward, upward = transfer_matrix_field_amplitudes(
        angle_deg,
        energy_ev,
        layers,
        roughness_step=roughness_step,
        width_factor=width_factor,
    )
    return upward[0] / downward[0]


def transfer_matrix_reflectivity(
    angle_deg: float,
    energy_ev: float,
    layers: Sequence[Layer],
    roughness_step: float = 0.5,
    width_factor: float = 4.0,
) -> float:
    """Return reflectivity from the transfer-matrix solution."""

    amplitude = transfer_matrix_reflection_amplitude(
        angle_deg,
        energy_ev,
        layers,
        roughness_step=roughness_step,
        width_factor=width_factor,
    )
    return float(abs(amplitude) ** 2)


def transfer_matrix_field_amplitudes(
    angle_deg: float,
    energy_ev: float,
    layers: Sequence[Layer],
    roughness_step: float = 0.5,
    width_factor: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return downward and upward amplitudes in each transfer-matrix layer.

    For nonzero roughness, the returned amplitudes correspond to the
    discretized effective stack, not the original nominal layers.
    """

    _validate_field_inputs(angle_deg, layers)
    effective_layers = effective_layers_with_roughness(
        layers,
        step=roughness_step,
        width_factor=width_factor,
    )

    return _transfer_matrix_field_amplitudes_sharp(angle_deg, energy_ev, effective_layers)


def _transfer_matrix_field_amplitudes_sharp(
    angle_deg: float,
    energy_ev: float,
    layers: Sequence[Layer],
) -> tuple[np.ndarray, np.ndarray]:
    """Return transfer-matrix amplitudes for a sharp-interface stack."""

    wavelength = energy_to_wavelength(energy_ev)
    kz = kz_in_layers(angle_deg, wavelength, [layer.n for layer in layers])
    thicknesses = np.asarray([layer.thickness for layer in layers], dtype=float)

    total = np.eye(2, dtype=complex)
    for j in range(len(layers) - 1):
        total = _interface_matrix(kz[j], kz[j + 1]) @ _propagation_matrix(
            kz[j],
            thicknesses[j],
        ) @ total

    reflection = -total[1, 0] / total[1, 1]

    downward = np.zeros(len(layers), dtype=complex)
    upward = np.zeros(len(layers), dtype=complex)
    downward[0] = 1.0 + 0.0j
    upward[0] = reflection

    state = np.array([downward[0], upward[0]], dtype=complex)
    for j in range(len(layers) - 1):
        state = _interface_matrix(kz[j], kz[j + 1]) @ _propagation_matrix(
            kz[j],
            thicknesses[j],
        ) @ state
        downward[j + 1] = state[0]
        upward[j + 1] = state[1]

    return downward, upward


def transfer_matrix_electric_field_profile(
    angle_deg: float,
    energy_ev: float,
    layers: Sequence[Layer],
    step: float = 1.0,
    roughness_step: float = 0.5,
    width_factor: float = 4.0,
) -> FieldProfile:
    """Return a transfer-matrix electric-field profile.

    Nonzero roughness is represented by thin effective layers. The profile is
    sampled through the effective finite stack.
    """

    _validate_field_inputs(angle_deg, layers)
    effective_layers = effective_layers_with_roughness(
        layers,
        step=roughness_step,
        width_factor=width_factor,
    )

    depths, layer_indices = depth_grid(effective_layers, step)
    if depths.size == 0:
        return FieldProfile(
            depth=depths,
            electric_field=np.array([], dtype=complex),
            intensity=np.array([], dtype=float),
            layer_index=layer_indices,
        )

    downward, upward = _transfer_matrix_field_amplitudes_sharp(
        angle_deg,
        energy_ev,
        effective_layers,
    )
    wavelength = energy_to_wavelength(energy_ev)
    kz = kz_in_layers(angle_deg, wavelength, [layer.n for layer in effective_layers])
    starts = _finite_layer_start_depths(effective_layers)

    electric_field = np.empty(depths.shape, dtype=complex)
    for index in range(1, len(effective_layers) - 1):
        mask = layer_indices == index
        local_depth = depths[mask] - starts[index]
        electric_field[mask] = (
            downward[index] * np.exp(1j * kz[index] * local_depth)
            + upward[index] * np.exp(-1j * kz[index] * local_depth)
        )

    return FieldProfile(
        depth=depths,
        electric_field=electric_field,
        intensity=np.abs(electric_field) ** 2,
        layer_index=layer_indices,
    )


def effective_layers_with_roughness(
    layers: Sequence[Layer],
    step: float = 0.5,
    width_factor: float = 4.0,
) -> list[Layer]:
    """Return a sharp-interface effective stack with graded rough interfaces."""

    _validate_field_inputs(0.0, layers)
    if step <= 0:
        raise ValueError("step must be positive")
    if width_factor <= 0:
        raise ValueError("width_factor must be positive")
    if not _has_roughness(layers):
        return list(layers)

    total_thickness = sum(layer.thickness for layer in layers[1:-1])
    if total_thickness <= 0:
        return [layers[0], layers[-1]]

    n_slices = max(1, int(np.ceil(total_thickness / step)))
    edges = np.linspace(0.0, total_thickness, n_slices + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    boundaries = _nominal_boundaries(layers)
    finite_layers: list[Layer] = []
    for left, right, center in zip(edges[:-1], edges[1:], centers):
        delta, beta = _graded_delta_beta_at_depth(
            center,
            layers,
            boundaries,
            width_factor,
        )
        finite_layers.append(
            Layer(
                thickness=float(right - left),
                delta=delta,
                beta=beta,
                roughness=0.0,
            )
        )

    return [Layer(0.0, layers[0].delta, layers[0].beta), *finite_layers, Layer(0.0, layers[-1].delta, layers[-1].beta)]


def _finite_layer_start_depths(layers: Sequence[Layer]) -> np.ndarray:
    starts = np.zeros(len(layers), dtype=float)
    current_depth = 0.0
    for index in range(1, len(layers) - 1):
        starts[index] = current_depth
        current_depth += layers[index].thickness
    return starts


def _propagation_matrix(kz: complex, thickness: float) -> np.ndarray:
    return np.array(
        [
            [np.exp(1j * kz * thickness), 0.0 + 0.0j],
            [0.0 + 0.0j, np.exp(-1j * kz * thickness)],
        ],
        dtype=complex,
    )


def _interface_matrix(kz_top: complex, kz_bottom: complex) -> np.ndarray:
    ratio = kz_top / kz_bottom
    return 0.5 * np.array(
        [
            [1.0 + ratio, 1.0 - ratio],
            [1.0 - ratio, 1.0 + ratio],
        ],
        dtype=complex,
    )


def _has_roughness(layers: Sequence[Layer]) -> bool:
    return any(layer.roughness != 0 for layer in layers)


def _nominal_boundaries(layers: Sequence[Layer]) -> np.ndarray:
    boundaries = [0.0]
    current_depth = 0.0
    for layer in layers[1:-1]:
        current_depth += layer.thickness
        boundaries.append(current_depth)
    return np.asarray(boundaries, dtype=float)


def _graded_delta_beta_at_depth(
    depth: float,
    layers: Sequence[Layer],
    boundaries: np.ndarray,
    width_factor: float,
) -> tuple[float, float]:
    nearest = int(np.argmin(np.abs(boundaries - depth)))
    sigma = layers[nearest + 1].roughness

    if sigma > 0 and abs(depth - boundaries[nearest]) <= width_factor * sigma:
        above = layers[nearest]
        below = layers[nearest + 1]
        fraction = 0.5 * (
            1.0 + erf((depth - boundaries[nearest]) / (sqrt(2.0) * sigma))
        )
        delta = (1.0 - fraction) * above.delta + fraction * below.delta
        beta = (1.0 - fraction) * above.beta + fraction * below.beta
        return float(delta), float(beta)

    nominal_index = int(np.searchsorted(boundaries[1:], depth, side="right")) + 1
    nominal_index = min(max(nominal_index, 1), len(layers) - 2)
    return float(layers[nominal_index].delta), float(layers[nominal_index].beta)


def _validate_field_inputs(angle_deg: float, layers: Sequence[Layer]) -> None:
    if np.asarray(angle_deg).ndim != 0:
        raise ValueError("field profiles require a single scalar angle")
    if len(layers) < 2:
        raise ValueError("at least two layers are required")
    for layer in layers:
        if layer.thickness < 0:
            raise ValueError("layer thickness values must be non-negative")
        if not np.isfinite(layer.thickness):
            raise ValueError("layer thickness values must be finite")
        if layer.roughness < 0:
            raise ValueError("layer roughness values must be non-negative")
        if not np.isfinite(layer.roughness):
            raise ValueError("layer roughness values must be finite")
