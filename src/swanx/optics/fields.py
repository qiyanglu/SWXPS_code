"""Depth-dependent electric fields for multilayer stacks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import erf, sqrt
from typing import Literal

import numpy as np

from ..layers import Layer
from ..polarization import Polarization, polarization_weights
from .parratt import (
    apply_roughness,
    energy_to_wavelength,
    fresnel_r_s,
    kz_in_layers,
)


@dataclass(frozen=True)
class FieldProfile:
    """Electric field sampled on a depth grid.

    ``electric_field`` is a scalar representative amplitude. For p-polarized or
    mixed-polarization calculations, use ``intensity`` for physical weighting.
    """

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
    roughness_step: float | Sequence[float] = 1.0,
    roughness_profile: Literal["erf", "linear"] = "erf",
    erf_truncation_factor: float = 4.0,
    linear_width_factor: float = sqrt(3.0),
    polarization: Polarization = "s",
) -> complex:
    """Return the reflection amplitude from transfer matrices.

    If any layer roughness is nonzero, rough interfaces are approximated by a
    graded stack of thin effective layers.
    """

    s_weight, p_weight = polarization_weights(polarization)
    if s_weight and p_weight:
        raise ValueError("reflection amplitude requires a pure 's' or 'p' polarization")
    pure_polarization = "s" if s_weight else "p"
    downward, upward = transfer_matrix_field_amplitudes(
        angle_deg,
        energy_ev,
        layers,
        roughness_step=roughness_step,
        roughness_profile=roughness_profile,
        erf_truncation_factor=erf_truncation_factor,
        linear_width_factor=linear_width_factor,
        polarization=pure_polarization,
    )
    return upward[0] / downward[0]


def transfer_matrix_reflectivity(
    angle_deg: float,
    energy_ev: float,
    layers: Sequence[Layer],
    roughness_step: float | Sequence[float] = 1.0,
    roughness_profile: Literal["erf", "linear"] = "erf",
    erf_truncation_factor: float = 4.0,
    linear_width_factor: float = sqrt(3.0),
    polarization: Polarization = "s",
) -> float:
    """Return reflectivity from the transfer-matrix solution."""

    s_weight, p_weight = polarization_weights(polarization)
    if s_weight and p_weight:
        return float(
            s_weight
            * transfer_matrix_reflectivity(
                angle_deg,
                energy_ev,
                layers,
                roughness_step=roughness_step,
                roughness_profile=roughness_profile,
                erf_truncation_factor=erf_truncation_factor,
                linear_width_factor=linear_width_factor,
                polarization="s",
            )
            + p_weight
            * transfer_matrix_reflectivity(
                angle_deg,
                energy_ev,
                layers,
                roughness_step=roughness_step,
                roughness_profile=roughness_profile,
                erf_truncation_factor=erf_truncation_factor,
                linear_width_factor=linear_width_factor,
                polarization="p",
            )
        )
    pure_polarization = "s" if s_weight else "p"
    amplitude = transfer_matrix_reflection_amplitude(
        angle_deg,
        energy_ev,
        layers,
        roughness_step=roughness_step,
        roughness_profile=roughness_profile,
        erf_truncation_factor=erf_truncation_factor,
        linear_width_factor=linear_width_factor,
        polarization=pure_polarization,
    )
    return float(abs(amplitude) ** 2)


def transfer_matrix_reflectivity_array(
    angle_deg: np.ndarray,
    energy_ev: float,
    layers: Sequence[Layer],
    roughness_step: float | Sequence[float] = 1.0,
    roughness_profile: Literal["erf", "linear"] = "erf",
    erf_truncation_factor: float = 4.0,
    linear_width_factor: float = sqrt(3.0),
    polarization: Polarization = "s",
) -> np.ndarray:
    """Return transfer-matrix reflectivity for many angles.

    The layer recursion remains explicit, but all angles are propagated in one
    NumPy batch. Rough-interface discretization is built once for the request.
    """

    angles = np.asarray(angle_deg, dtype=float)
    s_weight, p_weight = polarization_weights(polarization)
    if s_weight and p_weight:
        return (
            s_weight
            * transfer_matrix_reflectivity_array(
                angles,
                energy_ev,
                layers,
                roughness_step=roughness_step,
                roughness_profile=roughness_profile,
                erf_truncation_factor=erf_truncation_factor,
                linear_width_factor=linear_width_factor,
                polarization="s",
            )
            + p_weight
            * transfer_matrix_reflectivity_array(
                angles,
                energy_ev,
                layers,
                roughness_step=roughness_step,
                roughness_profile=roughness_profile,
                erf_truncation_factor=erf_truncation_factor,
                linear_width_factor=linear_width_factor,
                polarization="p",
            )
        )
    pure_polarization = "s" if s_weight else "p"
    if angles.ndim == 0:
        return np.asarray(
            transfer_matrix_reflectivity(
                float(angles),
                energy_ev,
                layers,
                roughness_step=roughness_step,
                roughness_profile=roughness_profile,
                erf_truncation_factor=erf_truncation_factor,
                linear_width_factor=linear_width_factor,
                polarization=pure_polarization,
            )
        )

    effective_layers = effective_layers_with_roughness(
        layers,
        step=roughness_step,
        profile=roughness_profile,
        erf_truncation_factor=erf_truncation_factor,
        linear_width_factor=linear_width_factor,
    )
    downward, upward = _transfer_matrix_field_amplitudes_sharp_batched(
        angles,
        energy_ev,
        effective_layers,
        polarization=pure_polarization,
    )
    return np.abs(upward[0] / downward[0]) ** 2


def transfer_matrix_field_amplitudes(
    angle_deg: float,
    energy_ev: float,
    layers: Sequence[Layer],
    roughness_step: float | Sequence[float] = 1.0,
    roughness_profile: Literal["erf", "linear"] = "erf",
    erf_truncation_factor: float = 4.0,
    linear_width_factor: float = sqrt(3.0),
    polarization: Polarization = "s",
) -> tuple[np.ndarray, np.ndarray]:
    """Return downward and upward amplitudes in each transfer-matrix layer.

    For nonzero roughness, the returned amplitudes correspond to the
    discretized effective stack, not the original nominal layers.
    """

    _validate_field_inputs(angle_deg, layers)
    effective_layers = effective_layers_with_roughness(
        layers,
        step=roughness_step,
        profile=roughness_profile,
        erf_truncation_factor=erf_truncation_factor,
        linear_width_factor=linear_width_factor,
    )

    s_weight, p_weight = polarization_weights(polarization)
    if s_weight and p_weight:
        raise ValueError("field amplitudes require a pure 's' or 'p' polarization")
    pure_polarization = "s" if s_weight else "p"
    return _transfer_matrix_field_amplitudes_sharp(
        angle_deg,
        energy_ev,
        effective_layers,
        polarization=pure_polarization,
    )


def _transfer_matrix_field_amplitudes_sharp_batched(
    angle_deg: np.ndarray,
    energy_ev: float,
    layers: Sequence[Layer],
    polarization: Literal["s", "p"] = "s",
) -> tuple[np.ndarray, np.ndarray]:
    """Return transfer-matrix amplitudes for a sharp stack at many angles."""

    angles = np.asarray(angle_deg, dtype=float)
    if angles.ndim != 1:
        raise ValueError("batched transfer-matrix amplitudes require a 1D angle array")
    if angles.size == 0:
        return (
            np.zeros((len(layers), 0), dtype=complex),
            np.zeros((len(layers), 0), dtype=complex),
        )

    wavelength = energy_to_wavelength(energy_ev)
    kz = kz_in_layers(angles, wavelength, [layer.n for layer in layers])
    n_values = np.asarray([layer.n for layer in layers], dtype=complex)[:, np.newaxis]
    admittance = _admittance(kz, n_values, polarization)
    thicknesses = np.asarray([layer.thickness for layer in layers], dtype=float)

    total_00 = np.ones(angles.shape, dtype=complex)
    total_01 = np.zeros(angles.shape, dtype=complex)
    total_10 = np.zeros(angles.shape, dtype=complex)
    total_11 = np.ones(angles.shape, dtype=complex)
    for j in range(len(layers) - 1):
        m00, m01, m10, m11 = _interface_matrix_elements(
            admittance[j],
            admittance[j + 1],
        )
        p = np.exp(1j * kz[j] * thicknesses[j])
        q = np.exp(-1j * kz[j] * thicknesses[j])
        next_00 = m00 * p * total_00 + m01 * q * total_10
        next_01 = m00 * p * total_01 + m01 * q * total_11
        next_10 = m10 * p * total_00 + m11 * q * total_10
        next_11 = m10 * p * total_01 + m11 * q * total_11
        total_00, total_01, total_10, total_11 = (
            next_00,
            next_01,
            next_10,
            next_11,
        )

    reflection = -total_10 / total_11

    downward = np.zeros((len(layers), angles.size), dtype=complex)
    upward = np.zeros((len(layers), angles.size), dtype=complex)
    downward[0] = 1.0 + 0.0j
    upward[0] = reflection

    for j in range(len(layers) - 1):
        m00, m01, m10, m11 = _interface_matrix_elements(
            admittance[j],
            admittance[j + 1],
        )
        p = np.exp(1j * kz[j] * thicknesses[j])
        q = np.exp(-1j * kz[j] * thicknesses[j])
        propagated_down = p * downward[j]
        propagated_up = q * upward[j]
        downward[j + 1] = m00 * propagated_down + m01 * propagated_up
        upward[j + 1] = m10 * propagated_down + m11 * propagated_up

    return downward, upward


def _transfer_matrix_field_amplitudes_sharp(
    angle_deg: float,
    energy_ev: float,
    layers: Sequence[Layer],
    polarization: Literal["s", "p"] = "s",
) -> tuple[np.ndarray, np.ndarray]:
    """Return transfer-matrix amplitudes for a sharp-interface stack."""

    wavelength = energy_to_wavelength(energy_ev)
    kz = kz_in_layers(angle_deg, wavelength, [layer.n for layer in layers])
    n_values = np.asarray([layer.n for layer in layers], dtype=complex)
    admittance = _admittance(kz, n_values, polarization)
    thicknesses = np.asarray([layer.thickness for layer in layers], dtype=float)

    total = np.eye(2, dtype=complex)
    for j in range(len(layers) - 1):
        total = _interface_matrix(admittance[j], admittance[j + 1]) @ _propagation_matrix(
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
        state = _interface_matrix(admittance[j], admittance[j + 1]) @ _propagation_matrix(
            kz[j],
            thicknesses[j],
        ) @ state
        downward[j + 1] = state[0]
        upward[j + 1] = state[1]

    return downward, upward


def transfer_matrix_electric_field_profiles(
    angle_deg: np.ndarray,
    energy_ev: float,
    layers: Sequence[Layer],
    step: float = 1.0,
    roughness_step: float | Sequence[float] = 1.0,
    roughness_profile: Literal["erf", "linear"] = "erf",
    erf_truncation_factor: float = 4.0,
    linear_width_factor: float = sqrt(3.0),
    polarization: Polarization = "s",
) -> tuple[FieldProfile, ...]:
    """Return transfer-matrix electric-field profiles for many angles."""

    angles = np.asarray(angle_deg, dtype=float)
    s_weight, p_weight = polarization_weights(polarization)
    if s_weight and p_weight:
        s_profiles = transfer_matrix_electric_field_profiles(
            angles,
            energy_ev,
            layers,
            step=step,
            roughness_step=roughness_step,
            roughness_profile=roughness_profile,
            erf_truncation_factor=erf_truncation_factor,
            linear_width_factor=linear_width_factor,
            polarization="s",
        )
        p_profiles = transfer_matrix_electric_field_profiles(
            angles,
            energy_ev,
            layers,
            step=step,
            roughness_step=roughness_step,
            roughness_profile=roughness_profile,
            erf_truncation_factor=erf_truncation_factor,
            linear_width_factor=linear_width_factor,
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
    if angles.ndim == 0:
        return (
            transfer_matrix_electric_field_profile(
                angle_deg=float(angles),
                energy_ev=energy_ev,
                layers=layers,
                step=step,
                roughness_step=roughness_step,
                roughness_profile=roughness_profile,
                erf_truncation_factor=erf_truncation_factor,
                linear_width_factor=linear_width_factor,
                polarization=pure_polarization,
            ),
        )
    if angles.ndim != 1:
        raise ValueError("field profiles require a 1D angle array")
    if angles.size == 0:
        return ()

    _validate_field_inputs(float(angles[0]), layers)
    effective_layers = effective_layers_with_roughness(
        layers,
        step=roughness_step,
        profile=roughness_profile,
        erf_truncation_factor=erf_truncation_factor,
        linear_width_factor=linear_width_factor,
    )

    depths, layer_indices = depth_grid(effective_layers, step)
    if depths.size == 0:
        return tuple(
            FieldProfile(
                depth=depths,
                electric_field=np.array([], dtype=complex),
                intensity=np.array([], dtype=float),
                layer_index=layer_indices,
            )
            for _ in angles
        )

    downward, upward = _transfer_matrix_field_amplitudes_sharp_batched(
        angles,
        energy_ev,
        effective_layers,
        polarization=pure_polarization,
    )
    wavelength = energy_to_wavelength(energy_ev)
    kz = kz_in_layers(angles, wavelength, [layer.n for layer in effective_layers])
    starts = _finite_layer_start_depths(effective_layers)

    sampled_layers = layer_indices
    local_depth = depths - starts[sampled_layers]
    phase = kz[sampled_layers] * local_depth[:, np.newaxis]
    down_field = downward[sampled_layers] * np.exp(1j * phase)
    up_field = upward[sampled_layers] * np.exp(-1j * phase)
    electric_field_by_depth = down_field + up_field
    k0 = 2.0 * np.pi / wavelength
    n_sampled = np.asarray([layer.n for layer in effective_layers], dtype=complex)[sampled_layers, np.newaxis]
    kz_sampled = kz[sampled_layers]
    intensity_by_depth = _field_intensity(
        down_field,
        up_field,
        kz_sampled,
        n_sampled,
        k0,
        pure_polarization,
    )

    return tuple(
        FieldProfile(
            depth=depths,
            electric_field=electric_field_by_depth[:, angle_index],
            intensity=intensity_by_depth[:, angle_index],
            layer_index=layer_indices,
        )
        for angle_index in range(angles.size)
    )


def transfer_matrix_electric_field_profile(
    angle_deg: float,
    energy_ev: float,
    layers: Sequence[Layer],
    step: float = 1.0,
    roughness_step: float | Sequence[float] = 1.0,
    roughness_profile: Literal["erf", "linear"] = "erf",
    erf_truncation_factor: float = 4.0,
    linear_width_factor: float = sqrt(3.0),
    polarization: Polarization = "s",
) -> FieldProfile:
    """Return a transfer-matrix electric-field profile.

    Nonzero roughness is represented by thin effective layers. The profile is
    sampled through the effective finite stack.
    """

    _validate_field_inputs(angle_deg, layers)
    s_weight, p_weight = polarization_weights(polarization)
    if s_weight and p_weight:
        s_profile = transfer_matrix_electric_field_profile(
            angle_deg,
            energy_ev,
            layers,
            step=step,
            roughness_step=roughness_step,
            roughness_profile=roughness_profile,
            erf_truncation_factor=erf_truncation_factor,
            linear_width_factor=linear_width_factor,
            polarization="s",
        )
        p_profile = transfer_matrix_electric_field_profile(
            angle_deg,
            energy_ev,
            layers,
            step=step,
            roughness_step=roughness_step,
            roughness_profile=roughness_profile,
            erf_truncation_factor=erf_truncation_factor,
            linear_width_factor=linear_width_factor,
            polarization="p",
        )
        return FieldProfile(
            depth=s_profile.depth,
            electric_field=s_profile.electric_field,
            intensity=s_weight * s_profile.intensity + p_weight * p_profile.intensity,
            layer_index=s_profile.layer_index,
        )
    pure_polarization = "s" if s_weight else "p"
    effective_layers = effective_layers_with_roughness(
        layers,
        step=roughness_step,
        profile=roughness_profile,
        erf_truncation_factor=erf_truncation_factor,
        linear_width_factor=linear_width_factor,
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
        polarization=pure_polarization,
    )
    wavelength = energy_to_wavelength(energy_ev)
    kz = kz_in_layers(angle_deg, wavelength, [layer.n for layer in effective_layers])
    starts = _finite_layer_start_depths(effective_layers)

    electric_field = np.empty(depths.shape, dtype=complex)
    down_field = np.empty(depths.shape, dtype=complex)
    up_field = np.empty(depths.shape, dtype=complex)
    kz_sampled = np.empty(depths.shape, dtype=complex)
    for index in range(1, len(effective_layers) - 1):
        mask = layer_indices == index
        local_depth = depths[mask] - starts[index]
        down_field[mask] = downward[index] * np.exp(1j * kz[index] * local_depth)
        up_field[mask] = upward[index] * np.exp(-1j * kz[index] * local_depth)
        electric_field[mask] = down_field[mask] + up_field[mask]
        kz_sampled[mask] = kz[index]
    k0 = 2.0 * np.pi / wavelength
    n_sampled = np.asarray([layer.n for layer in effective_layers], dtype=complex)[layer_indices]

    return FieldProfile(
        depth=depths,
        electric_field=electric_field,
        intensity=_field_intensity(
            down_field,
            up_field,
            kz_sampled,
            n_sampled,
            k0,
            pure_polarization,
        ),
        layer_index=layer_indices,
    )


def effective_layers_with_roughness(
    layers: Sequence[Layer],
    step: float | Sequence[float] = 1.0,
    profile: Literal["erf", "linear"] = "erf",
    erf_truncation_factor: float = 4.0,
    linear_width_factor: float = sqrt(3.0),
) -> list[Layer]:
    """Return a sharp-interface effective stack with graded rough interfaces.

    ``step`` may be a scalar slice thickness used for every finite layer, or a
    sequence with one slice thickness per finite layer.
    """

    _validate_field_inputs(0.0, layers)
    finite_steps = _roughness_steps_for_finite_layers(layers, step)
    if erf_truncation_factor <= 0:
        raise ValueError("erf_truncation_factor must be positive")
    if linear_width_factor <= 0:
        raise ValueError("linear_width_factor must be positive")
    if profile not in {"erf", "linear"}:
        raise ValueError("profile must be 'erf' or 'linear'")
    if not _has_roughness(layers):
        return list(layers)

    total_thickness = sum(layer.thickness for layer in layers[1:-1])
    if total_thickness <= 0:
        return [layers[0], layers[-1]]

    edges = _finite_layer_edges(layers, finite_steps)
    centers = 0.5 * (edges[:-1] + edges[1:])

    boundaries = _nominal_boundaries(layers)
    finite_layers: list[Layer] = []
    for left, right, center in zip(edges[:-1], edges[1:], centers):
        delta, beta = _graded_delta_beta_at_depth(
            center,
            layers,
            boundaries,
            profile,
            erf_truncation_factor,
            linear_width_factor,
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


def _roughness_steps_for_finite_layers(
    layers: Sequence[Layer],
    step: float | Sequence[float],
) -> np.ndarray:
    finite_count = max(0, len(layers) - 2)
    if np.asarray(step).ndim == 0:
        steps = np.full(finite_count, float(step), dtype=float)
    else:
        steps = np.asarray(step, dtype=float)
        if len(steps) != finite_count:
            raise ValueError("step sequence must have one value per finite layer")
    if np.any(steps <= 0) or not np.all(np.isfinite(steps)):
        raise ValueError("step values must be positive and finite")
    return steps


def _finite_layer_edges(layers: Sequence[Layer], steps: np.ndarray) -> np.ndarray:
    edges: list[float] = []
    current_depth = 0.0
    for layer, step in zip(layers[1:-1], steps):
        n_slices = max(1, int(np.ceil(layer.thickness / step)))
        local_edges = np.linspace(
            current_depth,
            current_depth + layer.thickness,
            n_slices + 1,
        )
        if edges:
            edges.extend(float(value) for value in local_edges[1:])
        else:
            edges.extend(float(value) for value in local_edges)
        current_depth += layer.thickness
    return np.asarray(edges, dtype=float)


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


def _interface_matrix(y_top: complex, y_bottom: complex) -> np.ndarray:
    m00, m01, m10, m11 = _interface_matrix_elements(y_top, y_bottom)
    return np.array(
        [
            [m00, m01],
            [m10, m11],
        ],
        dtype=complex,
    )


def _interface_matrix_elements(
    y_top: complex | np.ndarray,
    y_bottom: complex | np.ndarray,
) -> tuple[
    complex | np.ndarray,
    complex | np.ndarray,
    complex | np.ndarray,
    complex | np.ndarray,
]:
    ratio = y_top / y_bottom
    same = 0.5 * (1.0 + ratio)
    opposite = 0.5 * (1.0 - ratio)
    return same, opposite, opposite, same


def _admittance(
    kz: complex | np.ndarray,
    n: complex | np.ndarray,
    polarization: Literal["s", "p"],
) -> complex | np.ndarray:
    if polarization == "s":
        return kz
    if polarization == "p":
        return kz / (n**2)
    raise ValueError("polarization must be 's' or 'p'")


def _field_intensity(
    down_field: np.ndarray,
    up_field: np.ndarray,
    kz: np.ndarray,
    n: np.ndarray,
    k0: float,
    polarization: Literal["s", "p"],
) -> np.ndarray:
    if polarization == "s":
        return np.abs(down_field + up_field) ** 2
    if polarization != "p":
        raise ValueError("polarization must be 's' or 'p'")
    sin_phi = kz / (k0 * n)
    cos_phi = np.sqrt(1.0 - sin_phi**2)
    return (
        np.abs((down_field - up_field) * sin_phi) ** 2
        + np.abs((down_field + up_field) * cos_phi) ** 2
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
    profile: Literal["erf", "linear"],
    erf_truncation_factor: float,
    linear_width_factor: float,
) -> tuple[float, float]:
    nearest = int(np.argmin(np.abs(boundaries - depth)))
    sigma = layers[nearest + 1].roughness
    active_width_factor = (
        erf_truncation_factor if profile == "erf" else linear_width_factor
    )

    if sigma > 0 and abs(depth - boundaries[nearest]) <= active_width_factor * sigma:
        above = layers[nearest]
        below = layers[nearest + 1]
        fraction = _roughness_fraction(
            depth - boundaries[nearest],
            sigma,
            profile,
            linear_width_factor,
        )
        delta = (1.0 - fraction) * above.delta + fraction * below.delta
        beta = (1.0 - fraction) * above.beta + fraction * below.beta
        return float(delta), float(beta)

    nominal_index = int(np.searchsorted(boundaries[1:], depth, side="right")) + 1
    nominal_index = min(max(nominal_index, 1), len(layers) - 2)
    return float(layers[nominal_index].delta), float(layers[nominal_index].beta)


def _roughness_fraction(
    distance: float,
    sigma: float,
    profile: Literal["erf", "linear"],
    linear_width_factor: float,
) -> float:
    if profile == "erf":
        return 0.5 * (1.0 + erf(distance / (sqrt(2.0) * sigma)))

    half_width = linear_width_factor * sigma
    return float(np.clip(0.5 + 0.5 * distance / half_width, 0.0, 1.0))


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

__all__ = [
    "FieldProfile",
    "depth_grid",
    "effective_layers_with_roughness",
    "electric_field_profile",
    "layer_field_amplitudes",
    "parratt_reflection_amplitudes",
    "transfer_matrix_electric_field_profile",
    "transfer_matrix_electric_field_profiles",
    "transfer_matrix_field_amplitudes",
    "transfer_matrix_reflectivity",
    "transfer_matrix_reflectivity_array",
    "transfer_matrix_reflection_amplitude",
]
