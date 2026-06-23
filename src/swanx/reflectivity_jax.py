"""Experimental JAX backend for the low-level Parratt reflectivity core."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError as error:  # pragma: no cover - exercised only without JAX
    raise ImportError(
        "JAX is required for swanx.reflectivity_jax; install the optional JAX "
        "dependencies before importing this experimental backend"
    ) from error

from .constants import HC_EV_ANGSTROM
from .layers import Layer

jax.config.update("jax_enable_x64", True)


def layer_arrays_from_layers(
    layers: Sequence[Layer],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed-shape layer arrays for the JAX Parratt backend."""

    if len(layers) < 2:
        raise ValueError("at least two layers are required")
    thicknesses = np.asarray([layer.thickness for layer in layers], dtype=float)
    deltas = np.asarray([layer.delta for layer in layers], dtype=float)
    betas = np.asarray([layer.beta for layer in layers], dtype=float)
    roughnesses = np.asarray([layer.roughness for layer in layers], dtype=float)
    if np.any(thicknesses < 0) or not np.all(np.isfinite(thicknesses)):
        raise ValueError("layer thickness values must be non-negative and finite")
    if np.any(roughnesses < 0) or not np.all(np.isfinite(roughnesses)):
        raise ValueError("layer roughness values must be non-negative and finite")
    return thicknesses, deltas, betas, roughnesses


def parratt_reflectivity_jax(
    angle_deg: jax.Array,
    energy_ev: float,
    thicknesses: jax.Array,
    deltas: jax.Array,
    betas: jax.Array,
    roughnesses: jax.Array,
) -> jax.Array:
    """Return reflectivity from a JAX implementation of Parratt recursion.

    The traced inputs are expected to have fixed shapes. ``angle_deg`` is a
    one-dimensional angle array and the layer arrays all have shape
    ``(n_layers,)``.
    """

    angles = jnp.asarray(angle_deg, dtype=jnp.float64)
    thickness_array = jnp.asarray(thicknesses, dtype=jnp.float64)
    delta_array = jnp.asarray(deltas, dtype=jnp.float64)
    beta_array = jnp.asarray(betas, dtype=jnp.float64)
    roughness_array = jnp.asarray(roughnesses, dtype=jnp.float64)

    wavelength = HC_EV_ANGSTROM / energy_ev
    theta = jnp.deg2rad(angles)
    k0 = 2.0 * jnp.pi / wavelength
    refractive_index = 1.0 - delta_array + 1j * beta_array
    kz = k0 * jnp.sqrt(
        refractive_index[:, jnp.newaxis] ** 2
        - jnp.cos(theta)[jnp.newaxis, :] ** 2
    )

    def step(reflection: jax.Array, index: jax.Array) -> tuple[jax.Array, None]:
        kz_top = kz[index]
        kz_bottom = kz[index + 1]
        r_interface = (kz_top - kz_bottom) / (kz_top + kz_bottom)
        r_interface = r_interface * jnp.exp(
            -2.0 * kz_top * kz_bottom * roughness_array[index + 1] ** 2
        )
        phase = jnp.exp(2.0j * kz_bottom * thickness_array[index + 1])
        next_reflection = (r_interface + reflection * phase) / (
            1.0 + r_interface * reflection * phase
        )
        return next_reflection, None

    layer_indices = jnp.arange(kz.shape[0] - 2, -1, -1)
    initial = jnp.zeros_like(kz[-1])
    amplitude, _ = jax.lax.scan(step, initial, layer_indices)
    return jnp.abs(amplitude) ** 2


jitted_parratt_reflectivity = jax.jit(parratt_reflectivity_jax)


def parratt_reflection_amplitudes_jax(
    angle_deg: jax.Array,
    energy_ev: float,
    thicknesses: jax.Array,
    deltas: jax.Array,
    betas: jax.Array,
    roughnesses: jax.Array,
) -> jax.Array:
    """Return Parratt reflection amplitudes at each layer top.

    The returned array has shape ``(n_layers, n_angles)``. The last substrate
    layer has zero upward reflection amplitude by construction.
    """

    kz = kz_in_layers_jax(angle_deg, energy_ev, deltas, betas)
    thickness_array = jnp.asarray(thicknesses, dtype=jnp.float64)
    roughness_array = jnp.asarray(roughnesses, dtype=jnp.float64)

    def step(reflection: jax.Array, index: jax.Array) -> tuple[jax.Array, jax.Array]:
        kz_top = kz[index]
        kz_bottom = kz[index + 1]
        r_interface = (kz_top - kz_bottom) / (kz_top + kz_bottom)
        r_interface = r_interface * jnp.exp(
            -2.0 * kz_top * kz_bottom * roughness_array[index + 1] ** 2
        )
        phase = jnp.exp(2.0j * kz_bottom * thickness_array[index + 1])
        next_reflection = (r_interface + reflection * phase) / (
            1.0 + r_interface * reflection * phase
        )
        return next_reflection, next_reflection

    layer_indices = jnp.arange(kz.shape[0] - 2, -1, -1)
    initial = jnp.zeros_like(kz[-1])
    _, scanned = jax.lax.scan(step, initial, layer_indices)
    amplitudes = jnp.zeros_like(kz)
    return amplitudes.at[layer_indices].set(scanned)


def layer_field_amplitudes_jax(
    angle_deg: jax.Array,
    energy_ev: float,
    thicknesses: jax.Array,
    deltas: jax.Array,
    betas: jax.Array,
    roughnesses: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return downward and upward field amplitudes at each layer top."""

    kz = kz_in_layers_jax(angle_deg, energy_ev, deltas, betas)
    thickness_array = jnp.asarray(thicknesses, dtype=jnp.float64)
    reflection = parratt_reflection_amplitudes_jax(
        angle_deg,
        energy_ev,
        thickness_array,
        deltas,
        betas,
        roughnesses,
    )
    downward0 = jnp.ones_like(kz[0])
    upward0 = reflection[0]

    def step(
        state: tuple[jax.Array, jax.Array],
        index: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        downward, upward = state
        bottom_field = (
            downward * jnp.exp(1.0j * kz[index] * thickness_array[index])
            + upward * jnp.exp(-1.0j * kz[index] * thickness_array[index])
        )
        next_downward = bottom_field / (1.0 + reflection[index + 1])
        next_upward = reflection[index + 1] * next_downward
        return (next_downward, next_upward), (next_downward, next_upward)

    layer_indices = jnp.arange(0, kz.shape[0] - 1)
    _, (downward_rest, upward_rest) = jax.lax.scan(
        step,
        (downward0, upward0),
        layer_indices,
    )
    downward = jnp.concatenate((downward0[jnp.newaxis, :], downward_rest), axis=0)
    upward = jnp.concatenate((upward0[jnp.newaxis, :], upward_rest), axis=0)
    return downward, upward


def electric_field_intensity_jax(
    angle_deg: jax.Array,
    energy_ev: float,
    thicknesses: jax.Array,
    deltas: jax.Array,
    betas: jax.Array,
    roughnesses: jax.Array,
    depth: jax.Array,
    layer_index: jax.Array,
) -> jax.Array:
    """Return field intensity on a fixed depth grid.

    The returned array has shape ``(n_depth, n_angles)``.
    """

    depth_array = jnp.asarray(depth, dtype=jnp.float64)
    sampled_layers = jnp.asarray(layer_index, dtype=jnp.int32)
    kz = kz_in_layers_jax(angle_deg, energy_ev, deltas, betas)
    downward, upward = layer_field_amplitudes_jax(
        angle_deg,
        energy_ev,
        thicknesses,
        deltas,
        betas,
        roughnesses,
    )
    starts = finite_layer_start_depths_jax(thicknesses)
    local_depth = depth_array - starts[sampled_layers]
    phase = kz[sampled_layers] * local_depth[:, jnp.newaxis]
    electric_field = (
        downward[sampled_layers] * jnp.exp(1.0j * phase)
        + upward[sampled_layers] * jnp.exp(-1.0j * phase)
    )
    return jnp.abs(electric_field) ** 2


jitted_electric_field_intensity = jax.jit(electric_field_intensity_jax)


def transfer_matrix_field_amplitudes_jax(
    angle_deg: jax.Array,
    energy_ev: float,
    thicknesses: jax.Array,
    deltas: jax.Array,
    betas: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return transfer-matrix amplitudes for a sharp effective stack.

    This mirrors ``fields._transfer_matrix_field_amplitudes_sharp_batched`` for
    fixed-shape arrays. Roughness should be represented by precomputing the
    same effective stack used by the NumPy backend before entering JIT.
    """

    kz = kz_in_layers_jax(angle_deg, energy_ev, deltas, betas)
    thickness_array = jnp.asarray(thicknesses, dtype=jnp.float64)

    total0 = (
        jnp.ones_like(kz[0]),
        jnp.zeros_like(kz[0]),
        jnp.zeros_like(kz[0]),
        jnp.ones_like(kz[0]),
    )

    def total_step(
        total: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        index: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], None]:
        total_00, total_01, total_10, total_11 = total
        m00, m01, m10, m11 = _interface_matrix_elements_jax(kz[index], kz[index + 1])
        p = jnp.exp(1.0j * kz[index] * thickness_array[index])
        q = jnp.exp(-1.0j * kz[index] * thickness_array[index])
        next_total = (
            m00 * p * total_00 + m01 * q * total_10,
            m00 * p * total_01 + m01 * q * total_11,
            m10 * p * total_00 + m11 * q * total_10,
            m10 * p * total_01 + m11 * q * total_11,
        )
        return next_total, None

    layer_indices = jnp.arange(0, kz.shape[0] - 1)
    _, _, total_10, total_11 = jax.lax.scan(
        total_step,
        total0,
        layer_indices,
    )[0]
    reflection = -total_10 / total_11

    downward0 = jnp.ones_like(kz[0])
    upward0 = reflection

    def amplitude_step(
        state: tuple[jax.Array, jax.Array],
        index: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        downward, upward = state
        m00, m01, m10, m11 = _interface_matrix_elements_jax(kz[index], kz[index + 1])
        p = jnp.exp(1.0j * kz[index] * thickness_array[index])
        q = jnp.exp(-1.0j * kz[index] * thickness_array[index])
        propagated_down = p * downward
        propagated_up = q * upward
        next_downward = m00 * propagated_down + m01 * propagated_up
        next_upward = m10 * propagated_down + m11 * propagated_up
        return (next_downward, next_upward), (next_downward, next_upward)

    _, (downward_rest, upward_rest) = jax.lax.scan(
        amplitude_step,
        (downward0, upward0),
        layer_indices,
    )
    downward = jnp.concatenate((downward0[jnp.newaxis, :], downward_rest), axis=0)
    upward = jnp.concatenate((upward0[jnp.newaxis, :], upward_rest), axis=0)
    return downward, upward


def transfer_matrix_reflectivity_jax(
    angle_deg: jax.Array,
    energy_ev: float,
    thicknesses: jax.Array,
    deltas: jax.Array,
    betas: jax.Array,
) -> jax.Array:
    """Return transfer-matrix reflectivity for a sharp effective stack."""

    downward, upward = transfer_matrix_field_amplitudes_jax(
        angle_deg,
        energy_ev,
        thicknesses,
        deltas,
        betas,
    )
    return jnp.abs(upward[0] / downward[0]) ** 2


jitted_transfer_matrix_reflectivity = jax.jit(transfer_matrix_reflectivity_jax)


def transfer_matrix_field_intensity_jax(
    angle_deg: jax.Array,
    energy_ev: float,
    thicknesses: jax.Array,
    deltas: jax.Array,
    betas: jax.Array,
    depth: jax.Array,
    layer_index: jax.Array,
) -> jax.Array:
    """Return transfer-matrix field intensity on a fixed effective depth grid."""

    depth_array = jnp.asarray(depth, dtype=jnp.float64)
    sampled_layers = jnp.asarray(layer_index, dtype=jnp.int32)
    kz = kz_in_layers_jax(angle_deg, energy_ev, deltas, betas)
    downward, upward = transfer_matrix_field_amplitudes_jax(
        angle_deg,
        energy_ev,
        thicknesses,
        deltas,
        betas,
    )
    starts = finite_layer_start_depths_jax(thicknesses)
    local_depth = depth_array - starts[sampled_layers]
    phase = kz[sampled_layers] * local_depth[:, jnp.newaxis]
    electric_field = (
        downward[sampled_layers] * jnp.exp(1.0j * phase)
        + upward[sampled_layers] * jnp.exp(-1.0j * phase)
    )
    return jnp.abs(electric_field) ** 2


jitted_transfer_matrix_field_intensity = jax.jit(transfer_matrix_field_intensity_jax)


def attenuation_factor_jax(
    depth: jax.Array,
    attenuation_length: jax.Array,
    emission_angle_deg: float = 0.0,
) -> jax.Array:
    """Return electron attenuation from each depth to the surface."""

    depth_array = jnp.asarray(depth, dtype=jnp.float64)
    attenuation_length_array = jnp.asarray(attenuation_length, dtype=jnp.float64)
    cos_alpha = jnp.cos(jnp.deg2rad(emission_angle_deg))
    coefficient = 1.0 / (attenuation_length_array * cos_alpha)
    dx = jnp.diff(depth_array)
    areas = 0.5 * (coefficient[:-1] + coefficient[1:]) * dx
    optical_depth = jnp.concatenate(
        (jnp.zeros((1,), dtype=jnp.float64), jnp.cumsum(areas))
    )
    return jnp.exp(-optical_depth)


def raw_xps_intensity_jax(
    field_intensity: jax.Array,
    depth: jax.Array,
    concentration: jax.Array,
    attenuation_length: jax.Array,
    emission_angle_deg: float = 0.0,
) -> jax.Array:
    """Integrate concentration, field intensity, and attenuation over depth."""

    depth_array = jnp.asarray(depth, dtype=jnp.float64)
    concentration_array = jnp.asarray(concentration, dtype=jnp.float64)
    attenuation = attenuation_factor_jax(
        depth_array,
        attenuation_length,
        emission_angle_deg=emission_angle_deg,
    )
    integrand = concentration_array[:, jnp.newaxis] * field_intensity * attenuation[:, jnp.newaxis]
    dx = jnp.diff(depth_array)
    areas = 0.5 * (integrand[:-1] + integrand[1:]) * dx[:, jnp.newaxis]
    return jnp.sum(areas, axis=0)


def normalized_rocking_curve_jax(
    angle_deg: jax.Array,
    energy_ev: float,
    thicknesses: jax.Array,
    deltas: jax.Array,
    betas: jax.Array,
    roughnesses: jax.Array,
    depth: jax.Array,
    layer_index: jax.Array,
    concentration: jax.Array,
    attenuation_length: jax.Array,
    emission_angle_deg: float,
    offpeak_mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return normalized, raw, and normalization values for one RC."""

    del roughnesses
    intensity = transfer_matrix_field_intensity_jax(
        angle_deg,
        energy_ev,
        thicknesses,
        deltas,
        betas,
        depth,
        layer_index,
    )
    raw = raw_xps_intensity_jax(
        intensity,
        depth,
        concentration,
        attenuation_length,
        emission_angle_deg=emission_angle_deg,
    )
    mask = jnp.asarray(offpeak_mask, dtype=bool)
    normalization = jnp.sum(jnp.where(mask, raw, 0.0)) / jnp.sum(mask)
    return raw / normalization, raw, normalization


jitted_normalized_rocking_curve = jax.jit(normalized_rocking_curve_jax)


def normalized_rocking_curve_from_field_jax(
    field_intensity: jax.Array,
    depth: jax.Array,
    concentration: jax.Array,
    attenuation_length: jax.Array,
    emission_angle_deg: float,
    offpeak_mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return one normalized RC from a precomputed field-intensity grid."""

    raw = raw_xps_intensity_jax(
        field_intensity,
        depth,
        concentration,
        attenuation_length,
        emission_angle_deg=emission_angle_deg,
    )
    mask = jnp.asarray(offpeak_mask, dtype=bool)
    normalization = jnp.sum(jnp.where(mask, raw, 0.0)) / jnp.sum(mask)
    return raw / normalization, raw, normalization


jitted_normalized_rocking_curve_from_field = jax.jit(
    normalized_rocking_curve_from_field_jax
)


def reflectivity_mse_loss(
    thicknesses: jax.Array,
    angle_deg: jax.Array,
    energy_ev: float,
    deltas: jax.Array,
    betas: jax.Array,
    roughnesses: jax.Array,
    target_reflectivity: jax.Array,
) -> jax.Array:
    """Return a scalar MSE loss for testing differentiability."""

    simulated = parratt_reflectivity_jax(
        angle_deg,
        energy_ev,
        thicknesses,
        deltas,
        betas,
        roughnesses,
    )
    residual = simulated - jnp.asarray(target_reflectivity, dtype=jnp.float64)
    return jnp.mean(residual**2)


jitted_value_and_grad_reflectivity_loss = jax.jit(
    jax.value_and_grad(reflectivity_mse_loss)
)


def rocking_curve_mse_loss(
    thicknesses: jax.Array,
    angle_deg: jax.Array,
    energy_ev: float,
    deltas: jax.Array,
    betas: jax.Array,
    roughnesses: jax.Array,
    depth: jax.Array,
    layer_index: jax.Array,
    concentration: jax.Array,
    attenuation_length: jax.Array,
    emission_angle_deg: float,
    offpeak_mask: jax.Array,
    target_intensity: jax.Array,
) -> jax.Array:
    """Return a scalar normalized-RC MSE loss for gradient checks."""

    simulated, _, _ = normalized_rocking_curve_jax(
        angle_deg,
        energy_ev,
        thicknesses,
        deltas,
        betas,
        roughnesses,
        depth,
        layer_index,
        concentration,
        attenuation_length,
        emission_angle_deg,
        offpeak_mask,
    )
    residual = simulated - jnp.asarray(target_intensity, dtype=jnp.float64)
    return jnp.mean(residual**2)


jitted_value_and_grad_rocking_curve_loss = jax.jit(
    jax.value_and_grad(rocking_curve_mse_loss)
)


def kz_in_layers_jax(
    angle_deg: jax.Array,
    energy_ev: float,
    deltas: jax.Array,
    betas: jax.Array,
) -> jax.Array:
    """Return kz with shape ``(n_layers, n_angles)`` for fixed JAX arrays."""

    angles = jnp.asarray(angle_deg, dtype=jnp.float64)
    delta_array = jnp.asarray(deltas, dtype=jnp.float64)
    beta_array = jnp.asarray(betas, dtype=jnp.float64)
    wavelength = HC_EV_ANGSTROM / energy_ev
    theta = jnp.deg2rad(angles)
    k0 = 2.0 * jnp.pi / wavelength
    refractive_index = 1.0 - delta_array + 1j * beta_array
    return k0 * jnp.sqrt(
        refractive_index[:, jnp.newaxis] ** 2
        - jnp.cos(theta)[jnp.newaxis, :] ** 2
    )


def finite_layer_start_depths_jax(thicknesses: jax.Array) -> jax.Array:
    """Return start depths for each layer using fixed-shape thickness arrays."""

    thickness_array = jnp.asarray(thicknesses, dtype=jnp.float64)
    finite_starts = jnp.cumsum(thickness_array[:-2])
    return jnp.concatenate(
        (
            jnp.zeros((1,), dtype=jnp.float64),
            finite_starts,
            jnp.array([jnp.sum(thickness_array[1:-1])], dtype=jnp.float64),
        )
    )


def _interface_matrix_elements_jax(
    kz_top: jax.Array,
    kz_bottom: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    ratio = kz_top / kz_bottom
    same = 0.5 * (1.0 + ratio)
    opposite = 0.5 * (1.0 - ratio)
    return same, opposite, opposite, same
