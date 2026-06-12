"""Differentiable JAX forward models for SWXPS."""

from __future__ import annotations

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

from .constants import HC_EV_ANGSTROM


@jax.jit
def reflectivity_forward_jax(
    angles_deg,
    energy_ev,
    thickness,
    delta,
    beta,
    roughness,
):
    """Return differentiable Parratt reflectivity using JAX.

    Parameters are array-like and ordered from vacuum/top medium to substrate.
    The first and last layers are treated as semi-infinite, matching the NumPy
    Parratt implementation. `roughness[j]` is the RMS roughness of layer `j`'s
    upper interface, so the interface between layers `j` and `j + 1` uses
    `roughness[j + 1]`.

    The function is JIT-compiled and uses `jax.lax.scan` for the reverse
    Parratt recursion, so it can be used directly inside differentiable inverse
    problem objectives.
    """

    angles = jnp.asarray(angles_deg, dtype=jnp.float64)
    angle_shape = angles.shape
    angle_values = jnp.ravel(angles)

    energy = jnp.asarray(energy_ev, dtype=jnp.float64)
    thickness = jnp.asarray(thickness, dtype=jnp.float64)
    delta = jnp.asarray(delta, dtype=jnp.float64)
    beta = jnp.asarray(beta, dtype=jnp.float64)
    roughness = jnp.asarray(roughness, dtype=jnp.float64)

    refractive_index = (1.0 - delta) + 1j * beta
    wavelength = HC_EV_ANGSTROM / energy
    k0 = 2.0 * jnp.pi / wavelength
    theta = jnp.deg2rad(angle_values)
    kz = k0 * jnp.sqrt(
        refractive_index[:, jnp.newaxis] ** 2
        - jnp.cos(theta)[jnp.newaxis, :] ** 2
    )

    kz_top = kz[:-1]
    kz_bottom = kz[1:]
    r_fresnel = (kz_top - kz_bottom) / (kz_top + kz_bottom)
    roughness_factor = jnp.exp(
        -2.0 * kz_top * kz_bottom * roughness[1:, jnp.newaxis] ** 2
    )
    r_interface = r_fresnel * roughness_factor
    phase = jnp.exp(2j * kz_bottom * thickness[1:, jnp.newaxis])

    def parratt_step(reflection, interface_terms):
        r_current, phase_current = interface_terms
        numerator = r_current + reflection * phase_current
        denominator = 1.0 + r_current * reflection * phase_current
        return numerator / denominator, None

    initial_reflection = jnp.zeros_like(angle_values, dtype=jnp.complex128)
    reflection, _ = jax.lax.scan(
        parratt_step,
        initial_reflection,
        (jnp.flip(r_interface, axis=0), jnp.flip(phase, axis=0)),
    )

    reflectivity = jnp.real(jnp.abs(reflection) ** 2)
    return jnp.reshape(reflectivity, angle_shape)
