"""Parratt recursion for s-polarized x-ray reflectivity."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..constants import HC_EV_ANGSTROM
from ..layers import Layer


def energy_to_wavelength(energy_ev: float) -> float:
    """Convert photon energy in eV to wavelength in Angstrom."""

    if energy_ev <= 0:
        raise ValueError("energy_ev must be positive")
    return HC_EV_ANGSTROM / energy_ev


def kz_in_layers(
    angle_deg: float | np.ndarray,
    wavelength: float,
    refractive_indices: Sequence[complex],
) -> np.ndarray:
    """Return kz for each layer at grazing incidence angles.

    The returned array has shape ``(n_layers, n_angles)`` for array input.
    For scalar input, it has shape ``(n_layers,)``.
    """

    if wavelength <= 0:
        raise ValueError("wavelength must be positive")
    if len(refractive_indices) < 2:
        raise ValueError("at least two refractive indices are required")

    angles = np.asarray(angle_deg, dtype=float)
    angle_values = np.atleast_1d(angles)
    theta = np.deg2rad(angle_values)
    k0 = 2.0 * np.pi / wavelength
    n = np.asarray(refractive_indices, dtype=complex)

    kz = k0 * np.sqrt(n[:, np.newaxis] ** 2 - np.cos(theta)[np.newaxis, :] ** 2)
    if angles.ndim == 0:
        return kz[:, 0]
    return kz


def fresnel_r_s(kz_top: complex | np.ndarray, kz_bottom: complex | np.ndarray) -> complex | np.ndarray:
    """Return the s-polarized Fresnel reflection amplitude."""

    return (kz_top - kz_bottom) / (kz_top + kz_bottom)


def apply_roughness(
    reflection_amplitude: complex | np.ndarray,
    kz_top: complex | np.ndarray,
    kz_bottom: complex | np.ndarray,
    roughness: float,
) -> complex | np.ndarray:
    """Apply the Nevot-Croce roughness factor to an interface amplitude."""

    if roughness == 0:
        return reflection_amplitude
    return reflection_amplitude * np.exp(-2.0 * kz_top * kz_bottom * roughness**2)


def parratt_amplitude(angle_deg: float | np.ndarray, energy_ev: float, layers: Sequence[Layer]) -> complex | np.ndarray:
    """Return the complex reflection amplitude from the Parratt recursion."""

    _validate_layers(layers)

    wavelength = energy_to_wavelength(energy_ev)
    refractive_indices = [layer.n for layer in layers]
    kz = kz_in_layers(angle_deg, wavelength, refractive_indices)
    thicknesses = np.asarray([layer.thickness for layer in layers], dtype=float)
    roughnesses = np.asarray([layer.roughness for layer in layers], dtype=float)

    reflection = np.zeros_like(kz[-1], dtype=complex)
    for j in range(len(layers) - 2, -1, -1):
        r_interface = fresnel_r_s(kz[j], kz[j + 1])
        r_interface = apply_roughness(
            r_interface,
            kz[j],
            kz[j + 1],
            roughnesses[j + 1],
        )
        phase = np.exp(2j * kz[j + 1] * thicknesses[j + 1])
        reflection = (r_interface + reflection * phase) / (
            1.0 + r_interface * reflection * phase
        )

    if np.asarray(angle_deg).ndim == 0:
        return complex(reflection)
    return reflection


def parratt_reflectivity(angle_deg: float | np.ndarray, energy_ev: float, layers: Sequence[Layer]) -> float | np.ndarray:
    """Return specular x-ray reflectivity calculated by the Parratt recursion."""

    amplitude = parratt_amplitude(angle_deg, energy_ev, layers)
    reflectivity = np.abs(amplitude) ** 2
    if np.asarray(angle_deg).ndim == 0:
        return float(reflectivity)
    return reflectivity


def _validate_layers(layers: Sequence[Layer]) -> None:
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
    "apply_roughness",
    "energy_to_wavelength",
    "fresnel_r_s",
    "kz_in_layers",
    "parratt_amplitude",
    "parratt_reflectivity",
]