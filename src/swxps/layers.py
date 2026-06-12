"""Layer definitions for multilayer x-ray reflectivity."""

from __future__ import annotations

from dataclasses import dataclass


def refractive_index(delta: float = 0.0, beta: float = 0.0) -> complex:
    """Return the x-ray refractive index n = 1 - delta + i beta."""

    return complex(1.0 - delta, beta)


@dataclass(frozen=True)
class Layer:
    """A flat layer with thickness and manually supplied optical constants.

    Thickness and roughness are in Angstrom. Roughness is the RMS roughness of
    the layer's upper interface. The first and last layers are treated as
    semi-infinite by the Parratt calculation, so their thickness values do not
    affect the reflectivity.
    """

    thickness: float
    delta: float = 0.0
    beta: float = 0.0
    roughness: float = 0.0

    @property
    def n(self) -> complex:
        """Complex refractive index n = 1 - delta + i beta."""

        return refractive_index(self.delta, self.beta)


def vacuum() -> Layer:
    """Return a semi-infinite vacuum layer."""

    return Layer(thickness=0.0, delta=0.0, beta=0.0, roughness=0.0)
