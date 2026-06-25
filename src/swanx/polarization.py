"""Polarization parsing helpers shared by simulation backends."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypeAlias

import numpy as np

Polarization: TypeAlias = Literal["s", "p"] | Mapping[str, float]


def polarization_weights(polarization: Polarization) -> tuple[float, float]:
    """Return ``(s_weight, p_weight)`` for a supported polarization setting."""

    if polarization == "s":
        return 1.0, 0.0
    if polarization == "p":
        return 0.0, 1.0
    if isinstance(polarization, Mapping):
        unknown = set(polarization) - {"s", "p"}
        if unknown:
            raise ValueError("polarization mapping keys must be 's' and/or 'p'")
        s_weight = float(polarization.get("s", 0.0))
        p_weight = float(polarization.get("p", 0.0))
        if not np.isfinite(s_weight) or not np.isfinite(p_weight):
            raise ValueError("polarization weights must be finite")
        if s_weight < 0.0 or p_weight < 0.0:
            raise ValueError("polarization weights must be non-negative")
        if s_weight == 0.0 and p_weight == 0.0:
            raise ValueError("at least one polarization weight must be positive")
        if not np.isclose(s_weight + p_weight, 1.0):
            raise ValueError("mixed polarization weights must sum to 1")
        return s_weight, p_weight
    raise ValueError("polarization must be 's', 'p', or {'s': fs, 'p': fp}")


def polarization_code(polarization: Literal["s", "p"]) -> int:
    """Return the numeric code used by JAX kernels."""

    if polarization == "s":
        return 0
    if polarization == "p":
        return 1
    raise ValueError("polarization must be 's' or 'p'")


__all__ = ["Polarization", "polarization_code", "polarization_weights"]
