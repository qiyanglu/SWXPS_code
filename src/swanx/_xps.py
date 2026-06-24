"""Compatibility shim for the canonical :mod:`swanx.xps` implementation."""

from .xps.attenuation import attenuation_factor
from .xps.intensity import (
    graded_layer_property_at_depth,
    integrate_xps_intensity,
    nominal_layer_index_at_depth,
)
from .xps.rocking_curve import RockingCurve, normalized_rocking_curve

__all__ = [
    "RockingCurve",
    "attenuation_factor",
    "graded_layer_property_at_depth",
    "integrate_xps_intensity",
    "nominal_layer_index_at_depth",
    "normalized_rocking_curve",
]
