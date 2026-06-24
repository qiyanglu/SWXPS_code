"""Canonical XPS attenuation, intensity, grid, and rocking-curve APIs."""

from .attenuation import attenuation_factor
from .grid import cell_centered_attenuation, integrate_xps_on_grid
from .intensity import (
    graded_layer_property_at_depth,
    integrate_xps_intensity,
    nominal_layer_index_at_depth,
)
from .rocking_curve import RockingCurve, normalized_rocking_curve

_SIMULATION_EXPORTS = {
    "CoreLevelRequest",
    "CoreLevelResult",
    "RockingCurveRequest",
    "RockingCurveResult",
    "simulate_rocking_curve",
    "simulate_rocking_curves",
}


def __getattr__(name: str):
    if name not in _SIMULATION_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .. import simulation

    value = getattr(simulation, name)
    globals()[name] = value
    return value


__all__ = [
    "CoreLevelRequest",
    "CoreLevelResult",
    "RockingCurve",
    "RockingCurveRequest",
    "RockingCurveResult",
    "attenuation_factor",
    "cell_centered_attenuation",
    "graded_layer_property_at_depth",
    "integrate_xps_intensity",
    "integrate_xps_on_grid",
    "nominal_layer_index_at_depth",
    "normalized_rocking_curve",
    "simulate_rocking_curve",
    "simulate_rocking_curves",
]
