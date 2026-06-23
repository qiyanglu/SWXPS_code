"""XPS attenuation, intensity, and rocking-curve APIs."""

from .._xps import (
    RockingCurve,
    attenuation_factor,
    graded_layer_property_at_depth,
    integrate_xps_intensity,
    nominal_layer_index_at_depth,
    normalized_rocking_curve,
)
from ..simulation import (
    CoreLevelRequest,
    CoreLevelResult,
    RockingCurveRequest,
    RockingCurveResult,
    simulate_rocking_curve,
    simulate_rocking_curves,
)
from ..unified_grid import cell_centered_attenuation, integrate_xps_on_grid

__all__ = [name for name in globals() if not name.startswith("_")]
