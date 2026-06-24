"""Frozen public API for standing-wave analysis with X-ray spectroscopy."""

from .diagnostics import (
    compute_parameter_diagnostics,
    plot_correlation_matrix,
    plot_parameter_estimates,
)
from .stack.model import SimulationStack, StackLayer
from .workflows.simulate import (
    CoreLevelRequest,
    ReflectivityRequest,
    RockingCurveRequest,
    simulate_reflectivity,
    simulate_rocking_curves,
)

__all__ = [
    "CoreLevelRequest",
    "ReflectivityRequest",
    "RockingCurveRequest",
    "SimulationStack",
    "StackLayer",
    "compute_parameter_diagnostics",
    "plot_correlation_matrix",
    "plot_parameter_estimates",
    "simulate_reflectivity",
    "simulate_rocking_curves",
]
