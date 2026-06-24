"""Compatibility exports for the high-level simulation API."""

from .stack.model import SimulationStack, StackLayer, stack_from_layers
from .workflows.simulate import (
    CoreLevelRequest,
    CoreLevelResult,
    ReflectivityRequest,
    ReflectivityResult,
    RockingCurveRequest,
    RockingCurveResult,
    simulate_reflectivity,
    simulate_rocking_curve,
    simulate_rocking_curves,
)

__all__ = [
    "CoreLevelRequest",
    "CoreLevelResult",
    "ReflectivityRequest",
    "ReflectivityResult",
    "RockingCurveRequest",
    "RockingCurveResult",
    "SimulationStack",
    "StackLayer",
    "simulate_reflectivity",
    "simulate_rocking_curve",
    "simulate_rocking_curves",
    "stack_from_layers",
]
