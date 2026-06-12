"""Depth profiles for material and element concentrations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .simulation import SimulationStack
from .xps import graded_layer_property_at_depth


@dataclass(frozen=True)
class StackProfiles:
    """One or more stack properties sampled on a common depth grid."""

    depth: np.ndarray
    profiles: dict[str, np.ndarray]


def sample_stack_property(
    stack: SimulationStack,
    values_by_material: dict[str, float],
    step: float = 1.0,
    width_factor: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one material property through the finite stack depth."""

    depth = stack_depth_grid(stack, step=step)
    values_by_layer = [
        float(values_by_material.get(material, 0.0))
        for material in stack.materials
    ]
    values = graded_layer_property_at_depth(
        stack.optical_layers,
        values_by_layer,
        depth,
        width_factor=width_factor,
    )
    return depth, values


def sample_concentration_profiles(
    stack: SimulationStack,
    concentration_by_name: dict[str, dict[str, float]],
    step: float = 1.0,
    width_factor: float = 4.0,
) -> StackProfiles:
    """Sample multiple named concentration profiles through a stack."""

    depth = stack_depth_grid(stack, step=step)
    profiles: dict[str, np.ndarray] = {}
    for name, values_by_material in concentration_by_name.items():
        values_by_layer = [
            float(values_by_material.get(material, 0.0))
            for material in stack.materials
        ]
        profiles[name] = graded_layer_property_at_depth(
            stack.optical_layers,
            values_by_layer,
            depth,
            width_factor=width_factor,
        )
    return StackProfiles(depth=depth, profiles=profiles)


def stack_depth_grid(stack: SimulationStack, step: float = 1.0) -> np.ndarray:
    """Return a depth grid spanning the finite layers of a SimulationStack."""

    if step <= 0:
        raise ValueError("step must be positive")

    total_thickness = sum(layer.thickness for layer in stack.layers[1:-1])
    if total_thickness <= 0:
        return np.array([], dtype=float)

    n_points = max(2, int(np.ceil(total_thickness / step)) + 1)
    return np.linspace(0.0, total_thickness, n_points)
