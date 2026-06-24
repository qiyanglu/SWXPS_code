"""Utilities for exporting fitted curves and optimized stack information."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from ._fitting import FitSimulation, ReflectivityData, RockingCurveData
from .stack.model import SimulationStack


def save_fit_curve_data_csv(
    path: str | Path,
    reflectivity_data: ReflectivityData | None,
    rocking_curve_data: Sequence[RockingCurveData],
    simulation: FitSimulation,
) -> None:
    """Save experimental and fitted reflectivity/RC curves in one CSV file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    simulated_rc = {
        core.name: core.curve
        for core in simulation.rocking_curves.core_levels
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "dataset",
            "angle_deg",
            "calculation_angle_deg",
            "experimental",
            "fitted",
        ])
        if reflectivity_data is not None and simulation.reflectivity is not None:
            for angle, calc_angle, observed, fitted in zip(
                reflectivity_data.angles,
                simulation.reflectivity.calculation_angle,
                reflectivity_data.reflectivity,
                simulation.reflectivity.reflectivity,
            ):
                writer.writerow([
                    reflectivity_data.name,
                    angle,
                    calc_angle,
                    observed,
                    fitted,
                ])
        for data in rocking_curve_data:
            curve = simulated_rc[data.name]
            for angle, calc_angle, observed, fitted in zip(
                data.angles,
                simulation.rocking_curves.calculation_angle,
                data.intensity,
                curve.intensity,
            ):
                writer.writerow([
                    data.name,
                    angle,
                    calc_angle,
                    observed,
                    fitted,
                ])


def save_optimized_stack_csv(path: str | Path, stack: SimulationStack) -> None:
    """Save layer-by-layer optimized stack material, thickness, and roughness."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "layer_index",
            "material",
            "thickness_A",
            "roughness_A",
            "delta",
            "beta",
        ])
        for index, layer in enumerate(stack.layers):
            writer.writerow([
                index,
                layer.material,
                layer.thickness,
                layer.roughness,
                layer.delta,
                layer.beta,
            ])
