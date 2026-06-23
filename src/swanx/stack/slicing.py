"""Layer-aware unified grids for roughness, fields, and SW-XPS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


class _LayerLike(Protocol):
    thickness: float


@dataclass(frozen=True)
class LayerSlicingPolicy:
    """Resolution limits for finite nominal layers.

    ``max_slice_thickness`` is in Angstrom. Every positive finite layer gets at
    least ``min_slices`` cells, and no adaptive cell is thicker than the
    configured maximum.
    """

    min_slices: int = 10
    max_slice_thickness: float = 2.0

    def __post_init__(self) -> None:
        if isinstance(self.min_slices, bool) or not isinstance(self.min_slices, int):
            raise ValueError("min_slices must be an integer")
        if self.min_slices < 1:
            raise ValueError("min_slices must be at least 1")
        if (
            not np.isfinite(self.max_slice_thickness)
            or self.max_slice_thickness <= 0
        ):
            raise ValueError("max_slice_thickness must be positive and finite")

    def slice_count(self, thickness: float) -> int:
        """Return the adaptive cell count for one positive finite layer."""

        thickness = _positive_finite_thickness(thickness)
        return max(
            self.min_slices,
            int(np.ceil(thickness / self.max_slice_thickness)),
        )


@dataclass(frozen=True)
class FixedLayerGridPlan:
    """Fixed per-layer cell counts and their declared thickness capacities."""

    slice_counts: tuple[int, ...]
    capacity_thicknesses: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.slice_counts) != len(self.capacity_thicknesses):
            raise ValueError(
                "slice_counts and capacity_thicknesses must have the same length"
            )
        for count in self.slice_counts:
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("slice counts must be positive integers")
        for thickness in self.capacity_thicknesses:
            _positive_finite_thickness(thickness)


@dataclass(frozen=True)
class LayerGrid:
    """One cell-centered grid shared by all discretized forward calculations."""

    edges: np.ndarray
    centers: np.ndarray
    widths: np.ndarray
    nominal_layer_index: np.ndarray
    effective_layer_index: np.ndarray
    slice_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        cell_count = len(self.centers)
        if self.edges.shape != (cell_count + 1,):
            raise ValueError("edges must contain one more value than centers")
        for name, values in (
            ("widths", self.widths),
            ("nominal_layer_index", self.nominal_layer_index),
            ("effective_layer_index", self.effective_layer_index),
        ):
            if values.shape != (cell_count,):
                raise ValueError(f"{name} must contain one value per cell")
        if np.any(self.widths <= 0) or not np.all(np.isfinite(self.widths)):
            raise ValueError("cell widths must be positive and finite")
        if not np.all(np.diff(self.edges) > 0):
            raise ValueError("cell edges must be strictly increasing")
        if not np.all((self.centers > self.edges[:-1]) & (self.centers < self.edges[1:])):
            raise ValueError("cell centers must lie inside their cells")


def fixed_layer_grid_plan(
    capacity_layers: Sequence[_LayerLike],
    policy: LayerSlicingPolicy | None = None,
) -> FixedLayerGridPlan:
    """Create fixed counts from a stack built at fitting capacity thicknesses."""

    policy = LayerSlicingPolicy() if policy is None else policy
    thicknesses = _finite_thicknesses(capacity_layers)
    return FixedLayerGridPlan(
        slice_counts=tuple(policy.slice_count(value) for value in thicknesses),
        capacity_thicknesses=thicknesses,
    )


def adaptive_layer_grid(
    layers: Sequence[_LayerLike],
    policy: LayerSlicingPolicy | None = None,
) -> LayerGrid:
    """Materialize an adaptive grid from the stack's current thicknesses."""

    policy = LayerSlicingPolicy() if policy is None else policy
    thicknesses = _finite_thicknesses(layers)
    counts = tuple(policy.slice_count(value) for value in thicknesses)
    return _materialize_grid(thicknesses, counts)


def fixed_layer_grid(
    layers: Sequence[_LayerLike],
    plan: FixedLayerGridPlan,
) -> LayerGrid:
    """Materialize a fixed-shape grid for one trial stack."""

    thicknesses = _finite_thicknesses(layers)
    if len(thicknesses) != len(plan.slice_counts):
        raise ValueError("trial stack topology does not match the fixed grid plan")
    for index, (thickness, capacity) in enumerate(
        zip(thicknesses, plan.capacity_thicknesses)
    ):
        tolerance = 1.0e-12 * max(1.0, abs(capacity))
        if thickness > capacity + tolerance:
            raise ValueError(
                f"finite layer {index + 1} thickness exceeds grid-plan capacity"
            )
    return _materialize_grid(thicknesses, plan.slice_counts)


def _materialize_grid(
    thicknesses: tuple[float, ...],
    slice_counts: tuple[int, ...],
) -> LayerGrid:
    edges = [0.0]
    nominal_indices: list[int] = []
    current_depth = 0.0
    for nominal_index, (thickness, count) in enumerate(
        zip(thicknesses, slice_counts),
        start=1,
    ):
        local_edges = np.linspace(
            current_depth,
            current_depth + thickness,
            count + 1,
        )
        edges.extend(float(value) for value in local_edges[1:])
        nominal_indices.extend([nominal_index] * count)
        current_depth += thickness

    edge_array = _readonly(np.asarray(edges, dtype=float))
    widths = _readonly(np.diff(edge_array))
    centers = _readonly(0.5 * (edge_array[:-1] + edge_array[1:]))
    nominal_layer_index = _readonly(np.asarray(nominal_indices, dtype=int))
    effective_layer_index = _readonly(
        np.arange(1, len(centers) + 1, dtype=int)
    )
    return LayerGrid(
        edges=edge_array,
        centers=centers,
        widths=widths,
        nominal_layer_index=nominal_layer_index,
        effective_layer_index=effective_layer_index,
        slice_counts=tuple(slice_counts),
    )


def _finite_thicknesses(layers: Sequence[_LayerLike]) -> tuple[float, ...]:
    if len(layers) < 2:
        raise ValueError("a unified grid requires at least two boundary layers")
    return tuple(_positive_finite_thickness(layer.thickness) for layer in layers[1:-1])


def _positive_finite_thickness(thickness: float) -> float:
    thickness = float(thickness)
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError("finite layer thicknesses must be positive and finite")
    return thickness


def _readonly(values: np.ndarray) -> np.ndarray:
    values.setflags(write=False)
    return values

__all__ = [
    "FixedLayerGridPlan",
    "LayerGrid",
    "LayerSlicingPolicy",
    "adaptive_layer_grid",
    "fixed_layer_grid",
    "fixed_layer_grid_plan",
]