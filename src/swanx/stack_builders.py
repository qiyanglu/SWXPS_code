"""Declarative builders for common multilayer stack definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .optical_constants import layer_from_file
from .stack.model import SimulationStack, StackLayer

ScalarOrParameter = float | str


@dataclass(frozen=True)
class LayerTemplate:
    """A reusable layer recipe with numeric values or parameter names."""

    material: str
    thickness: ScalarOrParameter
    roughness: ScalarOrParameter = 0.0
    optical_constants_file: str | Path | None = None
    delta: ScalarOrParameter | None = None
    beta: ScalarOrParameter | None = None

    @classmethod
    def vacuum(cls) -> LayerTemplate:
        """Return the required semi-infinite vacuum layer."""

        return cls("vacuum", thickness=0.0, roughness=0.0, delta=0.0, beta=0.0)

    @classmethod
    def from_file(
        cls,
        material: str,
        optical_constants_file: str | Path,
        thickness: ScalarOrParameter,
        roughness: ScalarOrParameter = 0.0,
    ) -> LayerTemplate:
        """Return a layer recipe backed by an optical-constants table."""

        return cls(
            material=material,
            optical_constants_file=optical_constants_file,
            thickness=thickness,
            roughness=roughness,
        )

    @classmethod
    def from_constants(
        cls,
        material: str,
        thickness: ScalarOrParameter,
        delta: ScalarOrParameter,
        beta: ScalarOrParameter,
        roughness: ScalarOrParameter = 0.0,
    ) -> LayerTemplate:
        """Return a layer recipe with explicitly supplied optical constants."""

        return cls(
            material=material,
            thickness=thickness,
            roughness=roughness,
            delta=delta,
            beta=beta,
        )

    def build(
        self,
        values: dict[str, float],
        energy_ev: float,
        base_dir: str | Path = ".",
    ) -> StackLayer:
        """Build one material-labeled stack layer."""

        thickness = _resolve_value(self.thickness, values)
        roughness = _resolve_value(self.roughness, values)
        if self.optical_constants_file is not None:
            layer = layer_from_file(
                _resolve_path(self.optical_constants_file, base_dir),
                energy_ev=energy_ev,
                thickness=thickness,
                roughness=roughness,
            )
            return StackLayer(
                material=self.material,
                thickness=layer.thickness,
                delta=layer.delta,
                beta=layer.beta,
                roughness=layer.roughness,
            )
        if self.delta is None or self.beta is None:
            raise ValueError("delta and beta are required when no optical file is supplied")
        return StackLayer(
            material=self.material,
            thickness=thickness,
            delta=_resolve_value(self.delta, values),
            beta=_resolve_value(self.beta, values),
            roughness=roughness,
        )


@dataclass(frozen=True)
class SuperlatticeTemplate:
    """A repeated sequence of layer templates."""

    repeats: int
    period: tuple[LayerTemplate, ...]

    def __post_init__(self) -> None:
        if self.repeats <= 0:
            raise ValueError("superlattice repeats must be positive")
        if not self.period:
            raise ValueError("superlattice period must contain at least one layer")

    def build(
        self,
        values: dict[str, float],
        energy_ev: float,
        base_dir: str | Path = ".",
    ) -> list[StackLayer]:
        """Build all layers in the repeated superlattice."""

        layers: list[StackLayer] = []
        for _ in range(self.repeats):
            layers.extend(
                layer.build(values, energy_ev=energy_ev, base_dir=base_dir)
                for layer in self.period
            )
        return layers


StackPart = LayerTemplate | SuperlatticeTemplate


@dataclass(frozen=True)
class StackTemplate:
    """A declarative recipe for a complete vacuum-to-substrate stack.

    The first built layer must be vacuum-like and the last built layer is
    treated as the semi-infinite substrate. Therefore, the last layer thickness
    must be zero; its roughness still describes the roughness of its upper
    interface.
    """

    energy_ev: float
    parts: tuple[StackPart, ...]
    base_dir: str | Path = "."

    def __post_init__(self) -> None:
        if len(self.parts) < 2:
            raise ValueError("stack template requires at least vacuum and substrate")

    def build(self, values: dict[str, float] | None = None) -> SimulationStack:
        """Build a `SimulationStack` from fitted parameter values."""

        values = {} if values is None else values
        layers: list[StackLayer] = []
        for part in self.parts:
            if isinstance(part, LayerTemplate):
                layers.append(
                    part.build(values, energy_ev=self.energy_ev, base_dir=self.base_dir)
                )
            else:
                layers.extend(
                    part.build(values, energy_ev=self.energy_ev, base_dir=self.base_dir)
                )
        _validate_boundary_layers(layers)
        return SimulationStack(tuple(layers))

    def builder(self):
        """Return a callable suitable for `FittingProblem.stack_builder`."""

        return self.build


def _resolve_value(value: ScalarOrParameter, values: dict[str, float]) -> float:
    if isinstance(value, str):
        if value not in values:
            raise ValueError(f"missing stack parameter {value!r}")
        resolved = values[value]
    else:
        resolved = value
    if not np.isfinite(resolved):
        raise ValueError("stack parameter values must be finite")
    return float(resolved)


def _resolve_path(path: str | Path, base_dir: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(base_dir) / path


def _validate_boundary_layers(layers: list[StackLayer]) -> None:
    first = layers[0]
    if (
        first.material.lower() != "vacuum"
        or first.thickness != 0.0
        or first.delta != 0.0
        or first.beta != 0.0
    ):
        raise ValueError("first stack layer must be vacuum")
    if layers[-1].thickness != 0.0:
        raise ValueError("last stack layer is the semi-infinite substrate and must have zero thickness")
