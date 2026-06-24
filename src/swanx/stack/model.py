"""Material-labeled stack data models for high-level simulations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..layers import Layer


@dataclass(frozen=True)
class StackLayer:
    """A material-labeled layer used by high-level simulations."""

    material: str
    thickness: float
    delta: float = 0.0
    beta: float = 0.0
    roughness: float = 0.0

    def to_layer(self) -> Layer:
        """Return the optical Layer representation."""

        return Layer(
            thickness=self.thickness,
            delta=self.delta,
            beta=self.beta,
            roughness=self.roughness,
        )


@dataclass(frozen=True)
class SimulationStack:
    """A material-labeled stack from vacuum to substrate."""

    layers: tuple[StackLayer, ...]

    def __post_init__(self) -> None:
        if len(self.layers) < 2:
            raise ValueError("a simulation stack requires at least two layers")

    @property
    def optical_layers(self) -> list[Layer]:
        """Return low-level optical layers."""

        return [layer.to_layer() for layer in self.layers]

    @property
    def materials(self) -> list[str]:
        """Return material labels for each layer."""

        return [layer.material for layer in self.layers]


def stack_from_layers(materials: Sequence[str], layers: Sequence[Layer]) -> SimulationStack:
    """Create a material-labeled stack from existing Layer objects."""

    if len(materials) != len(layers):
        raise ValueError("materials and layers must have the same length")
    return SimulationStack(
        tuple(
            StackLayer(
                material=material,
                thickness=layer.thickness,
                delta=layer.delta,
                beta=layer.beta,
                roughness=layer.roughness,
            )
            for material, layer in zip(materials, layers)
        )
    )


__all__ = ["SimulationStack", "StackLayer", "stack_from_layers"]
