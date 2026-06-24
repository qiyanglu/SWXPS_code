"""Internal material and emitting-layer utilities for XPS workflows."""

from __future__ import annotations

from collections.abc import Sequence


def _values_by_material(
    materials: Sequence[str],
    values: dict[str, float],
    default: float | None,
) -> list[float]:
    output = []
    for material in materials:
        if material in values:
            output.append(float(values[material]))
        elif default is None:
            raise ValueError(f"missing value for material {material!r}")
        else:
            output.append(float(default))
    return output


def _apply_emitting_layer_filter(
    concentration_by_layer: Sequence[float],
    emitting_layer_indices: Sequence[int],
) -> list[float]:
    if not emitting_layer_indices:
        raise ValueError("emitting_layer_indices must not be empty")
    layer_count = len(concentration_by_layer)
    selected = set()
    for index in emitting_layer_indices:
        if index < 0 or index >= layer_count:
            raise ValueError("emitting_layer_indices contains an index outside the stack")
        selected.add(int(index))
    return [
        float(concentration) if index in selected else 0.0
        for index, concentration in enumerate(concentration_by_layer)
    ]


__all__ = ["_apply_emitting_layer_filter", "_values_by_material"]
