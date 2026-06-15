"""Schematic visualization helpers for fitted multilayer stacks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .simulation import SimulationStack, StackLayer


@dataclass(frozen=True)
class SchematicLayer:
    """One visible row in a collapsed stack schematic."""

    material: str
    thickness: float
    roughness: float
    source_index: int | None
    is_gap: bool = False
    collapsed_count: int = 0


def schematic_layers(
    stack: SimulationStack,
    top_layers: int = 8,
    bottom_layers: int = 3,
) -> tuple[SchematicLayer, ...]:
    """Return stack layers to draw, with a middle gap if the stack is long.

    The vacuum layer is omitted from the returned sample schematic. The
    semi-infinite substrate is kept as the final visible row.
    """

    if top_layers < 0 or bottom_layers < 0:
        raise ValueError("top_layers and bottom_layers must be non-negative")
    sample_layers = stack.layers[1:]
    if not sample_layers:
        raise ValueError("stack must contain at least one non-vacuum layer")
    if len(sample_layers) <= top_layers + bottom_layers + 1:
        return tuple(
            _schematic_layer(layer, source_index=index + 1)
            for index, layer in enumerate(sample_layers)
        )

    top = sample_layers[:top_layers]
    bottom = sample_layers[len(sample_layers) - bottom_layers :] if bottom_layers else ()
    collapsed_count = len(sample_layers) - len(top) - len(bottom)
    gap = SchematicLayer(
        material="...",
        thickness=0.0,
        roughness=0.0,
        source_index=None,
        is_gap=True,
        collapsed_count=collapsed_count,
    )
    output = [
        *(
            _schematic_layer(layer, source_index=index + 1)
            for index, layer in enumerate(top)
        ),
        gap,
    ]
    bottom_start = 1 + len(sample_layers) - len(bottom)
    output.extend(
        _schematic_layer(layer, source_index=bottom_start + index)
        for index, layer in enumerate(bottom)
    )
    return tuple(output)


def plot_stack_schematic(
    path: str | Path,
    stack: SimulationStack,
    *,
    title: str | None = None,
    top_layers: int = 8,
    bottom_layers: int = 3,
    width: float = 4.8,
    annotate_thickness: bool = True,
    show_roughness: bool = False,
    show_waves: bool = True,
    show_standing_wave: bool = True,
    collapse_repeated_annotations: bool = True,
) -> None:
    """Draw a 2.5D schematic of a multilayer sample stack.

    The first layer in `stack` is treated as vacuum and omitted from the sample
    body. The final layer is drawn as the semi-infinite substrate. Long stacks
    are collapsed with an ellipsis row between the top and bottom layers.
    """

    plt = _load_pyplot()
    visible = schematic_layers(stack, top_layers=top_layers, bottom_layers=bottom_layers)
    heights = _display_heights(visible)
    total_height = float(sum(heights))
    depth_edges = np.concatenate(([0.0], np.cumsum(heights)))
    colors = _material_colors([layer.material for layer in visible if not layer.is_gap])

    fig_height = max(6.0, 0.52 * len(visible) + 2.5)
    fig, ax = plt.subplots(figsize=(9.8, fig_height))
    x0 = 0.0
    skew = 0.58
    depth_scale = 0.34

    ax.text(
        x0 + 0.5 * width,
        0.42,
        "vacuum",
        ha="center",
        va="center",
        color="0.35",
        fontsize=12,
    )
    period_pair = _first_repeated_pair(visible) if collapse_repeated_annotations else None
    annotated_signatures: set[tuple[str, float, float]] = set()
    positions = [
        (index, layer, depth_edges[index] + 1.0, depth_edges[index + 1] + 1.0)
        for index, layer in enumerate(visible)
    ]

    for _, layer, top, bottom in reversed(positions):
        if layer.is_gap:
            continue
        _draw_layer_box(
            ax,
            x0,
            width,
            top,
            bottom,
            skew,
            depth_scale,
            _layer_color(layer, colors),
        )

    if show_standing_wave:
        _draw_standing_wave(ax, x0, width, depth_edges + 1.0)

    for _, layer, top, bottom in positions:
        if layer.is_gap:
            _draw_gap(ax, x0, width, top, bottom, skew, depth_scale, layer.collapsed_count)
            continue
        y_center = 0.5 * (top + bottom)
        label = layer.material
        if layer.source_index is not None:
            label = f"{label}  #{layer.source_index}"
        ax.text(
            x0 + 0.66 * width + 0.5 * skew,
            y_center,
            label,
            ha="center",
            va="center",
            fontsize=12,
            color="black",
        )
        if annotate_thickness and _should_annotate_layer(
            layer,
            annotated_signatures,
            collapse_repeated_annotations,
        ):
            _annotate_thickness(ax, layer, x0 + width + skew + 0.42, top, bottom)
        if show_roughness and layer.roughness > 0:
            _draw_roughness_hint(ax, x0, width, top, skew, depth_scale)

    if annotate_thickness and period_pair is not None:
        start_index, end_index = period_pair
        _annotate_period(
            ax,
            visible,
            depth_edges + 1.0,
            start_index,
            end_index,
            x0 + width + skew + 1.45,
        )

    if show_waves:
        _draw_xray_waves(ax, x0, width, skew)

    if title:
        ax.set_title(title, pad=14, fontsize=18)
    ax.set_xlim(-1.25, width + skew + 3.15)
    ax.set_ylim(total_height + 2.3, -1.3)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _schematic_layer(layer: StackLayer, source_index: int) -> SchematicLayer:
    return SchematicLayer(
        material=layer.material,
        thickness=float(layer.thickness),
        roughness=float(layer.roughness),
        source_index=source_index,
    )


def _display_heights(layers: Sequence[SchematicLayer]) -> np.ndarray:
    heights = []
    finite_thicknesses = [
        layer.thickness for layer in layers if not layer.is_gap and layer.thickness > 0
    ]
    median = float(np.median(finite_thicknesses)) if finite_thicknesses else 1.0
    for layer in layers:
        if layer.is_gap:
            heights.append(0.72)
        elif layer.thickness <= 0:
            heights.append(1.15)
        else:
            heights.append(float(np.clip(layer.thickness / median, 0.55, 1.7)))
    return np.asarray(heights, dtype=float)


def _draw_layer_box(ax, x0, width, top, bottom, skew, depth_scale, color) -> None:
    front = np.array(
        [
            [x0, top],
            [x0 + width, top],
            [x0 + width, bottom],
            [x0, bottom],
        ]
    )
    side = np.array(
        [
            [x0 + width, top],
            [x0 + width + skew, top - depth_scale],
            [x0 + width + skew, bottom - depth_scale],
            [x0 + width, bottom],
        ]
    )
    top_face = np.array(
        [
            [x0, top],
            [x0 + skew, top - depth_scale],
            [x0 + width + skew, top - depth_scale],
            [x0 + width, top],
        ]
    )
    ax.add_patch(_polygon(front, color, edgecolor="0.20", linewidth=0.8))
    ax.add_patch(_polygon(side, _shade_color(color, 0.82), edgecolor="0.22", linewidth=0.6))
    ax.add_patch(_polygon(top_face, _shade_color(color, 1.12), edgecolor="0.22", linewidth=0.6))


def _draw_gap(ax, x0, width, top, bottom, skew, depth_scale, collapsed_count) -> None:
    y = 0.5 * (top + bottom)
    ax.plot([x0, x0 + width], [y, y], color="0.35", linestyle=":", linewidth=1.1)
    ax.plot(
        [x0 + width, x0 + width + skew],
        [y, y - depth_scale],
        color="0.35",
        linestyle=":",
        linewidth=1.1,
    )
    ax.text(
        x0 + 0.5 * width,
        y + 0.12,
        f"... {collapsed_count} layers ...",
        ha="center",
        va="center",
        fontsize=12,
        color="0.25",
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.8},
    )


def _annotate_thickness(ax, layer: SchematicLayer, x, top, bottom) -> None:
    y_center = 0.5 * (top + bottom)
    if layer.thickness <= 0:
        text = "semi-infinite"
    else:
        text = f"{layer.thickness:g} A"
    if layer.roughness > 0:
        text += f"\nsigma={layer.roughness:g} A"
    ax.annotate(
        "",
        xy=(x, top),
        xytext=(x, bottom),
        arrowprops={"arrowstyle": "<->", "color": "0.25", "linewidth": 1.15},
    )
    ax.text(x + 0.14, y_center, text, ha="left", va="center", fontsize=11, color="0.25")


def _annotate_period(
    ax,
    layers: Sequence[SchematicLayer],
    edges: np.ndarray,
    start_index: int,
    end_index: int,
    x: float,
) -> None:
    top = float(edges[start_index])
    bottom = float(edges[end_index + 1])
    period = sum(layer.thickness for layer in layers[start_index : end_index + 1])
    ax.annotate(
        "",
        xy=(x, top),
        xytext=(x, bottom),
        arrowprops={"arrowstyle": "<->", "color": "0.15", "linewidth": 1.25},
    )
    ax.text(
        x + 0.14,
        0.5 * (top + bottom),
        f"period={period:g} A",
        ha="left",
        va="center",
        fontsize=11,
        color="0.15",
        fontweight="bold",
    )


def _draw_roughness_hint(ax, x0, width, y, skew, depth_scale) -> None:
    x = np.linspace(x0, x0 + width, 140)
    ripple = y + 0.035 * np.sin(np.linspace(0, 8 * np.pi, len(x)))
    ax.plot(x, ripple, color="white", linewidth=1.1, alpha=0.75)
    ax.plot(x + skew, ripple - depth_scale, color="white", linewidth=0.8, alpha=0.45)


def _draw_xray_waves(ax, x0, width, skew) -> None:
    _draw_wave_arrow(
        ax,
        start=(x0 - 0.85, -0.70),
        end=(x0 + 1.85, 0.84),
        color="#2f6f9f",
        label="incident x-ray",
        label_offset=0.38,
    )
    _draw_wave_arrow(
        ax,
        start=(x0 + width + skew - 0.35, 0.82),
        end=(x0 + width + skew + 1.55, -0.68),
        color="#b44945",
        label="diffracted x-ray",
        label_offset=-0.42,
    )


def _draw_wave_arrow(ax, start, end, color, label, label_offset) -> None:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    length = float(np.linalg.norm(direction))
    unit = direction / length
    normal = np.array([-unit[1], unit[0]])
    t = np.linspace(0.0, length, 220)
    points = start + t[:, None] * unit + 0.095 * np.sin(13.0 * np.pi * t / length)[:, None] * normal
    ax.plot(points[:, 0], points[:, 1], color=color, linewidth=3.0)
    ax.annotate("", xy=end, xytext=points[-12], arrowprops={"arrowstyle": "->", "color": color, "lw": 3.0})
    label_position = start + 0.50 * direction + label_offset * normal
    ax.text(
        label_position[0],
        label_position[1],
        label,
        color=color,
        fontsize=12,
        ha="center",
        va="center",
    )


def _draw_standing_wave(ax, x0, width, edges) -> None:
    top = float(edges[0])
    bottom = float(edges[-1])
    y = np.linspace(top, bottom, 600)
    center = x0 + 0.5 * width
    amplitude = 0.30
    x = center + amplitude * np.sin(2.0 * np.pi * 4.5 * (y - top) / max(bottom - top, 1.0))
    ax.plot(x, y, color="#d4a72c", linewidth=3.0, alpha=0.96)
    ax.text(
        x0 - 0.88,
        0.5 * (top + bottom),
        "standing wave",
        color="#7a5b00",
        fontsize=12,
        ha="center",
        va="center",
        rotation=90,
    )


def _material_colors(materials: Sequence[str]) -> dict[str, tuple[float, float, float, float]]:
    plt = _load_pyplot()
    unique = []
    for material in materials:
        if material not in unique:
            unique.append(material)
    cmap = plt.get_cmap("Pastel2")
    colors = {}
    for index, material in enumerate(unique):
        if material.lower() in {"c", "carbon"}:
            colors[material] = (0.43, 0.47, 0.48, 1.0)
        elif "lno" in material.lower() or "lanio" in material.lower():
            colors[material] = (0.76, 0.52, 0.48, 1.0)
        elif "sto" in material.lower() or "srtio" in material.lower():
            colors[material] = (0.50, 0.67, 0.62, 1.0)
        else:
            colors[material] = cmap(index % cmap.N)
    return colors


def _layer_color(
    layer: SchematicLayer,
    colors: dict[str, tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    if layer.thickness <= 0:
        return (0.58, 0.57, 0.52, 1.0)
    return colors[layer.material]


def _should_annotate_layer(
    layer: SchematicLayer,
    annotated_signatures: set[tuple[str, float, float]],
    collapse_repeated_annotations: bool,
) -> bool:
    if layer.thickness <= 0:
        return True
    signature = (
        layer.material,
        round(layer.thickness, 6),
        round(layer.roughness, 6),
    )
    if collapse_repeated_annotations and signature in annotated_signatures:
        return False
    annotated_signatures.add(signature)
    return True


def _first_repeated_pair(layers: Sequence[SchematicLayer]) -> tuple[int, int] | None:
    signatures = [
        (
            layer.material,
            round(layer.thickness, 6),
            round(layer.roughness, 6),
        )
        if not layer.is_gap and layer.thickness > 0
        else None
        for layer in layers
    ]
    for index in range(len(signatures) - 3):
        first = signatures[index]
        second = signatures[index + 1]
        if first is None or second is None or first == second:
            continue
        for later in range(index + 2, len(signatures) - 1):
            if signatures[later] == first and signatures[later + 1] == second:
                return index, index + 1
    return None


def _shade_color(color, factor):
    rgb = np.asarray(color[:3], dtype=float)
    shaded = np.clip(rgb * factor, 0.0, 1.0)
    return (*shaded, color[3])


def _polygon(points, facecolor, edgecolor, linewidth):
    from matplotlib.patches import Polygon

    return Polygon(points, closed=True, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)


def _load_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("matplotlib is required for stack schematic plots") from error
    return plt
