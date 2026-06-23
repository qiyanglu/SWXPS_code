"""Depth profiles for material and element concentrations."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Sequence
from typing import Literal

import numpy as np

from ..simulation import SimulationStack
from .._xps import graded_layer_property_at_depth


@dataclass(frozen=True)
class StackProfiles:
    """One or more stack properties sampled on a common depth grid."""

    depth: np.ndarray
    profiles: dict[str, np.ndarray]


def sample_stack_property(
    stack: SimulationStack,
    values_by_material: dict[str, float],
    step: float = 1.0,
    roughness_profile: Literal["erf", "linear"] = "erf",
    erf_truncation_factor: float = 4.0,
    linear_width_factor: float = sqrt(3.0),
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
        profile=roughness_profile,
        erf_truncation_factor=erf_truncation_factor,
        linear_width_factor=linear_width_factor,
    )
    return depth, values


def sample_concentration_profiles(
    stack: SimulationStack,
    concentration_by_name: dict[str, dict[str, float]],
    step: float = 1.0,
    roughness_profile: Literal["erf", "linear"] = "erf",
    erf_truncation_factor: float = 4.0,
    linear_width_factor: float = sqrt(3.0),
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
            profile=roughness_profile,
            erf_truncation_factor=erf_truncation_factor,
            linear_width_factor=linear_width_factor,
        )
    return StackProfiles(depth=depth, profiles=profiles)


def sample_layer_concentration_profiles(
    stack: SimulationStack,
    concentration_by_name: dict[str, Sequence[float]],
    step: float = 1.0,
    max_depth: float | None = None,
    roughness_profile: Literal["erf", "linear"] = "erf",
    erf_truncation_factor: float = 4.0,
    linear_width_factor: float = sqrt(3.0),
) -> StackProfiles:
    """Sample roughness-graded concentration profiles from per-layer values.

    This is useful when one material appears in multiple chemically distinct
    layers, for example a Ni-free LNO surface layer above Ni-containing LNO.
    Each concentration sequence must have one value per stack layer, including
    vacuum and the final substrate.
    """

    depth = stack_depth_grid(stack, step=step)
    if max_depth is not None:
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        depth = depth[depth <= max_depth]
        if depth.size == 0 or depth[-1] < max_depth:
            depth = np.append(depth, float(max_depth))
    profiles: dict[str, np.ndarray] = {}
    for name, values_by_layer in concentration_by_name.items():
        values = np.asarray(values_by_layer, dtype=float)
        if values.shape != (len(stack.layers),):
            raise ValueError(
                "each per-layer concentration sequence must match stack layer count"
            )
        profiles[name] = graded_layer_property_at_depth(
            stack.optical_layers,
            values,
            depth,
            profile=roughness_profile,
            erf_truncation_factor=erf_truncation_factor,
            linear_width_factor=linear_width_factor,
        )
    return StackProfiles(depth=depth, profiles=profiles)


def plot_vertical_concentration_profiles(
    path: str | Path,
    stack: SimulationStack,
    concentration_by_name: dict[str, Sequence[float]],
    *,
    max_depth: float,
    step: float = 0.1,
    title: str | None = None,
    colors: dict[str, str] | None = None,
    layer_labels: dict[int, str] | None = None,
    show_layer_shading: bool = True,
    layer_box_style: bool = False,
    separate_tracks: bool = False,
    categorical_strips: bool = False,
    roughness_profile: Literal["erf", "linear"] = "erf",
) -> StackProfiles:
    """Plot roughness-graded concentration profiles with depth on the y-axis.

    The depth axis increases downward to match the stack schematic convention.
    Concentration profiles are drawn as filled shades, not line traces. Layer
    labels are placed in a right-side strip so they do not overlap profile
    edges or interface transitions.
    """

    if max_depth <= 0:
        raise ValueError("max_depth must be positive")
    profiles = sample_layer_concentration_profiles(
        stack,
        concentration_by_name,
        step=step,
        max_depth=max_depth,
        roughness_profile=roughness_profile,
    )
    plt = _load_pyplot()
    profile_colors = _profile_colors(tuple(profiles.profiles), colors)

    if categorical_strips:
        _plot_vertical_concentration_strips(
            plt,
            path,
            stack,
            profiles,
            profile_colors,
            max_depth,
            title,
            layer_labels,
        )
        return profiles

    if separate_tracks:
        _plot_vertical_concentration_tracks(
            plt,
            path,
            stack,
            profiles,
            profile_colors,
            max_depth,
            title,
            layer_labels,
            show_layer_shading,
        )
        return profiles

    fig, ax = plt.subplots(figsize=(6.4, 7.4))
    if layer_box_style:
        _draw_vertical_layer_boxes(ax, stack, max_depth, layer_labels)
    elif show_layer_shading:
        _shade_vertical_layers(ax, stack, max_depth, layer_labels)

    for name, values in profiles.profiles.items():
        ax.fill_betweenx(
            profiles.depth,
            0.0,
            values,
            color=profile_colors[name],
            alpha=0.46,
            label=name,
            linewidth=0.0,
        )

    ax.set_xlim(0.0, 1.22)
    ax.set_ylim(max_depth, 0.0)
    ax.set_xlabel("Relative concentration")
    ax.set_ylabel("Depth from surface (A)")
    if title:
        ax.set_title(title, pad=12)
    ax.grid(True, axis="x", alpha=0.18)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return profiles


def _plot_vertical_concentration_strips(
    plt,
    path: str | Path,
    stack: SimulationStack,
    profiles: StackProfiles,
    profile_colors: dict[str, str],
    max_depth: float,
    title: str | None,
    layer_labels: dict[int, str] | None,
) -> None:
    names = tuple(profiles.profiles)
    if not names:
        raise ValueError("at least one concentration profile is required")
    fig, ax = plt.subplots(figsize=(7.0, 7.6))
    for index, name in enumerate(names):
        rgba = _rgba_for_profile(profile_colors[name], profiles.profiles[name])
        ax.imshow(
            rgba[:, np.newaxis, :],
            extent=(index, index + 1, profiles.depth[-1], profiles.depth[0]),
            aspect="auto",
            interpolation="nearest",
            zorder=1,
        )
    _draw_vertical_layer_boxes(
        ax,
        stack,
        max_depth,
        layer_labels,
        x_start=0.0,
        x_stop=float(len(names)),
        label_x=float(len(names)) + 0.08,
    )
    ax.set_xlim(0.0, float(len(names)) + 0.55)
    ax.set_ylim(max_depth, 0.0)
    ax.set_xticks(np.arange(len(names)) + 0.5, names, fontsize=15)
    ax.set_ylabel("Depth from surface (A)", fontsize=15)
    if title:
        ax.set_title(title, pad=16, fontsize=18)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_facecolor("#fbfaf7")
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("0.18")
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=1.4)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_vertical_concentration_tracks(
    plt,
    path: str | Path,
    stack: SimulationStack,
    profiles: StackProfiles,
    profile_colors: dict[str, str],
    max_depth: float,
    title: str | None,
    layer_labels: dict[int, str] | None,
    show_layer_shading: bool,
) -> None:
    names = tuple(profiles.profiles)
    fig_width = max(7.0, 2.25 * len(names) + 1.5)
    fig, axes = plt.subplots(
        1,
        len(names),
        figsize=(fig_width, 7.4),
        sharey=True,
        squeeze=False,
    )
    flat_axes = list(axes[0])
    for index, (ax, name) in enumerate(zip(flat_axes, names)):
        if show_layer_shading:
            _shade_vertical_layers(
                ax,
                stack,
                max_depth,
                layer_labels if index == len(flat_axes) - 1 else {},
            )
        ax.fill_betweenx(
            profiles.depth,
            0.0,
            profiles.profiles[name],
            color=profile_colors[name],
            alpha=0.55,
            linewidth=0.0,
        )
        ax.set_xlim(0.0, 1.05 if index < len(flat_axes) - 1 else 1.24)
        ax.set_ylim(max_depth, 0.0)
        ax.set_title(name, pad=8)
        ax.set_xlabel("Relative concentration")
        ax.grid(True, axis="x", alpha=0.18)
        if index == 0:
            ax.set_ylabel("Depth from surface (A)")
        else:
            ax.tick_params(axis="y", labelleft=False)
    if title:
        fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def stack_depth_grid(stack: SimulationStack, step: float = 1.0) -> np.ndarray:
    """Return a depth grid spanning the finite layers of a SimulationStack."""

    if step <= 0:
        raise ValueError("step must be positive")

    total_thickness = sum(layer.thickness for layer in stack.layers[1:-1])
    if total_thickness <= 0:
        return np.array([], dtype=float)

    n_points = max(2, int(np.ceil(total_thickness / step)) + 1)
    return np.linspace(0.0, total_thickness, n_points)


def _shade_vertical_layers(
    ax,
    stack: SimulationStack,
    max_depth: float,
    layer_labels: dict[int, str] | None,
) -> None:
    colors = _material_band_colors()
    top = 0.0
    label_slots: list[tuple[float, str]] = []
    for layer_index, layer in enumerate(stack.layers[1:-1], start=1):
        bottom = top + float(layer.thickness)
        visible_top = max(0.0, top)
        visible_bottom = min(max_depth, bottom)
        if visible_bottom > visible_top:
            color = colors.get(layer.material, "#f2f2f2")
            ax.axhspan(visible_top, visible_bottom, color=color, alpha=0.24, linewidth=0)
            if layer_labels is None:
                label_slots.append((0.5 * (visible_top + visible_bottom), layer.material))
            elif layer_index in layer_labels:
                label_slots.append((0.5 * (visible_top + visible_bottom), layer_labels[layer_index]))
        top = bottom
        if top >= max_depth:
            break

    min_gap = max(2.0, 0.035 * max_depth)
    last_y = -np.inf
    for y, label in label_slots:
        y_text = max(y, last_y + min_gap)
        y_text = min(y_text, max_depth - 0.5)
        ax.text(
            1.055,
            y_text,
            label,
            ha="left",
            va="center",
            fontsize=8,
            color="0.25",
            clip_on=False,
        )
        last_y = y_text


def _draw_vertical_layer_boxes(
    ax,
    stack: SimulationStack,
    max_depth: float,
    layer_labels: dict[int, str] | None,
    x_start: float = 0.0,
    x_stop: float = 1.0,
    label_x: float = 1.035,
) -> None:
    top = 0.0
    label_slots: list[tuple[float, str]] = []
    colors = _layer_box_colors()
    for layer_index, layer in enumerate(stack.layers[1:-1], start=1):
        bottom = top + float(layer.thickness)
        visible_top = max(0.0, top)
        visible_bottom = min(max_depth, bottom)
        if visible_bottom > visible_top:
            color = colors[(layer_index - 1) % len(colors)]
            ax.hlines(
                [visible_top, visible_bottom],
                x_start,
                x_stop,
                colors=color,
                linestyles=(0, (4, 3)),
                linewidth=1.25,
                alpha=0.88,
                zorder=3,
            )
            ax.vlines(
                [x_start, x_stop],
                visible_top,
                visible_bottom,
                colors=color,
                linestyles=(0, (4, 3)),
                linewidth=1.0,
                alpha=0.66,
                zorder=3,
            )
            label = (
                layer_labels.get(layer_index, layer.material)
                if layer_labels
                else layer.material
            )
            label_slots.append((0.5 * (visible_top + visible_bottom), label))
        top = bottom
        if top >= max_depth:
            break

    min_gap = max(2.0, 0.05 * max_depth)
    last_y = -np.inf
    for y, label in label_slots:
        y_text = max(y, last_y + min_gap)
        y_text = min(y_text, max_depth - 0.7)
        ax.text(
            label_x,
            y_text,
            label,
            ha="left",
            va="center",
            fontsize=11,
            color=color,
            clip_on=False,
        )
        last_y = y_text


def _profile_colors(names: tuple[str, ...], colors: dict[str, str] | None) -> dict[str, str]:
    defaults = {
        "La": "#d9aa68",
        "Ni": "#82a9be",
        "C": "#91bd90",
        "O": "#d99a96",
        "Ti": "#aa9cc8",
        "Sr": "#bba18b",
    }
    output = {}
    for index, name in enumerate(names):
        if colors and name in colors:
            output[name] = colors[name]
        else:
            output[name] = defaults.get(name, f"C{index}")
    return output


def _rgba_for_profile(color: str, values: np.ndarray) -> np.ndarray:
    import matplotlib.colors as mcolors

    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    clipped = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    white = np.ones(3, dtype=float)
    # Low-saturation concentration shade: white at zero, soft color at one.
    mixed = white * (1.0 - 0.66 * clipped[:, np.newaxis]) + rgb * (
        0.66 * clipped[:, np.newaxis]
    )
    alpha = np.ones((clipped.size, 1), dtype=float)
    return np.concatenate((mixed, alpha), axis=1)


def _material_band_colors() -> dict[str, str]:
    return {
        "C": "#dff0d8",
        "LNO": "#fee8c8",
        "STO": "#e0ecf4",
        "vacuum": "#f7f7f7",
    }


def _layer_box_colors() -> tuple[str, ...]:
    return (
        "#7f9f84",
        "#bd7f83",
        "#d09b4f",
        "#6f9bb5",
        "#9488b8",
        "#aa8d72",
    )


def _load_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("matplotlib is required for concentration profile plots") from error
    return plt

__all__ = [
    "StackProfiles",
    "plot_vertical_concentration_profiles",
    "sample_concentration_profiles",
    "sample_layer_concentration_profiles",
    "sample_stack_property",
    "stack_depth_grid",
]