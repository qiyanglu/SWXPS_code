"""Compare the current best-fit exports for Sample #12 and Sample #13.

The script reads only the stable CSV artifacts in each sample's
``best_results_so_far`` directory. It does not rerun either fit.
"""

from __future__ import annotations

import csv
from math import erf, sqrt
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
CASE_STUDIES_DIR = HERE.parent
OUTPUT_DPI = 240
MAX_DEPTH_A = 20.0

SAMPLE_COLORS = {
    "Sample #12": "#6F9BB3",  # soft blue
    "Sample #13": "#D6937A",  # soft coral
}
ELEMENT_COLORS = {
    "C": "#C9A77C",   # sand
    "La": "#86A98A",  # sage
    "Ni": "#A28DB3",  # mauve
}
PAPER = "#FCFBF8"
INK = "#30353A"
GRID = "#D9D7D1"

CURVE_DATASETS = ("reflectivity", "C 1s", "Ni 3p", "La 4d")
CURVE_TITLES = {
    "reflectivity": "X-ray reflectivity",
    "C 1s": "C 1s rocking curve",
    "Ni 3p": "Ni 3p rocking curve",
    "La 4d": "La 4d rocking curve",
}
CURVE_COLORS = {
    "reflectivity": "#657F8B",
    "C 1s": ELEMENT_COLORS["C"],
    "Ni 3p": ELEMENT_COLORS["Ni"],
    "La 4d": ELEMENT_COLORS["La"],
}


@dataclass(frozen=True)
class SampleExport:
    label: str
    curve_csv: Path
    concentration_csv: Path
    stack_csv: Path


def locate_exports() -> tuple[SampleExport, SampleExport]:
    """Locate each sample's current best-fit curve and concentration exports."""
    exports = []
    for number in (12, 13):
        best_dir = (
            CASE_STUDIES_DIR
            / f"sample_{number}"
            / "best_results_so_far"
        )
        curve_csv = best_dir / "best_fit_experiment_and_simulation.csv"
        stack_csv = best_dir / "optimized_stack_layers.csv"
        profile_matches = sorted(
            best_dir.glob("*vertical_concentration_profiles.csv")
        )
        if not curve_csv.is_file():
            raise FileNotFoundError(f"Missing best-fit curve export: {curve_csv}")
        if not stack_csv.is_file():
            raise FileNotFoundError(f"Missing optimized stack export: {stack_csv}")
        if len(profile_matches) != 1:
            raise FileNotFoundError(
                f"Expected one concentration export in {best_dir}, "
                f"found {len(profile_matches)}"
            )
        exports.append(
            SampleExport(
                label=f"Sample #{number}",
                curve_csv=curve_csv,
                concentration_csv=profile_matches[0],
                stack_csv=stack_csv,
            )
        )
    return tuple(exports)  # type: ignore[return-value]


def read_curve_csv(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Read and validate exported measured and fitted curves."""
    required = {
        "dataset",
        "angle_deg",
        "calculation_angle_deg",
        "experimental",
        "fitted",
    }
    grouped: dict[str, dict[str, list[float]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or ()):
            missing = sorted(required.difference(reader.fieldnames or ()))
            raise ValueError(f"{path} is missing columns: {missing}")
        for row in reader:
            dataset = row["dataset"]
            values = grouped.setdefault(
                dataset,
                {"angle_deg": [], "experimental": [], "fitted": []},
            )
            for key in values:
                values[key].append(float(row[key]))

    missing_datasets = set(CURVE_DATASETS).difference(grouped)
    if missing_datasets:
        raise ValueError(f"{path} is missing datasets: {sorted(missing_datasets)}")

    output: dict[str, dict[str, np.ndarray]] = {}
    for dataset in CURVE_DATASETS:
        output[dataset] = {
            key: np.asarray(values, dtype=float)
            for key, values in grouped[dataset].items()
        }
        arrays = output[dataset].values()
        if not all(np.all(np.isfinite(values)) for values in arrays):
            raise ValueError(f"Non-finite values in {path}, dataset {dataset}")
        if dataset == "reflectivity":
            reflectivity = output[dataset]
            if np.any(reflectivity["experimental"] <= 0) or np.any(
                reflectivity["fitted"] <= 0
            ):
                raise ValueError(f"Reflectivity must be positive in {path}")
    return output


def read_concentration_csv(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read and validate the exported C, La, and Ni concentration profiles."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"depth_A", *ELEMENT_COLORS}
        if not required.issubset(reader.fieldnames or ()):
            missing = sorted(required.difference(reader.fieldnames or ()))
            raise ValueError(f"{path} is missing columns: {missing}")
        rows = list(reader)

    depth = np.asarray([float(row["depth_A"]) for row in rows], dtype=float)
    profiles = {
        element: np.asarray([float(row[element]) for row in rows], dtype=float)
        for element in ELEMENT_COLORS
    }
    if depth.size < 2 or np.any(np.diff(depth) <= 0):
        raise ValueError(f"Depth must be strictly increasing in {path}")
    for element, values in profiles.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite {element} concentrations in {path}")
        if np.any(values < -1e-9) or np.any(values > 1.0 + 1e-9):
            raise ValueError(f"{element} concentration outside [0, 1] in {path}")
    if depth[-1] < MAX_DEPTH_A:
        raise ValueError(f"{path} does not extend to {MAX_DEPTH_A:g} A")
    return depth, profiles


def read_surface_stack(path: Path) -> tuple[float, float, float]:
    """Return C thickness, vacuum/C sigma, and C/LNO sigma from a stack export."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    carbon_index = next(
        (index for index, row in enumerate(rows) if row["material"] == "C"),
        None,
    )
    if carbon_index is None or carbon_index + 1 >= len(rows):
        raise ValueError(f"Could not locate C and its underlying layer in {path}")
    carbon = rows[carbon_index]
    below = rows[carbon_index + 1]
    return (
        float(carbon["thickness_A"]),
        float(carbon["roughness_A"]),
        float(below["roughness_A"]),
    )


def smooth_interface_fraction(distance: np.ndarray, sigma: float) -> np.ndarray:
    """Return an error-function step, or an exact step for zero roughness."""
    distance = np.asarray(distance, dtype=float)
    if sigma <= 0.0:
        return (distance >= 0.0).astype(float)
    scaled = distance / (sqrt(2.0) * sigma)
    erf_values = np.fromiter((erf(value) for value in scaled), dtype=float)
    return 0.5 * (1.0 + erf_values)


def read_corrected_concentrations(
    export: SampleExport,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read profiles and remove the overlapping-interface C overwrite artifact."""
    depth, profiles = read_concentration_csv(export.concentration_csv)
    carbon_thickness, surface_sigma, buried_sigma = read_surface_stack(
        export.stack_csv
    )
    entered_carbon = smooth_interface_fraction(depth, surface_sigma)
    entered_lno = smooth_interface_fraction(depth - carbon_thickness, buried_sigma)
    profiles["C"] = entered_carbon * (1.0 - entered_lno)
    return depth, profiles

def apply_plot_style() -> None:
    """Set a soft, legible style shared by both output figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 19,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.titleweight": "semibold",
            "axes.labelcolor": INK,
            "axes.edgecolor": "#8D9295",
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
        }
    )


def plot_curve_comparison(
    exports: tuple[SampleExport, ...],
    curves: dict[str, dict[str, dict[str, np.ndarray]]],
) -> Path:
    """Draw one 2x2 grid per sample, arranged left and right."""
    fig = plt.figure(figsize=(23.0, 11.7))
    grid = fig.add_gridspec(
        2,
        5,
        width_ratios=(1.0, 1.0, 0.16, 1.0, 1.0),
        left=0.055,
        right=0.985,
        bottom=0.09,
        top=0.79,
        hspace=0.29,
        wspace=0.25,
    )
    sample_columns = ((0, 1), (3, 4))
    axes_by_sample: dict[str, list[plt.Axes]] = {}

    shared_y_limits: dict[str, tuple[float, float]] = {}
    for dataset in CURVE_DATASETS:
        all_values = np.concatenate(
            [
                curves[export.label][dataset][key]
                for export in exports
                for key in ("experimental", "fitted")
            ]
        )
        if dataset == "reflectivity":
            positive = all_values[all_values > 0]
            shared_y_limits[dataset] = (
                10 ** np.floor(np.log10(positive.min())),
                10 ** np.ceil(np.log10(positive.max())),
            )
        else:
            span = all_values.max() - all_values.min()
            padding = max(0.015, 0.10 * span)
            shared_y_limits[dataset] = (
                all_values.min() - padding,
                all_values.max() + padding,
            )

    for sample_index, export in enumerate(exports):
        sample_axes: list[plt.Axes] = []
        sample_data = curves[export.label]
        sample_angles = np.concatenate(
            [sample_data[dataset]["angle_deg"] for dataset in CURVE_DATASETS]
        )
        sample_xlim = (sample_angles.min() - 0.06, sample_angles.max() + 0.06)
        for panel_index, dataset in enumerate(CURVE_DATASETS):
            row, local_column = divmod(panel_index, 2)
            axis = fig.add_subplot(grid[row, sample_columns[sample_index][local_column]])
            sample_axes.append(axis)
            values = sample_data[dataset]
            color = CURVE_COLORS[dataset]
            axis.scatter(
                values["angle_deg"],
                values["experimental"],
                s=28,
                color=color,
                alpha=0.60,
                edgecolors=PAPER,
                linewidths=0.55,
                zorder=3,
            )
            order = np.argsort(values["angle_deg"])
            axis.plot(
                values["angle_deg"][order],
                values["fitted"][order],
                color=color,
                linewidth=2.7,
                solid_capstyle="round",
                zorder=4,
            )
            axis.set_title(
                CURVE_TITLES[dataset],
                loc="left",
                pad=10,
                color=CURVE_COLORS[dataset],
            )
            axis.set_xlim(*sample_xlim)
            axis.set_ylim(*shared_y_limits[dataset])
            axis.grid(True, color=GRID, linewidth=0.8, alpha=0.6)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(length=5, width=1.0)
            if dataset == "reflectivity":
                axis.set_yscale("log")
                axis.set_ylabel("Reflectivity")
            else:
                axis.set_ylabel("Normalized intensity")
            if row == 1:
                axis.set_xlabel("Measured grazing angle (deg)")
        axes_by_sample[export.label] = sample_axes

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#8E9396",
            markeredgecolor=PAPER,
            markersize=8,
            label="Experiment",
        ),
        Line2D(
            [0],
            [0],
            color="#666B6E",
            linewidth=2.7,
            label="Best fit",
        ),
    ]
    fig.suptitle(
        "Best-fit reflectivity and rocking curves",
        fontsize=27,
        fontweight="semibold",
        y=0.982,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.927),
        ncol=2,
        frameon=False,
        fontsize=14,
        handlelength=2.2,
        columnspacing=2.2,
    )
    for export in exports:
        sample_axes = axes_by_sample[export.label]
        left = min(axis.get_position().x0 for axis in sample_axes)
        right = max(axis.get_position().x1 for axis in sample_axes)
        fig.text(
            (left + right) / 2,
            0.846,
            export.label,
            ha="center",
            va="center",
            fontsize=23,
            fontweight="bold",
            color=SAMPLE_COLORS[export.label],
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": SAMPLE_COLORS[export.label],
                "edgecolor": "none",
                "alpha": 0.13,
            },
        )

    output = HERE / "reflectivity_and_rcs_comparison.png"
    fig.savefig(output, dpi=OUTPUT_DPI)
    plt.close(fig)
    return output

def concentration_rgba(
    depth: np.ndarray,
    profiles: dict[str, np.ndarray],
    start_depth_A: float = 0.0,
    elements: tuple[str, ...] = tuple(ELEMENT_COLORS),
) -> np.ndarray:
    """Convert separate element profiles to vertical RGB concentration strips."""
    stop_depth_A = start_depth_A + MAX_DEPTH_A
    if start_depth_A < 0.0 or depth[-1] < stop_depth_A:
        raise ValueError(
            f"Profile depth {depth[-1]:g} A does not cover "
            f"{start_depth_A:g}--{stop_depth_A:g} A"
        )
    common_depth = np.linspace(start_depth_A, stop_depth_A, 801)
    white = np.asarray(to_rgb(PAPER))
    columns = []
    for element in elements:
        color = ELEMENT_COLORS[element]
        concentration = np.clip(
            np.interp(common_depth, depth, profiles[element]), 0.0, 1.0
        )
        base = np.asarray(to_rgb(color))
        columns.append(white + concentration[:, None] * (base - white))
    return np.stack(columns, axis=1)


def plot_concentration_maps(
    exports: tuple[SampleExport, ...],
    concentrations: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]],
    *,
    depth_origins: dict[str, float] | None = None,
    output_name: str = "concentration_depth_maps_comparison.png",
    title: str = "Near-surface element concentration maps",
    depth_label: str = "Depth below surface (Å)",
    subtitle: str = (
        "Separate element columns · color saturation is concentration · zero is white"
    ),
    elements: tuple[str, ...] = tuple(ELEMENT_COLORS),
) -> Path:
    """Draw paired vertical maps with C, La, and Ni in distinct columns."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 10.0), sharey=True)
    origins = depth_origins or {export.label: 0.0 for export in exports}
    for panel_index, (axis, export) in enumerate(zip(axes, exports)):
        depth, profiles = concentrations[export.label]
        rgba = concentration_rgba(
            depth, profiles, origins[export.label], elements
        )
        axis.imshow(
            rgba,
            extent=(-0.5, len(elements) - 0.5, MAX_DEPTH_A, 0.0),
            origin="upper",
            aspect="auto",
            interpolation="nearest",
        )
        for boundary in np.arange(0.5, len(elements) - 0.5, 1.0):
            axis.axvline(boundary, color=PAPER, linewidth=5, zorder=3)
        for depth_mark in (5.0, 10.0, 15.0):
            axis.axhline(depth_mark, color="#FFFFFF", linewidth=0.9, alpha=0.7)
        axis.set_xticks(range(len(elements)), elements)
        for tick, element in zip(axis.get_xticklabels(), elements):
            tick.set_color(ELEMENT_COLORS[element])
            tick.set_fontweight("semibold")
            tick.set_fontsize(18)
        axis.set_yticks((0, 5, 10, 15, 20))
        axis.tick_params(axis="x", length=0, pad=10)
        axis.tick_params(axis="y", length=0, pad=7)
        axis.spines[:].set_visible(False)
        axis.set_title(
            f"({chr(ord('a') + panel_index)})  {export.label}",
            loc="left",
            pad=16,
            color=SAMPLE_COLORS[export.label],
        )
        axis.set_xlim(-0.5, len(elements) - 0.5)
        axis.set_ylim(MAX_DEPTH_A, 0.0)
    axes[0].set_ylabel(depth_label, fontsize=17)
    axes[1].tick_params(axis="y", labelleft=False)

    fig.suptitle(title, fontsize=25, fontweight="semibold", y=0.97)
    fig.text(
        0.5,
        0.89,
        subtitle,
        ha="center",
        fontsize=14,
        color="#6C7174",
    )
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.10, top=0.82, wspace=0.16)

    output = HERE / output_name
    fig.savefig(output, dpi=OUTPUT_DPI)
    plt.close(fig)
    return output

def main() -> None:
    apply_plot_style()
    exports = locate_exports()
    curves = {export.label: read_curve_csv(export.curve_csv) for export in exports}
    concentrations = {
        export.label: read_corrected_concentrations(export)
        for export in exports
    }
    lno_origins = {
        export.label: read_surface_stack(export.stack_csv)[0]
        for export in exports
    }
    curve_output = plot_curve_comparison(exports, curves)
    surface_map_output = plot_concentration_maps(exports, concentrations)
    lno_map_output = plot_concentration_maps(
        exports,
        concentrations,
        depth_origins=lno_origins,
        output_name="lno_aligned_concentration_depth_maps_comparison.png",
        title="Top-LNO-aligned element concentration maps",
        depth_label="Depth from nominal LNO top (Å)",
        subtitle=(
            "Nominal C/LNO interface aligned at 0 Å · "
            "identical 20 Å LNO depth window"
        ),
        elements=("La", "Ni"),
    )
    print(f"Wrote {curve_output}")
    print(f"Wrote {surface_map_output}")
    print(f"Wrote {lno_map_output}")

if __name__ == "__main__":
    main()





