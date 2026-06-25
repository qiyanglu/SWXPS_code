"""Compare legacy and unified slicing on the synthetic C/LNO/STO case."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

import fit_reflectivity_rc_bo as case  # noqa: E402
from swanx.stack import (
    LayerSlicingPolicy,
    fixed_layer_grid_plan,
)
from swanx.workflows.simulate import (
    ReflectivityRequest,
    RockingCurveRequest,
    simulate_reflectivity,
    simulate_rocking_curves,
)
from swanx.fields import effective_layers_with_roughness  # noqa: E402

DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "synthetic_c_lno_sto" / "slicing_comparison"


def capacity_values() -> dict[str, float]:
    """Return true values with fitted thicknesses moved to their upper bounds."""

    values = dict(case.TRUE_VALUES)
    for name in ("carbon_thickness", "lno_thickness", "sto_thickness"):
        values[name] = case.PARAMETER_BY_NAME[name].upper
    return values


def simulate_comparison(
    min_slices: int = 10,
    max_slice_thickness: float = 2.0,
) -> dict[str, object]:
    """Run matched legacy and unified simulations and return their arrays."""

    data = case.load_data(case.DATA_FILE)
    angles = data["angle_deg"]
    stack = case.build_stack(case.TRUE_VALUES)
    capacity_stack = case.build_stack(capacity_values())
    policy = LayerSlicingPolicy(
        min_slices=min_slices,
        max_slice_thickness=max_slice_thickness,
    )
    plan = fixed_layer_grid_plan(capacity_stack.optical_layers, policy)
    peak_angle = angles[np.argmax(data["reflectivity"])]
    offpeak_mask = np.abs(angles - peak_angle) > 1.25
    core_levels = case.core_level_requests()

    legacy_start = perf_counter()
    legacy_reflectivity = simulate_reflectivity(
        ReflectivityRequest(
            angles=angles,
            energy_ev=case.PHOTON_ENERGY_EV,
            stack=stack,
            roughness_step=1.0,
            slicing=None,
        )
    ).reflectivity
    legacy_rc_result = simulate_rocking_curves(
        RockingCurveRequest(
            angles=angles,
            photon_energy_ev=case.PHOTON_ENERGY_EV,
            stack=stack,
            core_levels=core_levels,
            field_step=1.0,
            roughness_step=1.0,
            offpeak_mask=offpeak_mask,
            slicing=None,
        )
    )
    legacy_seconds = perf_counter() - legacy_start

    unified_start = perf_counter()
    unified_reflectivity = simulate_reflectivity(
        ReflectivityRequest(
            angles=angles,
            energy_ev=case.PHOTON_ENERGY_EV,
            stack=stack,
            slicing=plan,
        )
    ).reflectivity
    unified_rc_result = simulate_rocking_curves(
        RockingCurveRequest(
            angles=angles,
            photon_energy_ev=case.PHOTON_ENERGY_EV,
            stack=stack,
            core_levels=core_levels,
            offpeak_mask=offpeak_mask,
            slicing=plan,
        )
    )
    unified_seconds = perf_counter() - unified_start

    legacy_curves = {
        core.name: core.curve.intensity for core in legacy_rc_result.core_levels
    }
    unified_curves = {
        core.name: core.curve.intensity for core in unified_rc_result.core_levels
    }
    legacy_effective = effective_layers_with_roughness(
        stack.optical_layers,
        step=1.0,
    )

    return {
        "angles": angles,
        "legacy_reflectivity": legacy_reflectivity,
        "unified_reflectivity": unified_reflectivity,
        "legacy_curves": legacy_curves,
        "unified_curves": unified_curves,
        "legacy_seconds": legacy_seconds,
        "unified_seconds": unified_seconds,
        "legacy_effective_cells": len(legacy_effective) - 2,
        "unified_cells": sum(plan.slice_counts),
        "unified_slice_counts": plan.slice_counts,
        "policy": policy,
    }


def difference_metrics(results: dict[str, object]) -> dict[str, dict[str, float]]:
    """Return absolute, relative, RMS, and log-space comparison metrics."""

    legacy_reflectivity = np.asarray(results["legacy_reflectivity"])
    unified_reflectivity = np.asarray(results["unified_reflectivity"])
    metrics = {
        "reflectivity": _curve_metrics(
            legacy_reflectivity,
            unified_reflectivity,
            include_log=True,
        )
    }
    legacy_curves = results["legacy_curves"]
    unified_curves = results["unified_curves"]
    for name in case.RC_COLUMN_BY_NAME:
        metrics[name] = _curve_metrics(
            np.asarray(legacy_curves[name]),
            np.asarray(unified_curves[name]),
            include_log=False,
        )
    return metrics


def _curve_metrics(
    legacy: np.ndarray,
    unified: np.ndarray,
    include_log: bool,
) -> dict[str, float]:
    difference = unified - legacy
    output = {
        "max_absolute": float(np.max(np.abs(difference))),
        "rms_absolute": float(np.sqrt(np.mean(difference**2))),
        "max_relative": float(
            np.max(np.abs(difference) / np.maximum(np.abs(legacy), 1.0e-12))
        ),
    }
    if include_log:
        log_difference = np.log10(np.maximum(unified, 1.0e-12)) - np.log10(
            np.maximum(legacy, 1.0e-12)
        )
        output["max_log10"] = float(np.max(np.abs(log_difference)))
        output["rms_log10"] = float(np.sqrt(np.mean(log_difference**2)))
    return output


def save_pointwise_csv(path: Path, results: dict[str, object]) -> None:
    """Save legacy, unified, and signed differences for every curve."""

    columns = [np.asarray(results["angles"])]
    names = ["angle_deg"]
    curve_pairs = [
        (
            "reflectivity",
            np.asarray(results["legacy_reflectivity"]),
            np.asarray(results["unified_reflectivity"]),
        )
    ]
    for name, column in case.RC_COLUMN_BY_NAME.items():
        curve_pairs.append(
            (
                column,
                np.asarray(results["legacy_curves"][name]),
                np.asarray(results["unified_curves"][name]),
            )
        )
    for label, legacy, unified in curve_pairs:
        columns.extend((legacy, unified, unified - legacy))
        names.extend((f"{label}_legacy", f"{label}_unified", f"{label}_difference"))
    np.savetxt(
        path,
        np.column_stack(columns),
        delimiter=",",
        header=",".join(names),
        comments="",
    )


def save_summary(
    path: Path,
    results: dict[str, object],
    metrics: dict[str, dict[str, float]],
) -> None:
    """Save a compact human-readable numerical comparison."""

    lines = [
        "Synthetic C/[LNO/STO]x20/STO slicing comparison",
        f"angles: {len(results['angles'])}",
        f"legacy effective cells: {results['legacy_effective_cells']}",
        f"unified cells: {results['unified_cells']}",
        f"legacy reflectivity+RC seconds: {results['legacy_seconds']:.6f}",
        f"unified reflectivity+RC seconds: {results['unified_seconds']:.6f}",
        "",
    ]
    for name, values in metrics.items():
        lines.append(name)
        lines.extend(f"  {key}: {value:.12g}" for key, value in values.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_comparison(
    path: Path,
    results: dict[str, object],
    metrics: dict[str, dict[str, float]],
) -> None:
    """Save legacy, unified, and difference panels with matched scales."""

    import matplotlib.pyplot as plt

    angles = np.asarray(results["angles"])
    rows = ["reflectivity", *case.RC_COLUMN_BY_NAME]
    fig, axes = plt.subplots(
        len(rows),
        3,
        figsize=(13.0, 12.5),
        sharex=True,
        constrained_layout=True,
    )
    for row, name in enumerate(rows):
        if name == "reflectivity":
            legacy = np.asarray(results["legacy_reflectivity"])
            unified = np.asarray(results["unified_reflectivity"])
            for column, values, title in (
                (0, legacy, "Legacy 1 A"),
                (1, unified, "Unified grid"),
            ):
                axes[row, column].semilogy(angles, values, color="black", linewidth=1.4)
                axes[row, column].set_title(title)
            positive = np.concatenate((legacy[legacy > 0], unified[unified > 0]))
            limits = (max(np.min(positive) * 0.8, 1.0e-12), np.max(positive) * 1.2)
            axes[row, 0].set_ylim(limits)
            axes[row, 1].set_ylim(limits)
            axes[row, 2].plot(angles, unified - legacy, color="tab:red", linewidth=1.2)
            axes[row, 2].set_title("Unified - legacy")
            axes[row, 2].text(
                0.02,
                0.95,
                f"max |dR| = {metrics[name]['max_absolute']:.2e}\n"
                f"RMS dlog10R = {metrics[name]['rms_log10']:.2e}",
                transform=axes[row, 2].transAxes,
                va="top",
                fontsize=8,
            )
            axes[row, 0].set_ylabel("Reflectivity")
        else:
            legacy = np.asarray(results["legacy_curves"][name])
            unified = np.asarray(results["unified_curves"][name])
            color = case.PLOT_COLORS[name]
            axes[row, 0].plot(angles, legacy, color=color, linewidth=1.4)
            axes[row, 1].plot(angles, unified, color=color, linewidth=1.4)
            combined = np.concatenate((legacy, unified))
            padding = max(0.02 * np.ptp(combined), 1.0e-4)
            limits = (np.min(combined) - padding, np.max(combined) + padding)
            axes[row, 0].set_ylim(limits)
            axes[row, 1].set_ylim(limits)
            axes[row, 2].plot(angles, unified - legacy, color="tab:red", linewidth=1.2)
            axes[row, 2].text(
                0.02,
                0.95,
                f"max |d| = {metrics[name]['max_absolute']:.2e}\n"
                f"RMS |d| = {metrics[name]['rms_absolute']:.2e}",
                transform=axes[row, 2].transAxes,
                va="top",
                fontsize=8,
            )
            axes[row, 0].set_ylabel(name)
        axes[row, 2].axhline(0.0, color="black", linestyle=":", linewidth=0.8)
        for column in range(3):
            axes[row, column].grid(True, alpha=0.25)

    for axis in axes[-1]:
        axis.set_xlabel("Grazing incidence angle (deg)")
    fig.suptitle(
        "Synthetic C/[LNO/STO]x20/STO: legacy vs unified slicing\n"
        f"cells {results['legacy_effective_cells']} vs {results['unified_cells']}; "
        f"time {results['legacy_seconds']:.3f} s vs {results['unified_seconds']:.3f} s",
        fontsize=13,
    )
    fig.savefig(path, dpi=200)
    plt.close(fig)


def validate_results(results: dict[str, object]) -> None:
    """Check basic physical and shape invariants before saving artifacts."""

    angles = np.asarray(results["angles"])
    for name in ("legacy_reflectivity", "unified_reflectivity"):
        values = np.asarray(results[name])
        if values.shape != angles.shape or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} is non-finite or has the wrong shape")
        if np.any(values < 0.0) or np.any(values > 1.0 + 1.0e-10):
            raise ValueError(f"{name} violates reflectivity bounds")
    for strategy in ("legacy_curves", "unified_curves"):
        for name, values in results[strategy].items():
            values = np.asarray(values)
            if values.shape != angles.shape or not np.all(np.isfinite(values)):
                raise ValueError(f"{strategy} {name} is non-finite or has the wrong shape")
            if np.any(values <= 0.0):
                raise ValueError(f"{strategy} {name} contains non-positive intensity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-slices", type=int, default=10)
    parser.add_argument("--max-slice-thickness", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = simulate_comparison(
        min_slices=args.min_slices,
        max_slice_thickness=args.max_slice_thickness,
    )
    validate_results(results)
    metrics = difference_metrics(results)
    figure_path = args.output_dir / "legacy_vs_unified_slicing.png"
    data_path = args.output_dir / "legacy_vs_unified_slicing.csv"
    summary_path = args.output_dir / "legacy_vs_unified_slicing_summary.txt"
    plot_comparison(figure_path, results, metrics)
    save_pointwise_csv(data_path, results)
    save_summary(summary_path, results, metrics)
    print(summary_path.read_text(encoding="utf-8"), end="")
    print(f"Saved {figure_path}")
    print(f"Saved {data_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
