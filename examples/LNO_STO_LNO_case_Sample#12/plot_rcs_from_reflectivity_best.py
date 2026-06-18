"""Simulate Sample#12 RCs using the best reflectivity-only BO result.

The reflectivity fit does not constrain the carbon layer or an RC angle offset,
so this script fixes the carbon thickness and sets the RC angle offset to zero
by default.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

from swxps import (  # noqa: E402
    FittingProblem,
    LayerTemplate,
    RockingCurveData,
    StackTemplate,
)

import fit_sample12_bo as sample12  # noqa: E402


DEFAULT_HISTORY = CASE_DIR / "sample12_reflectivity_bo_history.csv"
DEFAULT_OUTPUT = CASE_DIR / "sample12_rcs_from_reflectivity_best_c2_lno50.png"


def load_best_reflectivity_parameters(path: Path) -> dict[str, float]:
    """Return the best objective row from a reflectivity BO history CSV."""

    values = sample12.load_best_reflectivity_values(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} does not contain any BO evaluations")

    best = min(rows, key=lambda row: float(row["objective"]))
    for name in values:
        if name in best and best[name] != "":
            values[name] = float(best[name])
    return values


def build_graded_rc_stack(values: dict[str, float]):
    """Build an RC stack using the reflectivity-fitted graded superlattice."""

    top_lno_signal_thickness = values["top_lno_signal_thickness"]
    top_lno_buried_thickness = values["top_lno_buried_thickness"]
    if top_lno_buried_thickness < 0.0:
        raise ValueError("top LNO buried thickness must be non-negative")

    parts = (
        LayerTemplate.vacuum(),
        LayerTemplate.from_file(
            "C",
            sample12.C_OPC_FILE,
            "carbon_thickness",
            "carbon_roughness",
        ),
        LayerTemplate.from_file(
            "LNO",
            sample12.LNO_OPC_FILE,
            top_lno_signal_thickness,
            "cap_roughness",
        ),
        LayerTemplate.from_file(
            "LNO",
            sample12.LNO_OPC_FILE,
            top_lno_buried_thickness,
            0.0,
        ),
        *sample12.sample12_graded_superlattice_templates(values),
        LayerTemplate.from_file(
            "STO",
            sample12.STO_OPC_FILE,
            0.0,
            "substrate_roughness",
        ),
    )
    return StackTemplate(
        energy_ev=sample12.PHOTON_ENERGY_EV,
        base_dir=sample12.REPO_ROOT,
        parts=parts,
    ).build(values)


def make_rc_problem(data: sample12.PreparedData) -> FittingProblem:
    """Create a simulation-only RC problem for the three measured core levels."""

    rocking_curves = tuple(
        RockingCurveData(
            name,
            data.rc_angle,
            data.rc_normalized[name],
            weight=sample12.DATASET_WEIGHTS[name],
        )
        for name in sample12.RC_NAMES
    )
    return FittingProblem(
        parameters=sample12.PARAMETERS,
        stack_builder=build_graded_rc_stack,
        photon_energy_ev=sample12.PHOTON_ENERGY_EV,
        rocking_curves=rocking_curves,
        core_levels=sample12.core_level_requests(),
        angle_offset_parameter="rc_angle_offset",
        field_step=5.0,
        roughness_step=2.0,
        offpeak_mask=np.ones_like(data.rc_angle, dtype=bool),
    )


def plot_rc_comparison(
    path: Path,
    data: sample12.PreparedData,
    simulation,
) -> None:
    """Plot normalized experimental RCs against simulated RCs."""

    plt = sample12._load_pyplot()
    simulated = {
        core.name: core.curve.intensity
        for core in simulation.rocking_curves.core_levels
    }

    fig, axes = plt.subplots(
        len(sample12.RC_NAMES),
        1,
        figsize=(7.6, 6.4),
        sharex=True,
    )
    experimental_colors = {
        "C 1s": "tab:green",
        "Ni 3p": "tab:blue",
        "La 4d": "tab:orange",
    }
    axes = np.asarray(axes).ravel()
    for ax, name in zip(axes, sample12.RC_NAMES):
        ax.plot(
            data.rc_angle,
            data.rc_normalized[name],
            "o",
            color=experimental_colors[name],
            markersize=3,
            alpha=0.55,
            label="experiment",
        )
        ax.plot(
            simulation.rocking_curves.angle,
            simulated[name],
            color="black",
            linewidth=1.5,
            label="simulation",
        )
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Grazing incidence angle (deg)")
    fig.suptitle("Sample#12 RCs from Reflectivity BO Parameters")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def rc_comparison_metrics(data: sample12.PreparedData, simulation) -> dict[str, float]:
    """Return mean-squared residuals for each normalized RC."""

    simulated = {
        core.name: core.curve.intensity
        for core in simulation.rocking_curves.core_levels
    }
    return {
        name: float(np.mean((data.rc_normalized[name] - simulated[name]) ** 2))
        for name in sample12.RC_NAMES
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--background-percent", type=float, default=10.0)
    parser.add_argument("--background-order", type=int, default=2)
    parser.add_argument(
        "--carbon-thickness",
        type=float,
        default=2.0,
        help="Fixed top C layer thickness in Angstrom.",
    )
    parser.add_argument(
        "--top-lno-signal-thickness",
        type=float,
        default=50.0,
        help="Thickness of the top emitting LNO cap slab in Angstrom.",
    )
    parser.add_argument(
        "--rc-angle-offset",
        type=float,
        default=0.0,
        help="Optional RC angle offset in degrees.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = load_best_reflectivity_parameters(args.history)
    values["carbon_thickness"] = args.carbon_thickness
    values["top_lno_signal_thickness"] = args.top_lno_signal_thickness
    values["top_lno_buried_thickness"] = (
        values["top_lno_thickness"] - args.top_lno_signal_thickness
    )
    values["rc_angle_offset"] = args.rc_angle_offset

    data = sample12.load_and_prepare_data(
        args.background_percent,
        args.background_order,
    )
    problem = make_rc_problem(data)
    simulation = problem.simulate(values)
    plot_rc_comparison(args.output, data, simulation)
    metrics = rc_comparison_metrics(data, simulation)

    print(f"Saved {args.output}")
    print("Mean-squared normalized RC residuals:")
    for name in sample12.RC_NAMES:
        print(f"  {name}: {metrics[name]:.6g}")
    print("Parameters used for RC simulation:")
    for name in (
        "carbon_thickness",
        "carbon_roughness",
        "top_lno_thickness",
        "top_lno_signal_thickness",
        "top_lno_buried_thickness",
        "sto_thickness_start",
        "lno_thickness_start",
        "sto_thickness_delta",
        "lno_thickness_delta",
        "thickness_transition_repeat",
        "thickness_transition_width",
        "cap_roughness",
        "sto_roughness_first",
        "sto_roughness_last",
        "lno_roughness_first",
        "lno_roughness_last",
        "substrate_roughness",
        "rc_angle_offset",
    ):
        print(f"  {name}: {values[name]:.6g}")


if __name__ == "__main__":
    main()
