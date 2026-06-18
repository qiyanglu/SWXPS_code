"""Simulate Sample#13 RCs using the best reflectivity-only BO result.

The reflectivity fit does not constrain the carbon layer or an RC angle offset,
so this script keeps the default carbon values from ``fit_sample13_bo.py`` and
sets the RC angle offset to zero by default.
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
from swxps.optical_constants import load_optical_constants  # noqa: E402

import fit_sample13_bo as sample13  # noqa: E402


DEFAULT_HISTORY = CASE_DIR / "sample13_reflectivity_bo_history.csv"
DEFAULT_OUTPUT = CASE_DIR / "sample13_rcs_from_reflectivity_best.png"


def load_best_reflectivity_parameters(path: Path) -> dict[str, float]:
    """Return the best objective row from a reflectivity BO history CSV."""

    values = sample13.load_best_reflectivity_values(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} does not contain any BO evaluations")

    best = min(rows, key=lambda row: float(row["objective"]))
    for name in values:
        if name in best and best[name] != "":
            values[name] = float(best[name])
    return values


def preflight_optical_constants() -> None:
    """Load optical constants once before stack building."""

    for relative_path in (sample13.C_OPC_FILE, sample13.LNO_OPC_FILE, sample13.STO_OPC_FILE):
        load_optical_constants(sample13.REPO_ROOT / relative_path)


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
            sample13.C_OPC_FILE,
            "carbon_thickness",
            "carbon_roughness",
        ),
        LayerTemplate.from_file(
            "LNO",
            sample13.LNO_OPC_FILE,
            top_lno_signal_thickness,
            "cap_roughness",
        ),
        LayerTemplate.from_file(
            "LNO",
            sample13.LNO_OPC_FILE,
            top_lno_buried_thickness,
            0.0,
        ),
        *sample13.sample13_graded_superlattice_templates(values),
        LayerTemplate.from_file(
            "STO",
            sample13.STO_OPC_FILE,
            0.0,
            "substrate_roughness",
        ),
    )
    return StackTemplate(
        energy_ev=sample13.PHOTON_ENERGY_EV,
        base_dir=sample13.REPO_ROOT,
        parts=parts,
    ).build(values)


def make_rc_problem(data: sample13.PreparedData) -> FittingProblem:
    """Create a simulation-only RC problem for the three measured core levels."""

    rocking_curves = tuple(
        RockingCurveData(
            name,
            data.rc_angle,
            data.rc_normalized[name],
            weight=sample13.DATASET_WEIGHTS[name],
        )
        for name in sample13.RC_NAMES
    )
    return FittingProblem(
        parameters=sample13.PARAMETERS,
        stack_builder=build_graded_rc_stack,
        photon_energy_ev=sample13.PHOTON_ENERGY_EV,
        rocking_curves=rocking_curves,
        core_levels=sample13.core_level_requests(),
        angle_offset_parameter="rc_angle_offset",
        field_step=5.0,
        roughness_step=2.0,
        offpeak_mask=np.ones_like(data.rc_angle, dtype=bool),
    )


def plot_rc_comparison(
    path: Path,
    data: sample13.PreparedData,
    simulation,
) -> None:
    """Plot normalized experimental RCs against simulated RCs."""

    plt = sample13._load_pyplot()
    simulated = {
        core.name: core.curve.intensity
        for core in simulation.rocking_curves.core_levels
    }

    fig, axes = plt.subplots(
        len(sample13.RC_NAMES),
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
    for ax, name in zip(axes, sample13.RC_NAMES):
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
    fig.suptitle("Sample#13 RCs from Reflectivity BO Parameters")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--background-percent", type=float, default=10.0)
    parser.add_argument("--background-order", type=int, default=2)
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
    values["top_lno_signal_thickness"] = args.top_lno_signal_thickness
    values["top_lno_buried_thickness"] = (
        values["top_lno_thickness"] - args.top_lno_signal_thickness
    )
    values["rc_angle_offset"] = args.rc_angle_offset

    data = sample13.load_and_prepare_data(
        args.background_percent,
        args.background_order,
    )
    preflight_optical_constants()
    problem = make_rc_problem(data)
    simulation = problem.simulate(values)
    plot_rc_comparison(args.output, data, simulation)

    print(f"Saved {args.output}")
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
