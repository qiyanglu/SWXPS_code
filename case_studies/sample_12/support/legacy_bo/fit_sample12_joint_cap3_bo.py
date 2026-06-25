"""Fit Sample#12 reflectivity plus C 1s, Ni 3p, and La 4d RCs.

The cap stack is:

vacuum / C / LNO-1 / LNO-2 / LNO-bottom / [STO/LNO]x40 / STO substrate

LNO-bottom is fixed at 160 A with zero roughness. LNO-1 and LNO-2 have a
fitted total thickness, while LNO-1 has fitted thickness and roughness and
LNO-2 gets the remaining thickness with zero roughness.

By default this script only prints the proposed fitting setup. Use
``--run-fit`` after checking the parameter list.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = CASE_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

from swanx.diagnostics import (

    plot_fit_convergence,

    plot_stack_schematic,

    plot_surrogate_slices,

    save_fit_history_csv,

)

from swanx.fitting import (

    BayesianOptimizationSettings,

    FitParameter,

    FitSimulation,

    FittingProblem,

    JointObjective,

    ReflectivityData,

    RockingCurveData,

    evaluation_from_contributions,

    run_bayesian_optimization,

)

from swanx.imfp import imfp_from_file

from swanx.stack import (

    LayerTemplate,

    StackTemplate,

)

from swanx.workflows.simulate import CoreLevelRequest

import fit_sample12_bo as sample12  # noqa: E402


OUTPUT_PREFIX = "sample12_joint_cap3_bo"
BOTTOM_LNO_THICKNESS = 160.0
DEFAULT_REFLECTIVITY_WEIGHT = "auto"

PARAMETERS = (
    FitParameter("carbon_thickness", 2.0, 15.0, "Angstrom", initial=2.0),
    FitParameter("carbon_roughness_fraction", 0.0, 1.0, "", initial=1.0 / 7.0),
    FitParameter("top_lno_total_thickness", 45.0, 65.0, "Angstrom", initial=50.0),
    FitParameter("top_lno_layer1_thickness", 1.01, 30.0, "Angstrom", initial=5.0),
    FitParameter("cap_roughness_fraction", 0.0, 1.0, "", initial=0.425),
    FitParameter("sto_thickness_start", 13.0, 18.5, "Angstrom", initial=14.93),
    FitParameter("lno_thickness_start", 13.0, 18.5, "Angstrom", initial=15.79),
    FitParameter("sto_thickness_delta", 0.0, 3.0, "Angstrom", initial=2.99),
    FitParameter("lno_thickness_delta", 0.0, 3.0, "Angstrom", initial=1.99),
    FitParameter("thickness_transition_repeat", 0.0, 39.0, "repeat", initial=1.9),
    FitParameter("thickness_transition_width", 1.0, 20.0, "repeat", initial=18.98),
    FitParameter("sto_roughness_first", 2.0, 5.0, "Angstrom", initial=3.08),
    FitParameter("sto_roughness_last", 2.0, 5.0, "Angstrom", initial=2.78),
    FitParameter("lno_roughness_first", 2.0, 5.0, "Angstrom", initial=3.66),
    FitParameter("lno_roughness_last", 2.0, 5.0, "Angstrom", initial=3.32),
    FitParameter("substrate_roughness", 1.0, 5.0, "Angstrom", initial=3.25),
    FitParameter("reflectivity_angle_offset", -0.30, 0.30, "deg", initial=0.153),
    FitParameter("rc_angle_offset", -0.30, 0.30, "deg", initial=0.029),
)

PARAMETER_BY_NAME = {parameter.name: parameter for parameter in PARAMETERS}

DATASET_WEIGHTS = {
    "C 1s": 0.5,
    "La 4d": 3.0,
    "Ni 3p": 3.0,
}


@dataclass(frozen=True)
class JointCap3Problem:
    """Joint reflectivity/RC problem with independent angle offsets."""

    parameters: tuple[FitParameter, ...]
    reflectivity_problem: FittingProblem
    rc_problem: FittingProblem

    @property
    def reflectivity(self):
        return self.reflectivity_problem.reflectivity

    @property
    def rocking_curves(self):
        return self.rc_problem.rocking_curves

    def objective(self) -> JointObjective:
        return JointObjective(self.parameters, self.evaluate)

    def evaluate(self, values: dict[str, float]):
        reflectivity_evaluation = self.reflectivity_problem.evaluate(values)
        rc_evaluation = self.rc_problem.evaluate(values)
        timings = _combine_timings(
            reflectivity_evaluation.timings,
            rc_evaluation.timings,
        )
        return evaluation_from_contributions(
            values,
            (*reflectivity_evaluation.contributions, *rc_evaluation.contributions),
            timings=timings,
        )

    def simulate(self, values: dict[str, float]) -> FitSimulation:
        reflectivity_simulation = self.reflectivity_problem.simulate(values)
        rc_simulation = self.rc_problem.simulate(values)
        return FitSimulation(
            parameters=dict(values),
            stack=rc_simulation.stack,
            reflectivity=reflectivity_simulation.reflectivity,
            rocking_curves=rc_simulation.rocking_curves,
        )


def initial_values() -> dict[str, float]:
    """Return initial parameter values."""

    return {
        parameter.name: float(parameter.initial)
        for parameter in PARAMETERS
        if parameter.initial is not None
    }


def carbon_roughness_from_values(values: dict[str, float]) -> float:
    """Return a C roughness in [1, min(8, carbon thickness)] Angstrom."""

    carbon_thickness = values["carbon_thickness"]
    max_roughness = min(8.0, carbon_thickness)
    fraction = values["carbon_roughness_fraction"]
    return 1.0 + fraction * (max_roughness - 1.0)


def cap_roughness_from_values(values: dict[str, float]) -> float:
    """Return LNO layer-1 roughness in [1, min(5, layer thickness)] Angstrom."""

    layer1_thickness = values["top_lno_layer1_thickness"]
    max_roughness = min(5.0, layer1_thickness - 1.0e-6)
    if max_roughness < 1.0:
        raise ValueError("top LNO layer 1 thickness must be at least 1 Angstrom")
    fraction = values["cap_roughness_fraction"]
    return 1.0 + fraction * (max_roughness - 1.0)


def build_cap3_stack(values: dict[str, float]):
    """Build the Sample#12 joint-fit stack with the revised three-layer LNO cap."""

    layer1_thickness = values["top_lno_layer1_thickness"]
    layer2_thickness = values["top_lno_total_thickness"] - layer1_thickness
    if layer2_thickness < 0.0:
        raise ValueError("top LNO layer 1 cannot exceed total top-LNO thickness")
    stack_values = {
        **values,
        "carbon_roughness": carbon_roughness_from_values(values),
        "cap_roughness": cap_roughness_from_values(values),
        "top_lno_layer2_thickness": layer2_thickness,
    }
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
            "top_lno_layer1_thickness",
            "cap_roughness",
        ),
        LayerTemplate.from_file(
            "LNO",
            sample12.LNO_OPC_FILE,
            layer2_thickness,
            0.0,
        ),
        LayerTemplate.from_file(
            "LNO",
            sample12.LNO_OPC_FILE,
            BOTTOM_LNO_THICKNESS,
            0.0,
        ),
        *sample12.sample12_graded_superlattice_templates(stack_values),
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
    ).build(stack_values)


def core_level_requests() -> tuple[CoreLevelRequest, ...]:
    """Return layer-selective C/Ni/La requests for the revised cap stack."""

    imfp_files = {
        "C": sample12.REPO_ROOT / "examples" / "data" / "IMFP" / "C.ANG",
        "LNO": sample12.REPO_ROOT / "examples" / "data" / "IMFP" / "LNO.ANG",
        "STO": sample12.REPO_ROOT / "examples" / "data" / "IMFP" / "STO.ANG",
    }
    imfp_by_core = {}
    for core_name, binding_energy in sample12.BINDING_ENERGIES.items():
        kinetic_energy = sample12.PHOTON_ENERGY_EV - binding_energy
        imfp_by_core[core_name] = {
            material: imfp_from_file(path, kinetic_energy)
            for material, path in imfp_files.items()
        }

    return (
        CoreLevelRequest(
            name="C 1s",
            binding_energy_ev=sample12.BINDING_ENERGIES["C 1s"],
            concentration_by_material={"C": 1.0},
            imfp_by_material={"vacuum": imfp_by_core["C 1s"]["C"], **imfp_by_core["C 1s"]},
            emitting_layer_indices=(1,),
        ),
        CoreLevelRequest(
            name="Ni 3p",
            binding_energy_ev=sample12.BINDING_ENERGIES["Ni 3p"],
            concentration_by_material={"LNO": 1.0},
            imfp_by_material={"vacuum": imfp_by_core["Ni 3p"]["C"], **imfp_by_core["Ni 3p"]},
            emitting_layer_indices=(2, 3),
        ),
        CoreLevelRequest(
            name="La 4d",
            binding_energy_ev=sample12.BINDING_ENERGIES["La 4d"],
            concentration_by_material={"LNO": 1.0},
            imfp_by_material={"vacuum": imfp_by_core["La 4d"]["C"], **imfp_by_core["La 4d"]},
            emitting_layer_indices=(2, 3),
        ),
    )


def make_problem(
    data: sample12.PreparedData,
    reflectivity_weight: float,
) -> JointCap3Problem:
    """Create a joint reflectivity plus three-RC problem."""

    rc_data = tuple(
        RockingCurveData(
            name,
            data.rc_angle,
            data.rc_normalized[name],
            weight=DATASET_WEIGHTS[name],
        )
        for name in sample12.RC_NAMES
    )
    fixed_values = {
        "carbon_roughness": carbon_roughness_from_values(initial_values()),
        "cap_roughness": cap_roughness_from_values(initial_values()),
        "top_lno_layer2_thickness": 45.0,
        "bottom_lno_thickness": BOTTOM_LNO_THICKNESS,
    }
    reflectivity_problem = FittingProblem(
        parameters=PARAMETERS,
        stack_builder=build_cap3_stack,
        photon_energy_ev=sample12.PHOTON_ENERGY_EV,
        reflectivity=ReflectivityData(
            name="reflectivity",
            angles=data.reflectivity_angle,
            reflectivity=data.reflectivity_raw,
            weight=reflectivity_weight,
            log_floor=1.0e-12,
        ),
        angle_offset_parameter="reflectivity_angle_offset",
        field_step=5.0,
        roughness_step=2.0,
        slicing=None,
        fixed_values=fixed_values,
    )
    rc_problem = FittingProblem(
        parameters=PARAMETERS,
        stack_builder=build_cap3_stack,
        photon_energy_ev=sample12.PHOTON_ENERGY_EV,
        rocking_curves=rc_data,
        core_levels=core_level_requests(),
        angle_offset_parameter="rc_angle_offset",
        field_step=5.0,
        roughness_step=2.0,
        slicing=None,
        offpeak_mask=np.ones_like(data.rc_angle, dtype=bool),
        fixed_values=fixed_values,
    )
    return JointCap3Problem(
        parameters=PARAMETERS,
        reflectivity_problem=reflectivity_problem,
        rc_problem=rc_problem,
    )


def resolve_reflectivity_weight(
    data: sample12.PreparedData,
    requested_weight: str,
) -> tuple[float, dict[str, float]]:
    """Return the reflectivity weight and initial contribution diagnostics."""

    if requested_weight != "auto":
        return float(requested_weight), {}

    diagnostic_problem = make_problem(data, reflectivity_weight=1.0)
    evaluation = diagnostic_problem.evaluate(initial_values())
    raw = {contribution.name: contribution.raw for contribution in evaluation.contributions}
    weighted = {
        contribution.name: contribution.weighted
        for contribution in evaluation.contributions
    }
    rc_weighted = sum(
        weighted[name]
        for name in ("C 1s", "Ni 3p", "La 4d")
    )
    reflectivity_raw = raw["reflectivity"]
    if reflectivity_raw <= 0:
        return 1.0, {"reflectivity_raw": reflectivity_raw, "rc_weighted": rc_weighted}
    return rc_weighted / reflectivity_raw, {
        "reflectivity_raw": reflectivity_raw,
        "rc_weighted": rc_weighted,
    }


def print_setup(
    data: sample12.PreparedData,
    reflectivity_weight_request: str,
    reflectivity_weight: float,
    diagnostics: dict[str, float],
) -> None:
    """Print stack, dataset weights, and fitting parameter ranges."""

    values = initial_values()
    carbon_roughness = carbon_roughness_from_values(values)
    cap_roughness = cap_roughness_from_values(values)
    layer2 = values["top_lno_total_thickness"] - values["top_lno_layer1_thickness"]
    print("Sample#12 joint cap3 BO setup")
    print(f"Photon energy: {sample12.PHOTON_ENERGY_EV:g} eV")
    print(
        f"Reflectivity data: {data.reflectivity_angle[0]:g} to "
        f"{data.reflectivity_angle[-1]:g} deg, {len(data.reflectivity_angle)} points"
    )
    print(
        f"RC data: {data.rc_angle[0]:g} to {data.rc_angle[-1]:g} deg, "
        f"{len(data.rc_angle)} points"
    )
    print()
    print("Stack model, top to bottom:")
    print("  vacuum")
    print("  C: thickness=carbon_thickness, roughness=carbon_roughness")
    print("  LNO layer 1: thickness=top_lno_layer1_thickness, roughness=cap_roughness")
    print("  LNO layer 2: thickness=top_lno_total_thickness - top_lno_layer1_thickness, roughness=0")
    print(f"  LNO bottom cap: fixed thickness={BOTTOM_LNO_THICKNESS:g} A, roughness=0")
    print(f"  [STO/LNO] x {sample12.SUPERLATTICE_REPEATS}, erf-like graded thickness profile")
    print("  STO substrate: semi-infinite, roughness=substrate_roughness")
    print()
    print("Initial derived thicknesses:")
    print(f"  carbon_roughness: {carbon_roughness:g} A")
    print(f"  cap_roughness: {cap_roughness:g} A")
    print(f"  top_lno_layer2_thickness: {layer2:g} A")
    print(f"  total LNO cap thickness: {values['top_lno_total_thickness'] + BOTTOM_LNO_THICKNESS:g} A")
    print()
    print("Layer-selective RC emission:")
    print("  C 1s: C layer only")
    print("  Ni 3p: LNO layer 1 + LNO layer 2")
    print("  La 4d: LNO layer 1 + LNO layer 2")
    print()
    print("Dataset weights:")
    if reflectivity_weight_request == "auto":
        print(
            "  reflectivity: auto weight = "
            f"{reflectivity_weight:.6g} "
            "(initial weighted reflectivity log-MSE matched to total weighted RC MSE)"
        )
        if diagnostics:
            print(f"    initial reflectivity raw log-MSE: {diagnostics['reflectivity_raw']:.6g}")
            print(f"    initial total weighted RC MSE: {diagnostics['rc_weighted']:.6g}")
    else:
        print(f"  reflectivity: {reflectivity_weight:.6g}")
    print("  C 1s: 0.5")
    print("  La 4d: 3")
    print("  Ni 3p: 3")
    print()
    print("Fitting parameters:")
    for parameter in PARAMETERS:
        unit = f" {parameter.unit}" if parameter.unit else ""
        print(
            f"  {parameter.name}: {parameter.lower:g} to {parameter.upper:g}{unit}, "
            f"initial={parameter.initial:g}{unit}"
        )
    print("  derived carbon_roughness: 1 to min(8, carbon_thickness) Angstrom")
    print("  derived cap_roughness: 1 to min(5, top_lno_layer1_thickness) Angstrom")


def plot_joint_best_fit(
    path: Path,
    reflectivity_data: ReflectivityData,
    rocking_curve_data: tuple[RockingCurveData, ...],
    simulation: FitSimulation,
) -> None:
    """Save joint fit overlays with distinct data colors for each panel."""

    plt = sample12._load_pyplot()
    fig, axes = plt.subplots(4, 1, figsize=(7.6, 8.2), sharex=False)
    colors = {
        "reflectivity": "tab:purple",
        "C 1s": "tab:green",
        "Ni 3p": "tab:blue",
        "La 4d": "tab:orange",
    }

    ax = axes[0]
    ax.semilogy(
        reflectivity_data.angles,
        reflectivity_data.reflectivity,
        "o",
        color=colors["reflectivity"],
        markersize=3,
        label="data",
    )
    ax.semilogy(
        simulation.reflectivity.angle,
        simulation.reflectivity.reflectivity,
        color="black",
        linewidth=1.5,
        label="fit",
    )
    ax.set_ylabel("reflectivity")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")

    simulated_rc = {
        core.name: core.curve.intensity
        for core in simulation.rocking_curves.core_levels
    }
    for ax, data in zip(axes[1:], rocking_curve_data):
        ax.plot(
            data.angles,
            data.intensity,
            "o",
            color=colors[data.name],
            markersize=3,
            alpha=0.55,
            label="data",
        )
        ax.plot(
            simulation.rocking_curves.angle,
            simulated_rc[data.name],
            color="black",
            linewidth=1.5,
            label="fit",
        )
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.set_ylabel(data.name)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Grazing incidence angle (deg)")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_fit(
    data: sample12.PreparedData,
    reflectivity_weight: float,
    output_prefix: str,
    n_calls: int,
    n_initial_points: int,
    random_state: int,
    show_progress: bool,
    progress_interval: int,
) -> None:
    """Run joint BO and save diagnostics."""

    problem = make_problem(data, reflectivity_weight)
    result = run_bayesian_optimization(
        problem.objective(),
        BayesianOptimizationSettings(
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            random_state=random_state,
            show_progress=show_progress,
            progress_interval=progress_interval,
        ),
    )
    best_simulation = problem.simulate(result.best_parameters)
    save_fit_history_csv(CASE_DIR / f"{output_prefix}_history.csv", result.history, PARAMETERS)
    plot_fit_convergence(CASE_DIR / f"{output_prefix}_convergence.png", result.history)
    plot_joint_best_fit(
        CASE_DIR / f"{output_prefix}_best_fit.png",
        problem.reflectivity,
        problem.rocking_curves,
        best_simulation,
    )
    plot_surrogate_slices(
        CASE_DIR / f"{output_prefix}_surrogate_slices.png",
        result,
        PARAMETERS,
        initial_values(),
    )
    plot_stack_schematic(
        CASE_DIR / f"{output_prefix}_stack_schematic.png",
        best_simulation.stack,
        title="Sample#12 Joint Cap3 BO Stack",
        top_layers=6,
        bottom_layers=3,
    )
    print(f"Best objective: {result.best_objective:.6g}")
    print("Best parameters:")
    for parameter in PARAMETERS:
        unit = f" {parameter.unit}" if parameter.unit else ""
        print(f"  {parameter.name}: {result.best_parameters[parameter.name]:.6g}{unit}")
    print("Derived best parameters:")
    print(f"  carbon_roughness: {carbon_roughness_from_values(result.best_parameters):.6g} Angstrom")
    print(f"  cap_roughness: {cap_roughness_from_values(result.best_parameters):.6g} Angstrom")
    print(
        "  top_lno_layer2_thickness: "
        f"{result.best_parameters['top_lno_total_thickness'] - result.best_parameters['top_lno_layer1_thickness']:.6g} Angstrom"
    )


def _combine_timings(*timings: dict[str, float]) -> dict[str, float]:
    combined: dict[str, float] = {}
    for timing in timings:
        for name, value in timing.items():
            combined[name] = combined.get(name, 0.0) + float(value)
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background-percent", type=float, default=10.0)
    parser.add_argument("--background-order", type=int, default=2)
    parser.add_argument(
        "--reflectivity-min-angle",
        type=float,
        default=sample12.RC_START_DEG,
        help="Minimum reflectivity angle included in fitting.",
    )
    parser.add_argument(
        "--reflectivity-max-angle",
        type=float,
        default=sample12.RC_STOP_DEG,
        help="Maximum reflectivity angle included in fitting.",
    )
    parser.add_argument(
        "--reflectivity-weight",
        default=DEFAULT_REFLECTIVITY_WEIGHT,
        help="Reflectivity dataset weight, or 'auto' to balance the initial log-MSE contribution.",
    )
    parser.add_argument(
        "--output-prefix",
        default=OUTPUT_PREFIX,
        help="Prefix for saved BO history and diagnostic plots.",
    )
    parser.add_argument("--run-fit", action="store_true")
    parser.add_argument("--n-calls", type=int, default=80)
    parser.add_argument("--n-initial-points", type=int, default=24)
    parser.add_argument("--random-state", type=int, default=12)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--progress-interval", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = sample12.load_and_prepare_data(args.background_percent, args.background_order)
    data = sample12.apply_reflectivity_window(
        data,
        args.reflectivity_min_angle,
        args.reflectivity_max_angle,
    )
    reflectivity_weight, diagnostics = resolve_reflectivity_weight(
        data,
        args.reflectivity_weight,
    )
    print_setup(data, args.reflectivity_weight, reflectivity_weight, diagnostics)
    if not args.run_fit:
        print()
        print("BO fitting was not run. Re-run with --run-fit after checking this setup.")
        return
    run_fit(
        data,
        reflectivity_weight,
        args.output_prefix,
        args.n_calls,
        args.n_initial_points,
        args.random_state,
        args.progress,
        args.progress_interval,
    )


if __name__ == "__main__":
    main()
