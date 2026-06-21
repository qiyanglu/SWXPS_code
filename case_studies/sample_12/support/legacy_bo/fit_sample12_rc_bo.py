"""Fit Sample#12 C 1s, Ni 3p, and La 4d RCs with Bayesian optimization.

The fit starts from the best reflectivity-only BO result. It keeps the top C
layer thickness fixed at 2 A and splits the top LNO cap into a 50 A rough
emitting slab plus a zero-roughness buried remainder.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
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

from swxps import (  # noqa: E402
    BayesianOptimizationSettings,
    FittingProblem,
    LayerTemplate,
    RockingCurveData,
    StackTemplate,
    plot_best_fit,
    plot_fit_convergence,
    plot_stack_schematic,
    plot_surrogate_slices,
    run_bayesian_optimization,
    save_fit_history_csv,
)

import fit_sample12_bo as sample12  # noqa: E402


DEFAULT_HISTORY = CASE_DIR / "sample12_reflectivity_bo_history.csv"
OUTPUT_PREFIX = "sample12_rc_bo_c2_lno50"

RC_FIT_PARAMETER_NAMES = (
    "top_lno_thickness",
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
)


def load_best_reflectivity_parameters(path: Path) -> dict[str, float]:
    """Return Sample#12 defaults updated with the best reflectivity BO row."""

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


def fixed_initial_values(
    reflectivity_history: Path,
    carbon_thickness: float,
    carbon_roughness: float,
    top_lno_signal_thickness: float,
    rc_angle_offset: float,
) -> dict[str, float]:
    """Build the fixed/initial value dictionary used by the RC problem."""

    values = load_best_reflectivity_parameters(reflectivity_history)
    values["carbon_thickness"] = carbon_thickness
    values["carbon_roughness"] = carbon_roughness
    values["top_lno_signal_thickness"] = top_lno_signal_thickness
    values["top_lno_buried_thickness"] = (
        values["top_lno_thickness"] - top_lno_signal_thickness
    )
    values["rc_angle_offset"] = rc_angle_offset
    return values


def rc_fit_parameters(initial_values: dict[str, float]):
    """Return active RC BO parameters with initials seeded from reflectivity."""

    return tuple(
        replace(
            sample12.PARAMETER_BY_NAME[name],
            initial=initial_values.get(name, sample12.PARAMETER_BY_NAME[name].initial),
        )
        for name in RC_FIT_PARAMETER_NAMES
    )


def build_rc_stack(values: dict[str, float]):
    """Build the RC stack with fixed C thickness and split top LNO cap."""

    top_lno_signal_thickness = values["top_lno_signal_thickness"]
    top_lno_buried_thickness = values["top_lno_thickness"] - top_lno_signal_thickness
    if top_lno_buried_thickness < 0.0:
        raise ValueError("top LNO signal slab cannot exceed total cap thickness")

    stack_values = {
        **values,
        "top_lno_buried_thickness": top_lno_buried_thickness,
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
            top_lno_signal_thickness,
            "cap_roughness",
        ),
        LayerTemplate.from_file(
            "LNO",
            sample12.LNO_OPC_FILE,
            top_lno_buried_thickness,
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


def make_rc_problem(data: sample12.PreparedData, initial_values: dict[str, float]):
    """Create the three-RC fitting problem."""

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
        parameters=rc_fit_parameters(initial_values),
        stack_builder=build_rc_stack,
        photon_energy_ev=sample12.PHOTON_ENERGY_EV,
        rocking_curves=rocking_curves,
        core_levels=sample12.core_level_requests(),
        angle_offset_parameter="rc_angle_offset",
        field_step=5.0,
        roughness_step=2.0,
        offpeak_mask=np.ones_like(data.rc_angle, dtype=bool),
        fixed_values=initial_values,
    )


def save_rc_overlay(path: Path, data: sample12.PreparedData, simulation) -> None:
    """Save a compact normalized RC overlay."""

    plt = sample12._load_pyplot()
    simulated = {
        core.name: core.curve.intensity
        for core in simulation.rocking_curves.core_levels
    }
    colors = {
        "C 1s": "tab:green",
        "Ni 3p": "tab:blue",
        "La 4d": "tab:orange",
    }
    fig, axes = plt.subplots(len(sample12.RC_NAMES), 1, figsize=(7.6, 6.4), sharex=True)
    axes = np.asarray(axes).ravel()
    for ax, name in zip(axes, sample12.RC_NAMES):
        ax.plot(
            data.rc_angle,
            data.rc_normalized[name],
            "o",
            color=colors[name],
            markersize=3,
            alpha=0.55,
            label="experiment",
        )
        ax.plot(
            simulation.rocking_curves.angle,
            simulated[name],
            color="black",
            linewidth=1.5,
            label="BO fit",
        )
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Grazing incidence angle (deg)")
    fig.suptitle("Sample#12 Three-RC BO Fit")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def rc_residuals(data: sample12.PreparedData, simulation) -> dict[str, float]:
    """Return mean-squared residuals for each normalized RC."""

    simulated = {
        core.name: core.curve.intensity
        for core in simulation.rocking_curves.core_levels
    }
    return {
        name: float(np.mean((data.rc_normalized[name] - simulated[name]) ** 2))
        for name in sample12.RC_NAMES
    }


def print_timing_summary(result) -> None:
    """Print optimizer and forward-model timing diagnostics."""

    print("Timing summary:")
    print(f"  total optimizer wall time: {result.timing.total_seconds:.3f} s")
    print(f"  objective evaluations: {result.timing.evaluations}")
    print(f"  objective/forward time: {result.timing.objective_seconds:.3f} s")
    print(f"  BO/GP overhead time: {result.timing.optimizer_overhead_seconds:.3f} s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--background-percent", type=float, default=10.0)
    parser.add_argument("--background-order", type=int, default=2)
    parser.add_argument("--carbon-thickness", type=float, default=2.0)
    parser.add_argument("--carbon-roughness", type=float, default=2.0)
    parser.add_argument("--top-lno-signal-thickness", type=float, default=50.0)
    parser.add_argument("--rc-angle-offset", type=float, default=0.0)
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
    initial_values = fixed_initial_values(
        args.history,
        args.carbon_thickness,
        args.carbon_roughness,
        args.top_lno_signal_thickness,
        args.rc_angle_offset,
    )
    data = sample12.load_and_prepare_data(
        args.background_percent,
        args.background_order,
    )
    problem = make_rc_problem(data, initial_values)
    print("Sample#12 three-RC BO fitting setup")
    print(f"Reflectivity seed: {args.history}")
    print(f"Fixed C thickness: {initial_values['carbon_thickness']:g} A")
    print(f"Fixed C roughness: {initial_values['carbon_roughness']:g} A")
    print(f"Fixed top LNO emitting slab: {initial_values['top_lno_signal_thickness']:g} A")
    print("Active parameters:")
    for parameter in problem.parameters:
        unit = f" {parameter.unit}" if parameter.unit else ""
        print(
            f"  {parameter.name}: {parameter.lower:g} to {parameter.upper:g}{unit}, "
            f"initial={parameter.initial:g}{unit}"
        )

    result = run_bayesian_optimization(
        problem.objective(),
        BayesianOptimizationSettings(
            n_calls=args.n_calls,
            n_initial_points=args.n_initial_points,
            random_state=args.random_state,
            show_progress=args.progress,
            progress_interval=args.progress_interval,
        ),
    )
    best_values = {**initial_values, **result.best_parameters}
    best_values["top_lno_buried_thickness"] = (
        best_values["top_lno_thickness"] - best_values["top_lno_signal_thickness"]
    )
    best_simulation = problem.simulate(best_values)

    history_path = CASE_DIR / f"{OUTPUT_PREFIX}_history.csv"
    convergence_path = CASE_DIR / f"{OUTPUT_PREFIX}_convergence.png"
    best_fit_path = CASE_DIR / f"{OUTPUT_PREFIX}_best_fit.png"
    overlay_path = CASE_DIR / f"{OUTPUT_PREFIX}_overlay.png"
    surrogate_path = CASE_DIR / f"{OUTPUT_PREFIX}_surrogate_slices.png"
    stack_path = CASE_DIR / f"{OUTPUT_PREFIX}_stack_schematic.png"
    profile_path = CASE_DIR / f"{OUTPUT_PREFIX}_superlattice_profile.png"

    save_fit_history_csv(history_path, result.history, problem.parameters)
    plot_fit_convergence(convergence_path, result.history)
    plot_best_fit(best_fit_path, problem.reflectivity, problem.rocking_curves, best_simulation)
    save_rc_overlay(overlay_path, data, best_simulation)
    plot_surrogate_slices(surrogate_path, result, problem.parameters, initial_values)
    plot_stack_schematic(
        stack_path,
        best_simulation.stack,
        title="Sample#12 Three-RC BO Stack",
        top_layers=5,
        bottom_layers=3,
    )
    sample12.save_superlattice_profile_plot(best_values, profile_path)

    print(f"Best objective: {result.best_objective:.6g}")
    print("Mean-squared normalized RC residuals:")
    for name, value in rc_residuals(data, best_simulation).items():
        print(f"  {name}: {value:.6g}")
    print("Best parameters:")
    for parameter in problem.parameters:
        value = result.best_parameters[parameter.name]
        unit = f" {parameter.unit}" if parameter.unit else ""
        print(f"  {parameter.name}: {value:.6g}{unit}")
    print("Fixed/derived parameters:")
    for name in (
        "carbon_thickness",
        "carbon_roughness",
        "top_lno_signal_thickness",
        "top_lno_buried_thickness",
    ):
        print(f"  {name}: {best_values[name]:.6g}")
    print_timing_summary(result)
    print("Saved outputs:")
    for path in (
        history_path,
        convergence_path,
        best_fit_path,
        overlay_path,
        surrogate_path,
        stack_path,
        profile_path,
    ):
        print(f"  {path}")


if __name__ == "__main__":
    main()
