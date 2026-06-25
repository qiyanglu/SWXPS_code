"""Fit Sample#13 reflectivity, C 1s, and Ni 3p with bounded TRF least squares."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import sys

import matplotlib.pyplot as plt
import numpy as np


FIT_DIR = Path(__file__).resolve().parent
CASE_DIR = FIT_DIR.parent
REPO_ROOT = CASE_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
GRADIENT_DIR = CASE_DIR / "jax_gradient_fit_without_la4d"
RUNS_ROOT = REPO_ROOT / "runs" / "sample_13"
GRADIENT_RUN_DIR = RUNS_ROOT / "jax_gradient_without_la4d" / "single_60iter"
START_SUMMARY = (
    GRADIENT_RUN_DIR
    / "sample13_reflectivity_c1s_ni3p_jax_gradient_best_summary.csv"
)
DEFAULT_OUTPUT_DIR = RUNS_ROOT / "jax_least_squares" / "single_run"
FINITE_DIFF_REL_STEP = 2.0e-4
FINITE_DIFF_ABS_STEP = 1.0e-4

for path in (SRC_DIR, GRADIENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from swanx.diagnostics import plot_stack_schematic  # noqa: E402
from swanx.fitting import (  # noqa: E402
    JaxLeastSquaresOptimizerSettings,
    JaxResidualFunction,
    RockingCurveData,
    optimize_with_jax_least_squares,
)
from swanx.imfp import imfp_from_file  # noqa: E402
from swanx.workflows.simulate import CoreLevelRequest  # noqa: E402
from swanx.result_exports import (  # noqa: E402
    save_fit_curve_data_csv,
    save_optimized_stack_csv,
)

import fit_sample13_reflectivity_c1s_ni3p_jax_gradient as fit13  # noqa: E402

BOUND_OVERRIDES = {
    "sto_thickness_start": (12.5, 19.0),
    "lno_thickness_start": (12.5, 19.0),
    "reflectivity_angle_offset": (-0.35, 0.35),
    "rc_angle_offset": (-0.35, 0.35),
}
PARAMETERS = tuple(
    replace(
        parameter,
        lower=BOUND_OVERRIDES[parameter.name][0],
        upper=BOUND_OVERRIDES[parameter.name][1],
    )
    if parameter.name in BOUND_OVERRIDES
    else parameter
    for parameter in fit13.PARAMETERS
)



class ExactResidualModel:
    """Exact high-level residuals with a cached finite-difference Jacobian."""

    def __init__(self, problem):
        self.problem = problem
        self.parameters = PARAMETERS
        self.lower = np.asarray([item.lower for item in self.parameters], dtype=float)
        self.upper = np.asarray([item.upper for item in self.parameters], dtype=float)
        self.widths = self.upper - self.lower
        self.cache: dict[tuple[float, ...], np.ndarray] = {}
        self.residual_count = problem.reflectivity.reflectivity.size + sum(
            data.intensity.size for data in problem.rocking_curves
        )

    def residuals(self, vector) -> np.ndarray:
        physical = np.clip(np.asarray(vector, dtype=float), self.lower, self.upper)
        key = tuple(np.round(physical, 12))
        if key not in self.cache:
            values = {
                parameter.name: float(value)
                for parameter, value in zip(self.parameters, physical)
            }
            simulation = self.problem.simulate(values)
            reflectivity_data = self.problem.reflectivity
            measured = np.asarray(reflectivity_data.reflectivity, dtype=float)
            simulated = np.asarray(simulation.reflectivity.reflectivity, dtype=float)
            floor = reflectivity_data.log_floor
            reflectivity_block = np.log10(np.maximum(measured, floor)) - np.log10(
                np.maximum(simulated, floor)
            )
            if reflectivity_data.sigma is not None:
                reflectivity_block = reflectivity_block / reflectivity_data.sigma
            blocks = [
                np.sqrt(reflectivity_data.weight / reflectivity_block.size)
                * reflectivity_block
            ]

            simulated_by_name = {
                item.name: np.asarray(item.curve.intensity, dtype=float)
                for item in simulation.rocking_curves.core_levels
            }
            for data in self.problem.rocking_curves:
                block = np.asarray(data.intensity, dtype=float) - simulated_by_name[data.name]
                if data.sigma is not None:
                    block = block / data.sigma
                blocks.append(np.sqrt(data.weight / block.size) * block)
            residuals = np.concatenate(blocks)
            if residuals.shape != (self.residual_count,):
                raise ValueError("least-squares residual vector changed shape")
            if not np.all(np.isfinite(residuals)):
                raise ValueError("least-squares residual vector is non-finite")
            self.cache[key] = residuals
        return self.cache[key].copy()

    def jacobian(self, vector) -> np.ndarray:
        physical = np.clip(np.asarray(vector, dtype=float), self.lower, self.upper)
        jacobian = np.empty((self.residual_count, physical.size), dtype=float)
        for index in range(physical.size):
            step = max(
                FINITE_DIFF_ABS_STEP,
                FINITE_DIFF_REL_STEP * self.widths[index],
            )
            plus = physical.copy()
            minus = physical.copy()
            plus[index] = min(self.upper[index], physical[index] + step)
            minus[index] = max(self.lower[index], physical[index] - step)
            jacobian[:, index] = (self.residuals(plus) - self.residuals(minus)) / (
                plus[index] - minus[index]
            )
        return jacobian


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--start-summary", type=Path, default=START_SUMMARY)
    parser.add_argument("--ftol", type=float, default=1.0e-10)
    parser.add_argument("--xtol", type=float, default=1.0e-10)
    parser.add_argument("--gtol", type=float, default=1.0e-8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--setup-only", action="store_true")
    args = parser.parse_args()

    fit13.patch_archived_context_paths()
    data = fit13.sample13.load_and_prepare_data(10.0, 2)
    data = fit13.sample13.apply_reflectivity_window(
        data,
        fit13.sample13.RC_START_DEG,
        fit13.sample13.RC_STOP_DEG,
    )
    start_values = load_start_values(args.start_summary)
    reflectivity_weight, diagnostics = resolve_reflectivity_weight(data, start_values)
    numpy_problem = make_problem(data, reflectivity_weight)
    jax_problem = fit13.make_jax_problem(numpy_problem)
    model = ExactResidualModel(jax_problem)
    initial = interior_vector(start_values)
    initial_values = values_from_vector(initial)
    initial_residuals = model.residuals(initial)
    initial_residual_objective = float(initial_residuals @ initial_residuals)
    initial_jax = jax_problem.evaluate(initial_values)
    initial_numpy = numpy_problem.evaluate(initial_values)

    print("Sample#13 bounded TRF least-squares setup")
    print(f"Output directory: {args.output_dir}")
    print(f"Residual count: {model.residual_count}")
    print(f"Parameter count: {len(PARAMETERS)}")
    print(f"Reflectivity weight: {reflectivity_weight:.8g}")
    print(f"Initial residual objective: {initial_residual_objective:.12g}")
    print(f"Initial JAX objective: {initial_jax.objective:.12g}")
    print(f"Initial NumPy objective: {initial_numpy.objective:.12g}")
    print("Datasets: reflectivity, C 1s, Ni 3p, La 4d")
    print("LNO-1 roughness: fixed 0 A")
    print("LNO-2 roughness: fitted fraction times LNO-1 thickness")
    if args.setup_only:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    residual_function = JaxResidualFunction(
        residuals_jax=model.residuals,
        jacobian_jax=model.jacobian,
        residual_count=model.residual_count,
    )
    start = perf_counter()
    result = optimize_with_jax_least_squares(
        PARAMETERS,
        residual_function,
        initial=initial,
        settings=JaxLeastSquaresOptimizerSettings(
            max_nfev=args.max_nfev,
            ftol=args.ftol,
            xtol=args.xtol,
            gtol=args.gtol,
        ),
    )
    wall_seconds = perf_counter() - start
    best_jax = jax_problem.evaluate(result.best_parameters)
    best_numpy = numpy_problem.evaluate(result.best_parameters)
    simulation = numpy_problem.simulate(result.best_parameters)

    save_outputs(
        args.output_dir,
        numpy_problem,
        simulation,
        result,
        initial_residual_objective,
        initial_jax,
        initial_numpy,
        best_jax,
        best_numpy,
        reflectivity_weight,
        diagnostics,
        wall_seconds,
    )
    print_result(result, best_jax, best_numpy, wall_seconds)


def make_problem(data, reflectivity_weight):
    base = fit13.make_problem(data, reflectivity_weight)
    la_data = RockingCurveData(
        "La 4d",
        data.rc_angle,
        data.rc_normalized["La 4d"],
        weight=3.0,
    )
    rc_problem = replace(
        base.rc_problem,
        parameters=PARAMETERS,
        rocking_curves=(*base.rc_problem.rocking_curves, la_data),
        core_levels=(*base.rc_problem.core_levels, la_core_level_request()),
    )
    return fit13.JointCap3Problem(
        parameters=PARAMETERS,
        reflectivity_problem=replace(base.reflectivity_problem, parameters=PARAMETERS),
        rc_problem=rc_problem,
    )


def la_core_level_request() -> CoreLevelRequest:
    kinetic_energy = (
        fit13.sample13.PHOTON_ENERGY_EV
        - fit13.sample13.BINDING_ENERGIES["La 4d"]
    )
    imfp_files = {
        "C": REPO_ROOT / "data" / "IMFP" / "C.ANG",
        "LNO": REPO_ROOT / "data" / "IMFP" / "LNO.ANG",
        "STO": REPO_ROOT / "data" / "IMFP" / "STO.ANG",
    }
    imfp = {
        material: imfp_from_file(path, kinetic_energy)
        for material, path in imfp_files.items()
    }
    return CoreLevelRequest(
        name="La 4d",
        binding_energy_ev=fit13.sample13.BINDING_ENERGIES["La 4d"],
        concentration_by_material={"LNO": 1.0},
        imfp_by_material={"vacuum": imfp["C"], **imfp},
        emitting_layer_indices=(2, 3),
    )


def resolve_reflectivity_weight(data, start_values):
    diagnostic = make_problem(data, reflectivity_weight=1.0).evaluate(start_values)
    raw = {item.name: item.raw for item in diagnostic.contributions}
    weighted = {item.name: item.weighted for item in diagnostic.contributions}
    rc_weighted = sum(weighted[name] for name in ("C 1s", "Ni 3p", "La 4d"))
    reflectivity_raw = raw["reflectivity"]
    weight = 1.0 if reflectivity_raw <= 0 else rc_weighted / reflectivity_raw
    return weight, {
        "reflectivity_raw": reflectivity_raw,
        "rc_weighted": rc_weighted,
    }

def load_start_values(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    return {parameter.name: float(row[parameter.name]) for parameter in PARAMETERS}


def interior_vector(values: dict[str, float], margin: float = 1.0e-4) -> np.ndarray:
    lower = np.asarray([parameter.lower for parameter in PARAMETERS], dtype=float)
    upper = np.asarray([parameter.upper for parameter in PARAMETERS], dtype=float)
    vector = np.asarray([values[parameter.name] for parameter in PARAMETERS])
    scaled = (vector - lower) / (upper - lower)
    return lower + np.clip(scaled, margin, 1.0 - margin) * (upper - lower)


def values_from_vector(vector: np.ndarray) -> dict[str, float]:
    return {
        parameter.name: float(value)
        for parameter, value in zip(PARAMETERS, vector)
    }


def save_outputs(
    output_dir,
    problem,
    simulation,
    result,
    initial_residual_objective,
    initial_jax,
    initial_numpy,
    best_jax,
    best_numpy,
    reflectivity_weight,
    diagnostics,
    wall_seconds,
) -> None:
    save_history(output_dir / "history.csv", result)
    plot_history(output_dir / "convergence.png", result)
    save_contributions(output_dir / "jax_contributions.csv", best_jax)
    save_contributions(output_dir / "numpy_validation_contributions.csv", best_numpy)
    save_summary(
        output_dir / "summary.csv",
        output_dir / "summary.txt",
        result,
        initial_residual_objective,
        initial_jax,
        initial_numpy,
        best_jax,
        best_numpy,
        reflectivity_weight,
        diagnostics,
        wall_seconds,
    )
    save_covariance(output_dir / "covariance.csv", result)
    save_uncertainties(output_dir / "parameter_uncertainties.csv", result)
    save_parameter_positions(output_dir / "parameter_positions.csv", result)
    plot_parameter_positions(output_dir / "parameter_positions.png", result)
    plot_best_fit(
        output_dir / "best_fit.png",
        problem.reflectivity,
        problem.rocking_curves,
        simulation,
    )
    plot_stack_schematic(
        output_dir / "stack_schematic.png",
        simulation.stack,
        title="Sample #13 TRF least-squares fit",
        top_layers=5,
        bottom_layers=3,
    )
    save_fit_curve_data_csv(
        output_dir / "best_fit_experiment_and_simulation.csv",
        problem.reflectivity,
        problem.rocking_curves,
        simulation,
    )
    save_optimized_stack_csv(
        output_dir / "optimized_stack_layers.csv",
        simulation.stack,
    )



def parameter_position_rows(result) -> list[dict]:
    rows = []
    for parameter in PARAMETERS:
        value = result.best_parameters[parameter.name]
        position = (value - parameter.lower) / (parameter.upper - parameter.lower)
        near_lower = position <= 0.01
        near_upper = position >= 0.99
        rows.append(
            {
                "parameter": parameter.name,
                "value": value,
                "lower_bound": parameter.lower,
                "upper_bound": parameter.upper,
                "normalized_position": position,
                "distance_to_nearest_bound_fraction": min(position, 1.0 - position),
                "near_lower_bound_1pct": near_lower,
                "near_upper_bound_1pct": near_upper,
            }
        )
    return rows


def save_parameter_positions(path: Path, result) -> None:
    write_rows(path, parameter_position_rows(result))


def plot_parameter_positions(path: Path, result) -> None:
    rows = parameter_position_rows(result)
    positions = np.asarray([row["normalized_position"] for row in rows])
    names = [row["parameter"] for row in rows]
    near_bound = np.asarray([
        row["near_lower_bound_1pct"] or row["near_upper_bound_1pct"]
        for row in rows
    ])
    y = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(10.0, 0.46 * len(rows) + 1.8))
    axis.axvspan(0.0, 0.01, color="#d9a39a", alpha=0.35, label="Within 1% of bound")
    axis.axvspan(0.99, 1.0, color="#d9a39a", alpha=0.35)
    axis.hlines(y, 0.0, 1.0, color="#c7cbcc", linewidth=2.0, zorder=1)
    axis.scatter(
        positions[~near_bound],
        y[~near_bound],
        s=48,
        color="#5f86a6",
        edgecolor="white",
        linewidth=0.7,
        label="Interior",
        zorder=3,
    )
    axis.scatter(
        positions[near_bound],
        y[near_bound],
        s=58,
        color="#b85f55",
        marker="D",
        edgecolor="white",
        linewidth=0.7,
        label="Near bound",
        zorder=4,
    )
    axis.set_yticks(y, labels=names)
    axis.invert_yaxis()
    axis.set_xlim(-0.02, 1.02)
    axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0], labels=["Lower", "25%", "50%", "75%", "Upper"])
    axis.set_xlabel("Position within fitted range")
    axis.grid(True, axis="x", alpha=0.22)
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_best_fit(path, reflectivity_data, rocking_curve_data, simulation) -> None:
    colors = {
        "reflectivity": "#7b6f93",
        "C 1s": "#789f8a",
        "Ni 3p": "#5f86a6",
        "La 4d": "#c47a6c",
    }
    fig, axes = plt.subplots(
        1 + len(rocking_curve_data),
        1,
        figsize=(8.0, 2.15 * (1 + len(rocking_curve_data))),
        sharex=False,
    )
    axes[0].semilogy(reflectivity_data.angles, reflectivity_data.reflectivity, "o", color=colors["reflectivity"], markersize=3.5, alpha=0.65, label="Experiment")
    axes[0].semilogy(simulation.reflectivity.angle, simulation.reflectivity.reflectivity, color="#303436", linewidth=1.8, label="TRF fit")
    axes[0].set_ylabel("Reflectivity")
    axes[0].legend(frameon=False)
    simulated_rc = {core.name: core.curve.intensity for core in simulation.rocking_curves.core_levels}
    for axis, data in zip(axes[1:], rocking_curve_data):
        axis.plot(data.angles, data.intensity, "o", color=colors[data.name], markersize=3.5, alpha=0.65, label="Experiment")
        axis.plot(simulation.rocking_curves.angle, simulated_rc[data.name], color="#303436", linewidth=1.8, label="TRF fit")
        axis.axhline(1.0, color="#666b6d", linestyle=":", linewidth=1.0)
        axis.set_ylabel(data.name)
        axis.legend(frameon=False)
    for axis in axes:
        axis.grid(True, which="both", alpha=0.22)
    axes[-1].set_xlabel("Grazing incidence angle (deg)")
    fig.suptitle("Sample #13 bounded least-squares fit", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def save_history(path: Path, result) -> None:
    rows = [
        {
            "iteration": record.iteration,
            "objective": 2.0 * record.cost,
            "gradient_norm": record.gradient_norm,
            **record.parameters,
        }
        for record in result.history
    ]
    write_rows(path, rows)


def plot_history(path: Path, result) -> None:
    if not result.history:
        return
    iterations = [record.iteration for record in result.history]
    objectives = [2.0 * record.cost for record in result.history]
    gradients = [record.gradient_norm for record in result.history]
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.0), sharex=True)
    axes[0].semilogy(iterations, objectives, marker="o", markersize=3, color="#5f86a6")
    axes[0].set_ylabel("Objective")
    axes[1].semilogy(iterations, gradients, marker="o", markersize=3, color="#c47a6c")
    axes[1].set_ylabel("Gradient norm")
    axes[1].set_xlabel("Jacobian evaluation")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_contributions(path: Path, evaluation) -> None:
    write_rows(
        path,
        [
            {
                "name": item.name,
                "raw": item.raw,
                "weight": item.weight,
                "weighted": item.weighted,
            }
            for item in evaluation.contributions
        ],
    )


def save_summary(
    csv_path,
    text_path,
    result,
    initial_residual_objective,
    initial_jax,
    initial_numpy,
    best_jax,
    best_numpy,
    reflectivity_weight,
    diagnostics,
    wall_seconds,
) -> None:
    row = {
        "initial_residual_objective": initial_residual_objective,
        "initial_jax_objective": initial_jax.objective,
        "initial_numpy_objective": initial_numpy.objective,
        "optimizer_objective": 2.0 * result.final_cost,
        "best_jax_objective": best_jax.objective,
        "best_numpy_objective": best_numpy.objective,
        "success": result.success,
        "status": result.status,
        "message": result.message,
        "nfev": result.nfev,
        "njev": result.njev,
        "optimality": result.optimality,
        "wall_seconds": wall_seconds,
        "reflectivity_weight": reflectivity_weight,
        "carbon_roughness": fit13.carbon_roughness_from_values(result.best_parameters),
        "lno1_roughness": fit13.lno1_roughness_from_values(result.best_parameters),
        "lno2_roughness": fit13.lno2_roughness_from_values(result.best_parameters),
        "top_lno_layer2_thickness": (
            result.best_parameters["top_lno_total_thickness"]
            - result.best_parameters["top_lno_layer1_thickness"]
        ),
        **result.best_parameters,
    }
    for contribution in best_numpy.contributions:
        row[f"{contribution.name}_raw"] = contribution.raw
        row[f"{contribution.name}_weighted"] = contribution.weighted
    for name, value in diagnostics.items():
        row[f"weight_diagnostic_{name}"] = value
    write_rows(csv_path, [row])
    with text_path.open("w", encoding="utf-8") as handle:
        for key, value in row.items():
            handle.write(f"{key}: {value}\n")


def save_covariance(path: Path, result) -> None:
    if result.covariance is None:
        return
    names = [parameter.name for parameter in PARAMETERS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", *names])
        for name, row in zip(names, result.covariance):
            writer.writerow([name, *row])


def save_uncertainties(path: Path, result) -> None:
    covariance = result.covariance
    standard_errors = (
        np.full(len(PARAMETERS), np.nan)
        if covariance is None
        else np.sqrt(np.maximum(np.diag(covariance), 0.0))
    )
    rows = []
    for parameter, standard_error in zip(PARAMETERS, standard_errors):
        value = result.best_parameters[parameter.name]
        rows.append(
            {
                "parameter": parameter.name,
                "value": value,
                "standard_error": float(standard_error),
                "lower_bound": parameter.lower,
                "upper_bound": parameter.upper,
                "at_lower_bound": np.isclose(value, parameter.lower, atol=1.0e-6),
                "at_upper_bound": np.isclose(value, parameter.upper, atol=1.0e-6),
            }
        )
    write_rows(path, rows)


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_result(result, best_jax, best_numpy, wall_seconds) -> None:
    print("TRF least-squares result")
    print(f"  success: {result.success}")
    print(f"  status: {result.status} ({result.message})")
    print(f"  nfev: {result.nfev}")
    print(f"  njev: {result.njev}")
    print(f"  optimality: {result.optimality:.6g}")
    print(f"  wall time: {wall_seconds:.3f} s")
    print(f"  optimizer objective: {2.0 * result.final_cost:.12g}")
    print(f"  best JAX objective: {best_jax.objective:.12g}")
    print(f"  best NumPy objective: {best_numpy.objective:.12g}")
    print("  derived cap values:")
    print(
        f"    carbon roughness: "
        f"{fit13.carbon_roughness_from_values(result.best_parameters):.6g} A"
    )
    print("    LNO-1 roughness: 0 A")
    print(
        f"    LNO-2 roughness: "
        f"{fit13.lno2_roughness_from_values(result.best_parameters):.6g} A"
    )


if __name__ == "__main__":
    main()
