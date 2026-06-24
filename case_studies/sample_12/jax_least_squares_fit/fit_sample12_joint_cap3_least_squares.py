"""Refit Sample#12 cap3 data with bounded TRF and promote an improvement."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import shutil
import sys

import numpy as np


FIT_DIR = Path(__file__).resolve().parent
CASE_DIR = FIT_DIR.parent
REPO_ROOT = CASE_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
GRADIENT_DIR = CASE_DIR / "jax_gradient_fit"
SAMPLE13_LS_DIR = CASE_DIR.parent / "sample_13" / "jax_least_squares_all_rcs"
BEST_DIR = CASE_DIR / "best_results_so_far"
RUNS_DIR = REPO_ROOT / "runs" / "sample_12" / "jax_least_squares"
DEFAULT_OUTPUT_DIR = RUNS_DIR / "edge_polynomial_normalization_run"

for path in (SRC_DIR, GRADIENT_DIR, SAMPLE13_LS_DIR, CASE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from swanx.jax_least_squares import (  # noqa: E402
    JaxLeastSquaresOptimizerSettings,
    JaxResidualFunction,
    optimize_with_jax_least_squares,
)
from swanx.stack.profiles import plot_vertical_concentration_profiles  # noqa: E402
from swanx.stack_visualization import plot_stack_schematic  # noqa: E402
from swanx.diagnostics import (  # noqa: E402
    diagnostics_from_least_squares_result,
    plot_correlation_matrix,
    plot_parameter_estimates,
)
from swanx.diagnostics.reports import (  # noqa: E402
    save_fit_curve_data_csv,
    save_optimized_stack_csv,
)

import collect_best_results_so_far as collector  # noqa: E402
import fit_sample12_joint_cap3_jax_gradient as fit12  # noqa: E402
import fit_sample13_reflectivity_all_rcs_least_squares as ls_utils  # noqa: E402


BASE_PARAMETERS = fit12.joint.PARAMETERS
PARAMETERS = BASE_PARAMETERS
ls_utils.PARAMETERS = PARAMETERS
GRADIENT_SUMMARY = BEST_DIR / "sample12_joint_cap3_jax_gradient_best_summary.csv"
TRF_SUMMARY = BEST_DIR / "sample12_trf_least_squares_summary.csv"


def main() -> None:
    args = parse_args()
    configure_parameters(args.carbon_min, args.carbon_max)
    fit12.patch_archived_context_paths()
    reference_path = TRF_SUMMARY if TRF_SUMMARY.exists() else GRADIENT_SUMMARY
    reference = read_row(reference_path)
    weight = reference_weight(reference)
    start_values = {item.name: float(reference[item.name]) for item in PARAMETERS}

    data = fit12.joint.sample12.load_and_prepare_data(10.0, 2)
    data = fit12.joint.sample12.apply_reflectivity_window(
        data, fit12.joint.sample12.RC_START_DEG, fit12.joint.sample12.RC_STOP_DEG
    )
    numpy_problem = make_problem(data, weight, args.rc_normalization)
    jax_problem = fit12.make_jax_problem(numpy_problem)
    model = ls_utils.ExactResidualModel(jax_problem)
    initial = interior_vector(start_values)
    initial_values = values_from_vector(initial)
    residuals = model.residuals(initial)
    initial_residual_objective = float(residuals @ residuals)
    initial_jax = jax_problem.evaluate(initial_values)
    initial_numpy = numpy_problem.evaluate(initial_values)

    print("Sample#12 bounded TRF least-squares setup")
    print(f"Reference: {reference_path}")
    print(f"Reflectivity weight: {weight:.12g}")
    print(f"Residual count / parameters: {model.residual_count} / {len(PARAMETERS)}")
    print(f"Initial residual objective: {initial_residual_objective:.12g}")
    print(f"Initial JAX objective: {initial_jax.objective:.12g}")
    print(f"Initial NumPy objective: {initial_numpy.objective:.12g}")
    if args.setup_only:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = optimize_with_jax_least_squares(
        PARAMETERS,
        JaxResidualFunction(model.residuals, model.jacobian, model.residual_count),
        initial=initial,
        settings=JaxLeastSquaresOptimizerSettings(
            max_nfev=args.max_nfev, ftol=args.ftol, xtol=args.xtol, gtol=args.gtol
        ),
    )
    best_jax = jax_problem.evaluate(result.best_parameters)
    best_numpy = numpy_problem.evaluate(result.best_parameters)
    simulation = numpy_problem.simulate(result.best_parameters)
    save_outputs(
        args.output_dir, numpy_problem, simulation, result,
        initial_residual_objective, initial_jax, initial_numpy,
        best_jax, best_numpy, weight, args.rc_normalization,
    )
    print_result(result, best_jax, best_numpy)
    if not args.skip_promotion:
        compare_and_promote(reference, weight, best_numpy, simulation, args.output_dir)
    else:
        print("Promotion skipped because this run uses a different RC normalization.")


def configure_parameters(carbon_min: float, carbon_max: float) -> None:
    global PARAMETERS
    if not 0.0 < carbon_min < carbon_max:
        raise ValueError("carbon bounds must satisfy 0 < minimum < maximum")
    PARAMETERS = tuple(
        replace(item, lower=carbon_min, upper=carbon_max)
        if item.name == "carbon_thickness"
        else item
        for item in BASE_PARAMETERS
    )
    ls_utils.PARAMETERS = PARAMETERS


def make_problem(data, weight, rc_normalization):
    base = fit12.joint.make_problem(data, weight)
    return fit12.joint.JointCap3Problem(
        parameters=PARAMETERS,
        reflectivity_problem=replace(base.reflectivity_problem, parameters=PARAMETERS),
        rc_problem=replace(
            base.rc_problem,
            parameters=PARAMETERS,
            rocking_curve_normalization=rc_normalization,
            normalization_edge_fraction=0.10,
            normalization_polynomial_order=2,
        ),
    )


def interior_vector(values: dict[str, float], margin: float = 1.0e-4) -> np.ndarray:
    lower = np.asarray([item.lower for item in PARAMETERS])
    upper = np.asarray([item.upper for item in PARAMETERS])
    vector = np.asarray([values[item.name] for item in PARAMETERS])
    scaled = (vector - lower) / (upper - lower)
    return lower + np.clip(scaled, margin, 1.0 - margin) * (upper - lower)


def values_from_vector(vector) -> dict[str, float]:
    return {item.name: float(value) for item, value in zip(PARAMETERS, vector)}


def reference_weight(row: dict[str, str]) -> float:
    if row.get("reflectivity_weight"):
        return float(row["reflectivity_weight"])
    return float(row["reflectivity_weighted"]) / float(row["reflectivity_raw"])


def common_objective(row: dict[str, str], weight: float) -> float:
    return (
        weight * float(row["reflectivity_raw"])
        + 0.5 * float(row["C 1s_raw"])
        + 3.0 * float(row["Ni 3p_raw"])
        + 3.0 * float(row["La 4d_raw"])
    )


def save_outputs(output_dir, problem, simulation, result, initial_residual_objective,
                 initial_jax, initial_numpy, best_jax, best_numpy, weight,
                 rc_normalization) -> None:
    ls_utils.save_history(output_dir / "history.csv", result)
    ls_utils.plot_history(output_dir / "convergence.png", result)
    ls_utils.save_contributions(output_dir / "jax_contributions.csv", best_jax)
    ls_utils.save_contributions(output_dir / "numpy_validation_contributions.csv", best_numpy)
    ls_utils.save_covariance(output_dir / "covariance.csv", result)
    ls_utils.save_uncertainties(output_dir / "parameter_uncertainties.csv", result)
    save_parameter_diagnostics(output_dir, result)
    save_summary(
        output_dir / "summary.csv", output_dir / "summary.txt", result,
        initial_residual_objective, initial_jax, initial_numpy, best_jax, best_numpy,
        weight, rc_normalization,
    )
    fit12.joint.plot_joint_best_fit(
        output_dir / "best_fit.png", problem.reflectivity, problem.rocking_curves, simulation
    )
    plot_stack_schematic(
        output_dir / "stack_schematic.png", simulation.stack,
        title="Sample#12 cap3 TRF least-squares stack", top_layers=6, bottom_layers=3,
    )
    fit12.joint.sample12.save_superlattice_profile_plot(
        result.best_parameters, output_dir / "superlattice_profile.png"
    )
    save_fit_curve_data_csv(
        output_dir / "best_fit_experiment_and_simulation.csv",
        problem.reflectivity, problem.rocking_curves, simulation,
    )
    save_optimized_stack_csv(output_dir / "optimized_stack_layers.csv", simulation.stack)


def save_parameter_diagnostics(output_dir: Path, result) -> None:
    """Save corrected public-API uncertainty and correlation diagnostics."""

    import matplotlib.pyplot as plt

    diagnostics = diagnostics_from_least_squares_result(result, PARAMETERS)
    uncertainty_figure, _ = plot_parameter_estimates(diagnostics)
    uncertainty_figure.savefig(
        output_dir / "parameter_uncertainty.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(uncertainty_figure)

    correlation_figure, _ = plot_correlation_matrix(diagnostics)
    correlation_figure.savefig(
        output_dir / "parameter_correlation.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(correlation_figure)

    with (output_dir / "parameter_correlation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("parameter", *diagnostics.names))
        for name, row in zip(diagnostics.names, diagnostics.correlation):
            writer.writerow((name, *row))
def save_summary(csv_path, text_path, result, initial_residual_objective,
                 initial_jax, initial_numpy, best_jax, best_numpy, weight,
                 rc_normalization) -> None:
    row = {
        "initial_residual_objective": initial_residual_objective,
        "initial_jax_objective": initial_jax.objective,
        "initial_numpy_objective": initial_numpy.objective,
        "optimizer_objective": 2.0 * result.final_cost,
        "best_jax_objective": best_jax.objective,
        "best_numpy_objective": best_numpy.objective,
        "success": result.success, "status": result.status, "message": result.message,
        "nfev": result.nfev, "njev": result.njev, "optimality": result.optimality,
        "wall_seconds": result.total_seconds, "reflectivity_weight": weight,
        "rocking_curve_normalization": rc_normalization,
        "normalization_edge_fraction": 0.10,
        "normalization_polynomial_order": 2,
        "carbon_thickness_lower_bound": next(
            item.lower for item in PARAMETERS if item.name == "carbon_thickness"
        ),
        "carbon_thickness_upper_bound": next(
            item.upper for item in PARAMETERS if item.name == "carbon_thickness"
        ),
        "carbon_roughness": fit12.joint.carbon_roughness_from_values(result.best_parameters),
        "cap_roughness": fit12.joint.cap_roughness_from_values(result.best_parameters),
        "top_lno_layer2_thickness": (
            result.best_parameters["top_lno_total_thickness"]
            - result.best_parameters["top_lno_layer1_thickness"]
        ),
        **result.best_parameters,
    }
    for item in best_numpy.contributions:
        row[f"{item.name}_raw"] = item.raw
        row[f"{item.name}_weighted"] = item.weighted
    ls_utils.write_rows(csv_path, [row])
    text_path.write_text("".join(f"{key}: {value}\n" for key, value in row.items()), encoding="utf-8")


def compare_and_promote(reference, weight, candidate, simulation, output_dir: Path) -> None:
    previous = common_objective(reference, weight)
    candidate_row = {f"{item.name}_raw": str(item.raw) for item in candidate.contributions}
    current = common_objective(candidate_row, weight)
    print(f"Previous common objective: {previous:.12g}")
    print(f"Candidate common objective: {current:.12g}")
    if current >= previous:
        print("Candidate was not promoted because it did not improve the saved best.")
        return

    clear_best_directory()
    artifacts = {
        "best_fit.png": "sample12_trf_least_squares_best_fit.png",
        "convergence.png": "sample12_trf_least_squares_convergence.png",
        "stack_schematic.png": "sample12_trf_least_squares_stack_schematic.png",
        "superlattice_profile.png": "sample12_trf_least_squares_superlattice_profile.png",
        "summary.csv": "sample12_trf_least_squares_summary.csv",
        "summary.txt": "sample12_trf_least_squares_summary.txt",
        "jax_contributions.csv": "sample12_trf_least_squares_jax_contributions.csv",
        "numpy_validation_contributions.csv": "sample12_trf_least_squares_validation_numpy_contributions.csv",
        "history.csv": "sample12_trf_least_squares_history.csv",
        "covariance.csv": "sample12_trf_least_squares_covariance.csv",
        "parameter_uncertainties.csv": "sample12_trf_least_squares_parameter_uncertainties.csv",
        "parameter_uncertainty.png": "sample12_trf_least_squares_parameter_uncertainty.png",
        "parameter_correlation.png": "sample12_trf_least_squares_parameter_correlation.png",
        "parameter_correlation.csv": "sample12_trf_least_squares_parameter_correlation.csv",
        "best_fit_experiment_and_simulation.csv": "best_fit_experiment_and_simulation.csv",
        "optimized_stack_layers.csv": "optimized_stack_layers.csv",
    }
    for source_name, target_name in artifacts.items():
        source = output_dir / source_name
        if source.exists():
            shutil.copy2(source, BEST_DIR / target_name)
    profiles = plot_vertical_concentration_profiles(
        BEST_DIR / "sample12_trf_top30_vertical_concentration_profiles.png",
        simulation.stack, collector.concentration_by_layer(simulation.stack),
        max_depth=30.0, step=0.2, title="Sample#12 top 30 A concentration profiles",
        layer_labels=collector.layer_labels(simulation.stack), show_layer_shading=False,
        layer_box_style=True, categorical_strips=True,
    )
    collector.save_concentration_profiles_csv(
        BEST_DIR / "sample12_trf_top30_vertical_concentration_profiles.csv",
        profiles.depth, profiles.profiles,
    )
    ls_utils.write_rows(BEST_DIR / "selection_comparison.csv", [{
        "reference_reflectivity_weight": weight,
        "previous_common_objective": previous,
        "promoted_common_objective": current,
        "improvement_percent": 100.0 * (previous - current) / previous,
    }])
    print(f"Promoted improved TRF result to {BEST_DIR}")


def clear_best_directory() -> None:
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    if BEST_DIR.resolve().parent != CASE_DIR.resolve():
        raise ValueError("best-results directory is outside the Sample#12 case folder")
    for child in BEST_DIR.iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()


def read_row(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return next(csv.DictReader(handle))


def print_result(result, best_jax, best_numpy) -> None:
    print("TRF least-squares result")
    print(f"  success: {result.success}")
    print(f"  status: {result.status} ({result.message})")
    print(f"  nfev/njev: {result.nfev}/{result.njev}")
    print(f"  wall time: {result.total_seconds:.3f} s")
    print(f"  best JAX objective: {best_jax.objective:.12g}")
    print(f"  best NumPy objective: {best_numpy.objective:.12g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-nfev", type=int, default=160)
    parser.add_argument("--carbon-min", type=float, default=2.0)
    parser.add_argument("--carbon-max", type=float, default=15.0)
    parser.add_argument(
        "--rc-normalization",
        choices=("mean", "edge_polynomial"),
        default="edge_polynomial",
    )
    parser.add_argument("--skip-promotion", action="store_true")
    parser.add_argument("--ftol", type=float, default=1.0e-12)
    parser.add_argument("--xtol", type=float, default=1.0e-12)
    parser.add_argument("--gtol", type=float, default=1.0e-9)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--setup-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
