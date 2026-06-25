"""Refine Sample#12 joint cap3 fitting with the JAX backend and L-BFGS-B.

This script reuses the archived ``fit_sample12_joint_cap3_bo.py`` setup for
data loading, the cap3 stack model, dataset weights, and plotting. The optimizer
path is replaced by ``optimize_with_jax_gradient`` using the high-level JAX
simulation backend and finite-difference gradients around that exact objective.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import sys

import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent
CASE_DIR = OUTPUT_DIR.parent
ARCHIVE_DIR = CASE_DIR / "support" / "legacy_bo"
REPO_ROOT = CASE_DIR.parents[1]
RUNS_DIR = REPO_ROOT / "runs" / "sample_12" / "jax_gradient"
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ARCHIVE_DIR) not in sys.path:
    sys.path.insert(0, str(ARCHIVE_DIR))

import fit_sample12_joint_cap3_bo as joint  # noqa: E402
from swanx.diagnostics import plot_stack_schematic
from swanx.fitting import (
    JaxGradientOptimizerSettings,
    optimize_with_jax_gradient,
)


DEFAULT_OUTPUT_PREFIX = "sample12_joint_cap3_jax_gradient"
DEFAULT_RUN_DIR = "attempts_run"
FINITE_DIFF_REL_STEP = 2.0e-4
FINITE_DIFF_ABS_STEP = 1.0e-4
STAGED_BEST_PATH = REPO_ROOT / "runs" / "sample_12" / "staged_multistart" / "final_best_parameters.csv"


def main() -> None:
    args = parse_args()
    patch_archived_context_paths()
    data = joint.sample12.load_and_prepare_data(
        args.background_percent,
        args.background_order,
    )
    data = joint.sample12.apply_reflectivity_window(
        data,
        args.reflectivity_min_angle,
        args.reflectivity_max_angle,
    )
    reflectivity_weight, diagnostics = joint.resolve_reflectivity_weight(
        data,
        args.reflectivity_weight,
    )
    numpy_problem = joint.make_problem(data, reflectivity_weight)
    jax_problem = make_jax_problem(numpy_problem)
    base_initial_values = starting_values(args.start)
    initial_vector = vector_from_values(base_initial_values)
    attempt_vectors = make_attempt_vectors(
        initial_vector,
        attempts=args.attempts,
        random_seed=args.random_seed,
        perturb_fraction=args.perturb_fraction,
    )
    initial_values = values_from_vector(attempt_vectors[0])
    initial_numpy = numpy_problem.evaluate(initial_values)
    initial_jax = jax_problem.evaluate(initial_values)

    print("Sample#12 joint cap3 JAX-gradient setup")
    run_dir = RUNS_DIR / args.output_dir
    print(f"Output directory: {run_dir}")
    print(f"Start: {args.start}")
    print(f"Attempts: {len(attempt_vectors)}")
    print(f"Reflectivity weight: {reflectivity_weight:.6g}")
    if diagnostics:
        print(f"Initial reflectivity raw log-MSE diagnostic: {diagnostics['reflectivity_raw']:.6g}")
        print(f"Initial weighted RC diagnostic sum: {diagnostics['rc_weighted']:.6g}")
    print(f"Initial NumPy objective: {initial_numpy.objective:.6g}")
    print(f"Initial JAX objective: {initial_jax.objective:.6g}")
    print(f"Max L-BFGS-B iterations: {args.maxiter}")

    if not args.run_fit:
        print("Gradient fitting was not run. Re-run with --run-fit after checking this setup.")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for attempt_index, attempt_vector in enumerate(attempt_vectors, start=1):
        attempt_dir = run_dir / f"attempt_{attempt_index:02d}"
        attempt_prefix = f"{args.output_prefix}_attempt_{attempt_index:02d}"
        attempt_initial_values = values_from_vector(attempt_vector)
        attempt_initial_numpy = numpy_problem.evaluate(attempt_initial_values)
        attempt_initial_jax = jax_problem.evaluate(attempt_initial_values)
        print(
            f"Attempt {attempt_index}/{len(attempt_vectors)}: "
            f"initial JAX objective={attempt_initial_jax.objective:.6g}",
            flush=True,
        )
        value_and_grad = make_finite_difference_value_and_grad(
            jax_problem,
            rel_step=args.finite_diff_rel_step,
            abs_step=args.finite_diff_abs_step,
        )
        start = perf_counter()
        result = optimize_with_jax_gradient(
            joint.PARAMETERS,
            value_and_grad,
            initial=attempt_vector,
            settings=JaxGradientOptimizerSettings(
                maxiter=args.maxiter,
                ftol=args.ftol,
                gtol=args.gtol,
            ),
        )
        elapsed = perf_counter() - start
        best_numpy = numpy_problem.evaluate(result.best_parameters)
        best_jax = jax_problem.evaluate(result.best_parameters)
        save_outputs(
            attempt_dir,
            attempt_prefix,
            numpy_problem,
            result,
            best_numpy,
            best_jax,
            attempt_initial_numpy,
            attempt_initial_jax,
            elapsed,
        )
        records.append(
            {
                "attempt": attempt_index,
                "folder": str(attempt_dir),
                "initial_jax_objective": attempt_initial_jax.objective,
                "best_jax_objective": best_jax.objective,
                "best_numpy_objective": best_numpy.objective,
                "success": result.success,
                "nit": result.nit,
                "nfev": result.nfev,
                "total_seconds": elapsed,
                "result": result,
                "best_numpy": best_numpy,
                "best_jax": best_jax,
                "initial_numpy": attempt_initial_numpy,
                "initial_jax": attempt_initial_jax,
            }
        )
        print_result(result, best_numpy, best_jax, elapsed)

    best_record = min(records, key=lambda record: float(record["best_jax_objective"]))
    write_attempt_summary(run_dir / f"{args.output_prefix}_attempt_summary.csv", records)
    save_outputs(
        run_dir,
        f"{args.output_prefix}_best",
        numpy_problem,
        best_record["result"],
        best_record["best_numpy"],
        best_record["best_jax"],
        best_record["initial_numpy"],
        best_record["initial_jax"],
        float(best_record["total_seconds"]),
    )
    print("Best attempt summary")
    print(f"  attempt: {best_record['attempt']}")
    print(f"  best JAX objective: {float(best_record['best_jax_objective']):.6g}")
    print(f"  best NumPy validation objective: {float(best_record['best_numpy_objective']):.6g}")


def patch_archived_context_paths() -> None:
    """Point the archived BO modules back to the Sample#12 data directory."""

    joint.CASE_DIR = CASE_DIR
    joint.REPO_ROOT = REPO_ROOT
    joint.sample12.CASE_DIR = CASE_DIR
    joint.sample12.REPO_ROOT = REPO_ROOT
    joint.sample12.REFLECTIVITY_FILE = CASE_DIR / "Reflectivity_Exp.dat"
    joint.sample12.RC_FILE = CASE_DIR / "ExpRCs.dat"
    joint.sample12.REFLECTIVITY_BO_HISTORY = CASE_DIR / "sample12_reflectivity_bo_history.csv"


def make_jax_problem(problem: joint.JointCap3Problem) -> joint.JointCap3Problem:
    """Return the same joint problem with both subproblems using JAX simulation."""

    return joint.JointCap3Problem(
        parameters=problem.parameters,
        reflectivity_problem=replace(problem.reflectivity_problem, simulation_backend="jax"),
        rc_problem=replace(problem.rc_problem, simulation_backend="jax"),
    )


def starting_values(mode: str) -> dict[str, float]:
    """Return initial values for the requested gradient start."""

    values = joint.initial_values()
    if mode == "default":
        return values
    staged = load_staged_best()
    if staged:
        values.update(staged)
    return values


def load_staged_best() -> dict[str, float]:
    """Load the best staged BO parameter CSV if available."""

    if not STAGED_BEST_PATH.exists():
        return {}
    values = {}
    with STAGED_BEST_PATH.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row.get("parameter", "")
            if name in joint.PARAMETER_BY_NAME:
                values[name] = float(row["value"])
    return values


def vector_from_values(values: dict[str, float]) -> np.ndarray:
    return np.asarray([values[parameter.name] for parameter in joint.PARAMETERS], dtype=float)


def values_from_vector(vector: np.ndarray) -> dict[str, float]:
    return {
        parameter.name: float(value)
        for parameter, value in zip(joint.PARAMETERS, vector)
    }


def make_attempt_vectors(
    base_vector: np.ndarray,
    attempts: int,
    random_seed: int,
    perturb_fraction: float,
) -> list[np.ndarray]:
    """Return one base start followed by clipped random perturbations."""

    if attempts <= 0:
        raise ValueError("attempts must be positive")
    if perturb_fraction < 0:
        raise ValueError("perturb_fraction must be non-negative")
    lower = np.asarray([parameter.lower for parameter in joint.PARAMETERS], dtype=float)
    upper = np.asarray([parameter.upper for parameter in joint.PARAMETERS], dtype=float)
    widths = upper - lower
    rng = np.random.default_rng(random_seed)
    vectors = [np.clip(np.asarray(base_vector, dtype=float), lower, upper)]
    for _ in range(1, attempts):
        perturbation = rng.normal(loc=0.0, scale=perturb_fraction, size=base_vector.shape)
        vectors.append(np.clip(base_vector + perturbation * widths, lower, upper))
    return vectors


def make_finite_difference_value_and_grad(problem, rel_step: float, abs_step: float):
    lower = np.asarray([parameter.lower for parameter in joint.PARAMETERS], dtype=float)
    upper = np.asarray([parameter.upper for parameter in joint.PARAMETERS], dtype=float)
    widths = upper - lower
    cache: dict[tuple[float, ...], float] = {}

    def objective(vector: np.ndarray) -> float:
        physical = np.clip(np.asarray(vector, dtype=float), lower, upper)
        key = tuple(np.round(physical, 12))
        if key not in cache:
            values = {
                parameter.name: float(value)
                for parameter, value in zip(joint.PARAMETERS, physical)
            }
            cache[key] = float(problem.evaluate(values).objective)
        return cache[key]

    def callback(vector: np.ndarray) -> tuple[float, np.ndarray]:
        physical = np.clip(np.asarray(vector, dtype=float), lower, upper)
        value = objective(physical)
        gradient = np.empty_like(physical)
        for index in range(physical.size):
            step = max(abs_step, rel_step * widths[index])
            plus = physical.copy()
            minus = physical.copy()
            plus[index] = min(upper[index], physical[index] + step)
            minus[index] = max(lower[index], physical[index] - step)
            if plus[index] == minus[index]:
                gradient[index] = 0.0
            elif plus[index] == physical[index]:
                gradient[index] = (objective(physical) - objective(minus)) / (
                    physical[index] - minus[index]
                )
            elif minus[index] == physical[index]:
                gradient[index] = (objective(plus) - objective(physical)) / (
                    plus[index] - physical[index]
                )
            else:
                gradient[index] = (objective(plus) - objective(minus)) / (
                    plus[index] - minus[index]
                )
        return value, gradient

    return callback


def save_outputs(
    output_dir: Path,
    output_prefix: str,
    problem: joint.JointCap3Problem,
    result,
    best_numpy,
    best_jax,
    initial_numpy,
    initial_jax,
    elapsed: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_gradient_history(output_dir / f"{output_prefix}_history.csv", result)
    plot_gradient_history(
        output_dir / f"{output_prefix}_convergence.png",
        output_dir / f"{output_prefix}_history.csv",
    )
    save_contributions(output_dir / f"{output_prefix}_validation_numpy_contributions.csv", best_numpy)
    save_contributions(output_dir / f"{output_prefix}_best_jax_contributions.csv", best_jax)
    save_summary(
        output_dir / f"{output_prefix}_summary.csv",
        result,
        best_numpy,
        best_jax,
        initial_numpy,
        initial_jax,
        elapsed,
    )
    simulation = problem.simulate(result.best_parameters)
    joint.plot_joint_best_fit(
        output_dir / f"{output_prefix}_best_fit.png",
        problem.reflectivity,
        problem.rocking_curves,
        simulation,
    )
    plot_stack_schematic(
        output_dir / f"{output_prefix}_stack_schematic.png",
        simulation.stack,
        title="Sample#12 Joint Cap3 JAX-Gradient Stack",
        top_layers=6,
        bottom_layers=3,
    )
    joint.sample12.save_superlattice_profile_plot(
        result.best_parameters,
        output_dir / f"{output_prefix}_superlattice_profile.png",
    )


def save_gradient_history(path: Path, result) -> None:
    columns = [
        "iteration",
        "loss",
        "gradient_norm",
        *[parameter.name for parameter in joint.PARAMETERS],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in result.history:
            row = {
                "iteration": record.iteration,
                "loss": record.loss,
                "gradient_norm": record.gradient_norm,
            }
            row.update(record.parameters)
            writer.writerow(row)


def save_contributions(path: Path, evaluation) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "raw", "weight", "weighted"])
        for contribution in evaluation.contributions:
            writer.writerow([
                contribution.name,
                contribution.raw,
                contribution.weight,
                contribution.weighted,
            ])


def save_summary(
    path: Path,
    result,
    best_numpy,
    best_jax,
    initial_numpy,
    initial_jax,
    elapsed: float,
) -> None:
    row = {
        "initial_numpy_objective": initial_numpy.objective,
        "initial_jax_objective": initial_jax.objective,
        "best_numpy_objective": best_numpy.objective,
        "best_jax_objective": best_jax.objective,
        "optimizer_loss": result.best_loss,
        "success": result.success,
        "status": result.status,
        "message": result.message,
        "nit": result.nit,
        "nfev": result.nfev,
        "total_seconds": elapsed,
        "carbon_roughness": joint.carbon_roughness_from_values(result.best_parameters),
        "cap_roughness": joint.cap_roughness_from_values(result.best_parameters),
        "top_lno_layer2_thickness": (
            result.best_parameters["top_lno_total_thickness"]
            - result.best_parameters["top_lno_layer1_thickness"]
        ),
    }
    row.update(result.best_parameters)
    for contribution in best_numpy.contributions:
        row[f"{contribution.name}_raw"] = contribution.raw
        row[f"{contribution.name}_weighted"] = contribution.weighted
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with path.with_suffix(".txt").open("w", encoding="utf-8") as handle:
        for key, value in row.items():
            handle.write(f"{key}: {value}\n")


def plot_gradient_history(path: Path, history_path: Path) -> None:
    plt = joint.sample12._load_pyplot()
    with history_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    iterations = np.asarray([float(row["iteration"]) for row in rows])
    losses = np.asarray([float(row["loss"]) for row in rows])
    gradient_norms = np.asarray([float(row["gradient_norm"]) for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True)
    axes[0].semilogy(iterations, losses, marker="o", markersize=3)
    axes[0].set_ylabel("Objective")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[1].semilogy(iterations, gradient_norms, marker="o", markersize=3, color="tab:orange")
    axes[1].set_ylabel("Gradient norm")
    axes[1].set_xlabel("Iteration")
    axes[1].grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_attempt_summary(path: Path, records: list[dict[str, object]]) -> None:
    columns = [
        "attempt",
        "folder",
        "initial_jax_objective",
        "best_jax_objective",
        "best_numpy_objective",
        "success",
        "nit",
        "nfev",
        "total_seconds",
        *[parameter.name for parameter in joint.PARAMETERS],
        "carbon_roughness",
        "cap_roughness",
        "top_lno_layer2_thickness",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            result = record["result"]
            row = {
                key: record[key]
                for key in (
                    "attempt",
                    "folder",
                    "initial_jax_objective",
                    "best_jax_objective",
                    "best_numpy_objective",
                    "success",
                    "nit",
                    "nfev",
                    "total_seconds",
                )
            }
            row.update(result.best_parameters)
            row["carbon_roughness"] = joint.carbon_roughness_from_values(result.best_parameters)
            row["cap_roughness"] = joint.cap_roughness_from_values(result.best_parameters)
            row["top_lno_layer2_thickness"] = (
                result.best_parameters["top_lno_total_thickness"]
                - result.best_parameters["top_lno_layer1_thickness"]
            )
            writer.writerow(row)


def print_result(result, best_numpy, best_jax, elapsed: float) -> None:
    print("Gradient fitting result")
    print(f"  success: {result.success}")
    print(f"  status: {result.status} ({result.message})")
    print(f"  iterations: {result.nit}")
    print(f"  function evaluations: {result.nfev}")
    print(f"  wall time: {elapsed:.3f} s")
    print(f"  optimizer loss: {result.best_loss:.6g}")
    print(f"  best NumPy objective: {best_numpy.objective:.6g}")
    print(f"  best JAX objective: {best_jax.objective:.6g}")
    print("  best NumPy contributions:")
    for contribution in best_numpy.contributions:
        print(
            f"    {contribution.name}: raw={contribution.raw:.6g}, "
            f"weighted={contribution.weighted:.6g}"
        )
    print("  best parameters:")
    for parameter in joint.PARAMETERS:
        unit = f" {parameter.unit}" if parameter.unit else ""
        print(f"    {parameter.name}: {result.best_parameters[parameter.name]:.6g}{unit}")
    print("  derived parameters:")
    print(f"    carbon_roughness: {joint.carbon_roughness_from_values(result.best_parameters):.6g} A")
    print(f"    cap_roughness: {joint.cap_roughness_from_values(result.best_parameters):.6g} A")
    print(
        "    top_lno_layer2_thickness: "
        f"{result.best_parameters['top_lno_total_thickness'] - result.best_parameters['top_lno_layer1_thickness']:.6g} A"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background-percent", type=float, default=10.0)
    parser.add_argument("--background-order", type=int, default=2)
    parser.add_argument("--reflectivity-min-angle", type=float, default=joint.sample12.RC_START_DEG)
    parser.add_argument("--reflectivity-max-angle", type=float, default=joint.sample12.RC_STOP_DEG)
    parser.add_argument("--reflectivity-weight", default="0.0503187")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--output-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--start", choices=("staged-best", "default"), default="staged-best")
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=120)
    parser.add_argument("--perturb-fraction", type=float, default=0.04)
    parser.add_argument("--maxiter", type=int, default=40)
    parser.add_argument("--ftol", type=float, default=1.0e-12)
    parser.add_argument("--gtol", type=float, default=1.0e-7)
    parser.add_argument("--finite-diff-rel-step", type=float, default=FINITE_DIFF_REL_STEP)
    parser.add_argument("--finite-diff-abs-step", type=float, default=FINITE_DIFF_ABS_STEP)
    parser.add_argument("--run-fit", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
