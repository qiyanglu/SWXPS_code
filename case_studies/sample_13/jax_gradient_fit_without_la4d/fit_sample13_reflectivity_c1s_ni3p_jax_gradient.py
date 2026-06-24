"""Fit Sample#13 reflectivity, C 1s, and Ni 3p with constrained cap roughness.

This mirrors the Sample#12 cap3 gradient workflow while preserving Sample#13
data ranges and optical constants from the archived BO script. The cap stack is:

vacuum / C / LNO-1 / LNO-2 / LNO-bottom / [STO/LNO]x40 / STO substrate

LNO-1 is constrained to a very thin 1-20 A Ni-free layer. Its upper-interface
roughness is fixed to zero. The LNO-2 upper-interface roughness is fitted as a
fraction of LNO-1 thickness, so it cannot exceed that thickness. La 4d is
excluded from this experiment; Ni 3p is emitted from LNO-2 only.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
import sys

import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent
CASE_DIR = OUTPUT_DIR.parent
ARCHIVE_DIR = CASE_DIR / "support" / "legacy_bo"

REPO_ROOT = CASE_DIR.parents[1]
RUNS_ROOT = REPO_ROOT / "runs" / "sample_13"
BO_OUTPUT_DIR = RUNS_ROOT / "bo_outputs"
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ARCHIVE_DIR) not in sys.path:
    sys.path.insert(0, str(ARCHIVE_DIR))

import fit_sample13_bo as sample13  # noqa: E402
from swxps import (  # noqa: E402
    CoreLevelRequest,
    FitParameter,
    FitSimulation,
    FittingProblem,
    JaxGradientOptimizerSettings,
    JointObjective,
    LayerTemplate,
    ReflectivityData,
    RockingCurveData,
    StackTemplate,
    evaluation_from_contributions,
    imfp_from_file,
    optimize_with_jax_gradient,
    plot_stack_schematic,
)


DEFAULT_OUTPUT_PREFIX = "sample13_reflectivity_c1s_ni3p_jax_gradient"
DEFAULT_RUN_DIR = "setup_check"
FINITE_DIFF_REL_STEP = 2.0e-4
FINITE_DIFF_ABS_STEP = 1.0e-4
BOTTOM_LNO_THICKNESS = 160.0
DEFAULT_REFLECTIVITY_WEIGHT = "auto"

PARAMETERS = (
    FitParameter("carbon_thickness", 2.0, 15.0, "Angstrom", initial=10.3836753781),
    FitParameter("carbon_roughness_fraction", 0.0, 1.0, "", initial=0.4028468258),
    FitParameter("top_lno_total_thickness", 45.0, 65.0, "Angstrom", initial=48.7699239281),
    FitParameter("top_lno_layer1_thickness", 1.0, 20.0, "Angstrom", initial=1.2),
    FitParameter("lno2_roughness_fraction", 0.0, 1.0, "", initial=0.5169415307),
    FitParameter("sto_thickness_start", 13.0, 18.5, "Angstrom", initial=14.02),
    FitParameter("lno_thickness_start", 13.0, 18.5, "Angstrom", initial=18.45),
    FitParameter("sto_thickness_delta", 0.0, 3.0, "Angstrom", initial=0.02),
    FitParameter("lno_thickness_delta", 0.0, 3.0, "Angstrom", initial=0.02),
    FitParameter("thickness_transition_repeat", 0.0, 39.0, "repeat", initial=38.8),
    FitParameter("thickness_transition_width", 1.0, 20.0, "repeat", initial=16.8897758363),
    FitParameter("sto_roughness_first", 2.0, 5.0, "Angstrom", initial=4.2320725073),
    FitParameter("sto_roughness_last", 2.0, 5.0, "Angstrom", initial=3.2267809258),
    FitParameter("lno_roughness_first", 2.0, 5.0, "Angstrom", initial=4.2706376939),
    FitParameter("lno_roughness_last", 2.0, 5.0, "Angstrom", initial=3.2115828023),
    FitParameter("substrate_roughness", 1.0, 5.0, "Angstrom", initial=2.9707190434),
    FitParameter("reflectivity_angle_offset", -0.30, 0.30, "deg", initial=0.1379020343),
    FitParameter("rc_angle_offset", -0.30, 0.30, "deg", initial=0.1488918363),
)
PARAMETER_BY_NAME = {parameter.name: parameter for parameter in PARAMETERS}

DATASET_WEIGHTS = {
    "C 1s": 0.5,
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


def main() -> None:
    args = parse_args()
    patch_archived_context_paths()
    data = sample13.load_and_prepare_data(
        args.background_percent,
        args.background_order,
    )
    data = sample13.apply_reflectivity_window(
        data,
        args.reflectivity_min_angle,
        args.reflectivity_max_angle,
    )
    reflectivity_weight, diagnostics = resolve_reflectivity_weight(
        data,
        args.reflectivity_weight,
    )
    numpy_problem = make_problem(data, reflectivity_weight)
    jax_problem = make_jax_problem(numpy_problem)
    base_initial_values = initial_values()
    initial_vector = vector_from_values(base_initial_values)
    attempt_vectors = make_attempt_vectors(
        initial_vector,
        attempts=args.attempts,
        random_seed=args.random_seed,
        perturb_fraction=args.perturb_fraction,
    )
    initial_values_for_report = values_from_vector(attempt_vectors[0])
    initial_numpy = numpy_problem.evaluate(initial_values_for_report)
    initial_jax = jax_problem.evaluate(initial_values_for_report)

    print("Sample#13 reflectivity + C 1s + Ni 3p JAX-gradient setup")
    run_dir = RUNS_ROOT / "jax_gradient_without_la4d" / args.output_dir
    print(f"Output directory: {run_dir}")
    print(f"Attempts: {len(attempt_vectors)}")
    print(f"Reflectivity weight: {reflectivity_weight:.6g}")
    if diagnostics:
        print(f"Initial reflectivity raw log-MSE diagnostic: {diagnostics['reflectivity_raw']:.6g}")
        print(f"Initial weighted RC diagnostic sum: {diagnostics['rc_weighted']:.6g}")
    print(f"Initial NumPy objective: {initial_numpy.objective:.6g}")
    print(f"Initial JAX objective: {initial_jax.objective:.6g}")
    print(f"Max L-BFGS-B iterations: {args.maxiter}")
    print("Layer roughness: LNO-1 fixed at 0 A; fitted LNO-2 roughness <= LNO-1 thickness")
    print("Layer-selective RC emission: C 1s -> C; Ni 3p -> LNO-2; La 4d excluded")

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
            PARAMETERS,
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
    """Point the archived BO module back to the Sample#13 data directory."""

    sample13.CASE_DIR = CASE_DIR
    sample13.REPO_ROOT = REPO_ROOT
    sample13.REFLECTIVITY_FILE = CASE_DIR / "Reflectivity_Exp.dat"
    sample13.RC_FILE = CASE_DIR / "ExpRCs.dat"
    sample13.REFLECTIVITY_BO_HISTORY = BO_OUTPUT_DIR / "sample13_reflectivity_bo_history.csv"


def initial_values() -> dict[str, float]:
    return {
        parameter.name: float(parameter.initial)
        for parameter in PARAMETERS
        if parameter.initial is not None
    }


def carbon_roughness_from_values(values: dict[str, float]) -> float:
    carbon_thickness = values["carbon_thickness"]
    max_roughness = min(8.0, carbon_thickness)
    return 1.0 + values["carbon_roughness_fraction"] * (max_roughness - 1.0)


def lno1_roughness_from_values(values: dict[str, float]) -> float:
    """Return the fixed C/LNO-1 roughness."""

    return 0.0

def lno2_roughness_from_values(values: dict[str, float]) -> float:
    """Return LNO-1/LNO-2 roughness constrained by LNO-1 thickness."""

    return values["lno2_roughness_fraction"] * values["top_lno_layer1_thickness"]


def build_cap3_stack(values: dict[str, float]):
    """Build the Sample#13 three-layer LNO cap stack."""

    layer1_thickness = values["top_lno_layer1_thickness"]
    layer2_thickness = values["top_lno_total_thickness"] - layer1_thickness
    if layer2_thickness < 0.0:
        raise ValueError("top LNO layer 1 cannot exceed total top-LNO thickness")
    stack_values = {
        **values,
        "carbon_roughness": carbon_roughness_from_values(values),
        "lno1_roughness": lno1_roughness_from_values(values),
        "lno2_roughness": lno2_roughness_from_values(values),
        "top_lno_layer2_thickness": layer2_thickness,
    }
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
            "top_lno_layer1_thickness",
            "lno1_roughness",
        ),
        LayerTemplate.from_file(
            "LNO",
            sample13.LNO_OPC_FILE,
            layer2_thickness,
            "lno2_roughness",
        ),
        LayerTemplate.from_file(
            "LNO",
            sample13.LNO_OPC_FILE,
            BOTTOM_LNO_THICKNESS,
            0.0,
        ),
        *sample13.sample13_graded_superlattice_templates(stack_values),
        LayerTemplate.from_file(
            "STO",
            sample13.STO_OPC_FILE,
            0.0,
            "substrate_roughness",
        ),
    )
    return StackTemplate(
        energy_ev=sample13.PHOTON_ENERGY_EV,
        base_dir=REPO_ROOT,
        parts=parts,
    ).build(stack_values)


def core_level_requests() -> tuple[CoreLevelRequest, ...]:
    """Return layer-selective C and Ni requests for the Sample#13 cap3 stack."""

    imfp_files = {
        "C": REPO_ROOT / "IMFP" / "C.ANG",
        "LNO": REPO_ROOT / "IMFP" / "LNO.ANG",
        "STO": REPO_ROOT / "IMFP" / "STO.ANG",
    }
    imfp_by_core = {}
    for core_name in ("C 1s", "Ni 3p"):
        binding_energy = sample13.BINDING_ENERGIES[core_name]
        kinetic_energy = sample13.PHOTON_ENERGY_EV - binding_energy
        imfp_by_core[core_name] = {
            material: imfp_from_file(path, kinetic_energy)
            for material, path in imfp_files.items()
        }

    return (
        CoreLevelRequest(
            name="C 1s",
            binding_energy_ev=sample13.BINDING_ENERGIES["C 1s"],
            concentration_by_material={"C": 1.0},
            imfp_by_material={"vacuum": imfp_by_core["C 1s"]["C"], **imfp_by_core["C 1s"]},
            emitting_layer_indices=(1,),
        ),
        CoreLevelRequest(
            name="Ni 3p",
            binding_energy_ev=sample13.BINDING_ENERGIES["Ni 3p"],
            concentration_by_material={"LNO": 1.0},
            imfp_by_material={"vacuum": imfp_by_core["Ni 3p"]["C"], **imfp_by_core["Ni 3p"]},
            emitting_layer_indices=(3,),
        ),
    )


def make_problem(
    data: sample13.PreparedData,
    reflectivity_weight: float,
) -> JointCap3Problem:
    rc_data = tuple(
        RockingCurveData(
            name,
            data.rc_angle,
            data.rc_normalized[name],
            weight=DATASET_WEIGHTS[name],
        )
        for name in ("C 1s", "Ni 3p")
    )
    fixed_values = {
        "carbon_roughness": carbon_roughness_from_values(initial_values()),
        "lno1_roughness": lno1_roughness_from_values(initial_values()),
        "lno2_roughness": lno2_roughness_from_values(initial_values()),
        "top_lno_layer2_thickness": 47.0,
        "bottom_lno_thickness": BOTTOM_LNO_THICKNESS,
    }
    reflectivity_problem = FittingProblem(
        parameters=PARAMETERS,
        stack_builder=build_cap3_stack,
        photon_energy_ev=sample13.PHOTON_ENERGY_EV,
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
        photon_energy_ev=sample13.PHOTON_ENERGY_EV,
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


def make_jax_problem(problem: JointCap3Problem) -> JointCap3Problem:
    return JointCap3Problem(
        parameters=problem.parameters,
        reflectivity_problem=replace(problem.reflectivity_problem, simulation_backend="jax"),
        rc_problem=replace(problem.rc_problem, simulation_backend="jax"),
    )


def resolve_reflectivity_weight(
    data: sample13.PreparedData,
    requested_weight: str,
) -> tuple[float, dict[str, float]]:
    if requested_weight != "auto":
        return float(requested_weight), {}

    diagnostic_problem = make_problem(data, reflectivity_weight=1.0)
    evaluation = diagnostic_problem.evaluate(initial_values())
    raw = {contribution.name: contribution.raw for contribution in evaluation.contributions}
    weighted = {
        contribution.name: contribution.weighted
        for contribution in evaluation.contributions
    }
    rc_weighted = sum(weighted[name] for name in ("C 1s", "Ni 3p"))
    reflectivity_raw = raw["reflectivity"]
    if reflectivity_raw <= 0:
        return 1.0, {"reflectivity_raw": reflectivity_raw, "rc_weighted": rc_weighted}
    return rc_weighted / reflectivity_raw, {
        "reflectivity_raw": reflectivity_raw,
        "rc_weighted": rc_weighted,
    }


def vector_from_values(values: dict[str, float]) -> np.ndarray:
    return np.asarray([values[parameter.name] for parameter in PARAMETERS], dtype=float)


def values_from_vector(vector: np.ndarray) -> dict[str, float]:
    return {
        parameter.name: float(value)
        for parameter, value in zip(PARAMETERS, vector)
    }


def make_attempt_vectors(
    base_vector: np.ndarray,
    attempts: int,
    random_seed: int,
    perturb_fraction: float,
) -> list[np.ndarray]:
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    if perturb_fraction < 0:
        raise ValueError("perturb_fraction must be non-negative")
    lower = np.asarray([parameter.lower for parameter in PARAMETERS], dtype=float)
    upper = np.asarray([parameter.upper for parameter in PARAMETERS], dtype=float)
    widths = upper - lower
    rng = np.random.default_rng(random_seed)
    vectors = [np.clip(np.asarray(base_vector, dtype=float), lower, upper)]
    for _ in range(1, attempts):
        perturbation = rng.normal(loc=0.0, scale=perturb_fraction, size=base_vector.shape)
        vectors.append(np.clip(base_vector + perturbation * widths, lower, upper))
    return vectors


def make_finite_difference_value_and_grad(problem, rel_step: float, abs_step: float):
    lower = np.asarray([parameter.lower for parameter in PARAMETERS], dtype=float)
    upper = np.asarray([parameter.upper for parameter in PARAMETERS], dtype=float)
    widths = upper - lower
    cache: dict[tuple[float, ...], float] = {}

    def objective(vector: np.ndarray) -> float:
        physical = np.clip(np.asarray(vector, dtype=float), lower, upper)
        key = tuple(np.round(physical, 12))
        if key not in cache:
            values = {
                parameter.name: float(value)
                for parameter, value in zip(PARAMETERS, physical)
            }
            try:
                cache[key] = float(problem.evaluate(values).objective)
            except ValueError:
                cache[key] = 1.0e6
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
    problem: JointCap3Problem,
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
    plot_joint_best_fit(
        output_dir / f"{output_prefix}_best_fit.png",
        problem.reflectivity,
        problem.rocking_curves,
        simulation,
    )
    plot_stack_schematic(
        output_dir / f"{output_prefix}_stack_schematic.png",
        simulation.stack,
        title="Sample#13 Joint Cap3 JAX-Gradient Stack",
        top_layers=7,
        bottom_layers=3,
    )
    sample13.save_superlattice_profile_plot(
        result.best_parameters,
        output_dir / f"{output_prefix}_superlattice_profile.png",
    )


def save_gradient_history(path: Path, result) -> None:
    columns = [
        "iteration",
        "loss",
        "gradient_norm",
        *[parameter.name for parameter in PARAMETERS],
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
        "carbon_roughness": carbon_roughness_from_values(result.best_parameters),
        "lno1_roughness": lno1_roughness_from_values(result.best_parameters),
        "lno2_roughness": lno2_roughness_from_values(result.best_parameters),
        "top_lno_layer2_thickness": (
            result.best_parameters["top_lno_total_thickness"]
            - result.best_parameters["top_lno_layer1_thickness"]
        ),
        "bottom_lno_thickness": BOTTOM_LNO_THICKNESS,
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
    plt = sample13._load_pyplot()
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


def plot_joint_best_fit(
    path: Path,
    reflectivity_data: ReflectivityData,
    rocking_curve_data: tuple[RockingCurveData, ...],
    simulation: FitSimulation,
) -> None:
    plt = sample13._load_pyplot()
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
        *[parameter.name for parameter in PARAMETERS],
        "carbon_roughness",
        "lno1_roughness",
        "lno2_roughness",
        "top_lno_layer2_thickness",
        "bottom_lno_thickness",
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
            row["carbon_roughness"] = carbon_roughness_from_values(result.best_parameters)
            row["lno1_roughness"] = lno1_roughness_from_values(result.best_parameters)
            row["lno2_roughness"] = lno2_roughness_from_values(result.best_parameters)
            row["top_lno_layer2_thickness"] = (
                result.best_parameters["top_lno_total_thickness"]
                - result.best_parameters["top_lno_layer1_thickness"]
            )
            row["bottom_lno_thickness"] = BOTTOM_LNO_THICKNESS
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
    for parameter in PARAMETERS:
        unit = f" {parameter.unit}" if parameter.unit else ""
        print(f"    {parameter.name}: {result.best_parameters[parameter.name]:.6g}{unit}")
    print("  derived parameters:")
    print(f"    carbon_roughness: {carbon_roughness_from_values(result.best_parameters):.6g} A")
    print(f"    lno1_roughness: {lno1_roughness_from_values(result.best_parameters):.6g} A")
    print(f"    lno2_roughness: {lno2_roughness_from_values(result.best_parameters):.6g} A")
    print(
        "    top_lno_layer2_thickness: "
        f"{result.best_parameters['top_lno_total_thickness'] - result.best_parameters['top_lno_layer1_thickness']:.6g} A"
    )
    print(f"    bottom_lno_thickness: {BOTTOM_LNO_THICKNESS:.6g} A")


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
    parser.add_argument("--reflectivity-min-angle", type=float, default=sample13.RC_START_DEG)
    parser.add_argument("--reflectivity-max-angle", type=float, default=sample13.RC_STOP_DEG)
    parser.add_argument("--reflectivity-weight", default=DEFAULT_REFLECTIVITY_WEIGHT)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--output-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=130)
    parser.add_argument("--perturb-fraction", type=float, default=0.04)
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--ftol", type=float, default=1.0e-12)
    parser.add_argument("--gtol", type=float, default=1.0e-7)
    parser.add_argument("--finite-diff-rel-step", type=float, default=FINITE_DIFF_REL_STEP)
    parser.add_argument("--finite-diff-abs-step", type=float, default=FINITE_DIFF_ABS_STEP)
    parser.add_argument("--run-fit", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
