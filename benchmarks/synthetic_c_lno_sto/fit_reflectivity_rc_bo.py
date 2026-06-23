"""Fit synthetic C/LNO/STO reflectivity plus SW-XPS rocking curves with BO.

The default run fits a downsampled synthetic dataset with a joint objective.
Use ``--staged`` to first fit period, surface, and roughness parameter groups
before a final all-parameter BO pass.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import shutil
import sys

import numpy as np

CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
RUNS_DIR = REPO_ROOT / "runs" / "synthetic_c_lno_sto" / "current"
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swxps import (  # noqa: E402
    BayesianOptimizationSettings,
    CoreLevelRequest,
    FitParameter,
    FitStage,
    FittingProblem,
    LayerTemplate,
    ReflectivityData,
    ReflectivityRequest,
    RockingCurveData,
    RockingCurveRequest,
    StackTemplate,
    SuperlatticeTemplate,
    energy_to_wavelength,
    imfp_from_file,
    plot_fit_convergence,
    plot_stack_schematic,
    plot_surrogate_slices,
    run_bayesian_fit,
    run_staged_multistart_bayesian_fit,
    save_fit_history_csv,
    save_staged_fit_summary_csv,
    simulate_reflectivity,
    simulate_rocking_curves,
)

PHOTON_ENERGY_EV = 1000.0
SUPERLATTICE_REPEATS = 20
DATA_FILE = CASE_DIR / "lno_sto_c_synthetic_data.csv"
DATA_PLOT = CASE_DIR / "lno_sto_c_synthetic_data.png"
DEFAULT_OUTPUT_PREFIX = "lno_sto_c_joint_bo"

TRUE_VALUES = {
    "carbon_thickness": 10.0,
    "carbon_roughness_fraction": 1.0 / 3.0,
    "lno_thickness": 20.0,
    "sto_thickness": 20.0,
    "superlattice_roughness": 3.0,
    "substrate_roughness": 3.0,
    "angle_offset": 0.0,
}

PARAMETERS = (
    FitParameter("carbon_thickness", 4.0, 16.0, "A", initial=8.5),
    FitParameter("carbon_roughness_fraction", 0.0, 1.0, "", initial=0.2),
    FitParameter("lno_thickness", 18.0, 22.0, "A", initial=19.2),
    FitParameter("sto_thickness", 18.0, 22.0, "A", initial=20.8),
    FitParameter("superlattice_roughness", 1.0, 5.0, "A", initial=2.2),
    FitParameter("substrate_roughness", 1.0, 5.0, "A", initial=3.8),
    FitParameter("angle_offset", -0.25, 0.25, "deg", initial=0.08),
)
PARAMETER_BY_NAME = {parameter.name: parameter for parameter in PARAMETERS}

RC_COLUMN_BY_NAME = {
    "La 4d": "la4d_rc",
    "O 1s": "o1s_rc",
    "Ti 2p": "ti2p_rc",
    "C 1s": "c1s_rc",
}
RC_WEIGHTS = {
    "La 4d": 5.0,
    "O 1s": 5.0,
    "Ti 2p": 5.0,
    "C 1s": 5.0,
}
PLOT_COLORS = {
    "reflectivity": "black",
    "La 4d": "tab:purple",
    "O 1s": "tab:green",
    "Ti 2p": "tab:orange",
    "C 1s": "tab:brown",
}


def carbon_roughness_from_values(values: dict[str, float]) -> float:
    """Map the fitted fraction to a valid C roughness."""

    max_roughness = min(5.0, values["carbon_thickness"])
    return 1.0 + values["carbon_roughness_fraction"] * (max_roughness - 1.0)


def stack_template(
    carbon_roughness: float | str = "carbon_roughness",
) -> StackTemplate:
    """Return the vacuum/C/[LNO/STO]xN/STO synthetic stack recipe."""

    return StackTemplate(
        energy_ev=PHOTON_ENERGY_EV,
        base_dir=REPO_ROOT,
        parts=(
            LayerTemplate.vacuum(),
            LayerTemplate.from_file(
                "C",
                "OPC/C.dat",
                thickness="carbon_thickness",
                roughness=carbon_roughness,
            ),
            SuperlatticeTemplate(
                repeats=SUPERLATTICE_REPEATS,
                period=(
                    LayerTemplate.from_file(
                        "LNO",
                        "OPC/LaNiO3.dat",
                        thickness="lno_thickness",
                        roughness="superlattice_roughness",
                    ),
                    LayerTemplate.from_file(
                        "STO",
                        "OPC/SrTiO3.dat",
                        thickness="sto_thickness",
                        roughness="superlattice_roughness",
                    ),
                ),
            ),
            LayerTemplate.from_file(
                "STO",
                "OPC/SrTiO3.dat",
                thickness=0.0,
                roughness="substrate_roughness",
            ),
        ),
    )


def stack_values(values: dict[str, float]) -> dict[str, float]:
    """Return values including derived roughness parameters."""

    return {
        **values,
        "carbon_roughness": carbon_roughness_from_values(values),
    }


def build_stack(values: dict[str, float]):
    """Build the stack used by the fitting objective."""

    return stack_template().build(stack_values(values))


def true_stack_values() -> dict[str, float]:
    """Return the exact values used to generate the synthetic dataset."""

    return {
        **TRUE_VALUES,
        "carbon_roughness": carbon_roughness_from_values(TRUE_VALUES),
    }


def core_level_requests() -> tuple[CoreLevelRequest, ...]:
    """Return material-selective La, O, Ti, and C core-level requests."""

    binding_energies = {
        "La 4d": 105.0,
        "O 1s": 530.0,
        "Ti 2p": 460.0,
        "C 1s": 285.0,
    }
    imfp_files = {
        "C": REPO_ROOT / "IMFP" / "C.ANG",
        "LNO": REPO_ROOT / "IMFP" / "LNO.ANG",
        "STO": REPO_ROOT / "IMFP" / "STO.ANG",
    }
    imfp_by_core = {}
    for core_name, binding_energy in binding_energies.items():
        kinetic_energy = PHOTON_ENERGY_EV - binding_energy
        imfp_by_core[core_name] = {
            material: imfp_from_file(path, kinetic_energy)
            for material, path in imfp_files.items()
        }

    def imfp_map(core_name: str) -> dict[str, float]:
        return {
            "vacuum": imfp_by_core[core_name]["C"],
            **imfp_by_core[core_name],
        }

    return (
        CoreLevelRequest(
            name="La 4d",
            binding_energy_ev=binding_energies["La 4d"],
            concentration_by_material={"LNO": 1.0},
            imfp_by_material=imfp_map("La 4d"),
        ),
        CoreLevelRequest(
            name="O 1s",
            binding_energy_ev=binding_energies["O 1s"],
            concentration_by_material={"LNO": 1.0, "STO": 1.0},
            imfp_by_material=imfp_map("O 1s"),
        ),
        CoreLevelRequest(
            name="Ti 2p",
            binding_energy_ev=binding_energies["Ti 2p"],
            concentration_by_material={"STO": 1.0},
            imfp_by_material=imfp_map("Ti 2p"),
        ),
        CoreLevelRequest(
            name="C 1s",
            binding_energy_ev=binding_energies["C 1s"],
            concentration_by_material={"C": 1.0},
            imfp_by_material=imfp_map("C 1s"),
        ),
    )


def generate_synthetic_data(
    path: Path = DATA_FILE,
    plot_path: Path = DATA_PLOT,
    angle_count: int = 161,
) -> None:
    """Generate the noiseless synthetic dataset and preview plot."""

    period = TRUE_VALUES["lno_thickness"] + TRUE_VALUES["sto_thickness"]
    wavelength = energy_to_wavelength(PHOTON_ENERGY_EV)
    bragg_angle = np.rad2deg(np.arcsin(wavelength / (2.0 * period)))
    angles = np.linspace(bragg_angle - 2.0, bragg_angle + 2.0, angle_count)
    stack = stack_template(carbon_roughness="carbon_roughness").build(true_stack_values())

    reflectivity = simulate_reflectivity(
        ReflectivityRequest(
            angles=angles,
            energy_ev=PHOTON_ENERGY_EV,
            stack=stack,
            roughness_step=1.0,
            slicing=None,
        )
    ).reflectivity
    peak_angle = angles[np.argmax(reflectivity)]
    offpeak_mask = np.abs(angles - peak_angle) > 1.25
    rc_result = simulate_rocking_curves(
        RockingCurveRequest(
            angles=angles,
            photon_energy_ev=PHOTON_ENERGY_EV,
            stack=stack,
            core_levels=core_level_requests(),
            field_step=1.0,
            roughness_step=1.0,
            offpeak_mask=offpeak_mask,
            slicing=None,
        )
    )
    curves = {core.name: core.curve.intensity for core in rc_result.core_levels}
    table = np.column_stack(
        [
            angles,
            reflectivity,
            curves["La 4d"],
            curves["O 1s"],
            curves["Ti 2p"],
            curves["C 1s"],
        ]
    )
    np.savetxt(
        path,
        table,
        delimiter=",",
        header="angle_deg,reflectivity,la4d_rc,o1s_rc,ti2p_rc,c1s_rc",
        comments="",
    )
    plot_synthetic_data(plot_path, angles, reflectivity, curves, bragg_angle, peak_angle)


def plot_synthetic_data(
    path: Path,
    angles: np.ndarray,
    reflectivity: np.ndarray,
    curves: dict[str, np.ndarray],
    bragg_angle: float,
    peak_angle: float,
) -> None:
    """Save a quick synthetic-data overview plot."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 1, figsize=(7.6, 9.0), sharex=True)
    axes[0].semilogy(angles, reflectivity, color="black", linewidth=1.4)
    axes[0].set_ylabel("Reflectivity")
    axes[0].grid(True, which="both", alpha=0.25)
    for ax, name in zip(axes[1:], RC_COLUMN_BY_NAME):
        ax.plot(angles, curves[name], linewidth=1.5, label=name)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0)
        ax.set_ylabel("Norm. RC")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    for ax in axes:
        ax.axvline(bragg_angle, color="tab:red", linestyle="--", linewidth=1.0)
        ax.axvline(peak_angle, color="tab:blue", linestyle="-", linewidth=1.0)
    axes[-1].set_xlabel("Grazing incidence angle (deg)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def load_data(path: Path, stride: int = 1) -> dict[str, np.ndarray]:
    """Load the synthetic CSV, optionally using every Nth point."""

    if stride <= 0:
        raise ValueError("stride must be positive")
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {name: np.asarray(data[name][::stride], dtype=float) for name in data.dtype.names}


def make_problem(
    data: dict[str, np.ndarray],
    reflectivity_weight: float,
    rc_weights: dict[str, float] | None = None,
) -> FittingProblem:
    """Create a weighted joint reflectivity plus four-RC fitting problem."""

    rc_weights = RC_WEIGHTS if rc_weights is None else rc_weights
    angles = data["angle_deg"]
    peak_angle = angles[np.argmax(data["reflectivity"])]
    offpeak_mask = np.abs(angles - peak_angle) > 1.25
    reflectivity = ReflectivityData(
        name="reflectivity",
        angles=angles,
        reflectivity=data["reflectivity"],
        weight=reflectivity_weight,
        log_floor=1.0e-12,
    )
    rocking_curves = tuple(
        RockingCurveData(
            name=name,
            angles=angles,
            intensity=data[column],
            weight=rc_weights[name],
        )
        for name, column in RC_COLUMN_BY_NAME.items()
    )
    return FittingProblem(
        parameters=PARAMETERS,
        stack_builder=build_stack,
        photon_energy_ev=PHOTON_ENERGY_EV,
        reflectivity=reflectivity,
        rocking_curves=rocking_curves,
        core_levels=core_level_requests(),
        angle_offset_parameter="angle_offset",
        field_step=1.0,
        roughness_step=1.0,
        offpeak_mask=offpeak_mask,
    )


def initial_values() -> dict[str, float]:
    """Return parameter initial values."""

    return {
        parameter.name: float(parameter.initial)
        for parameter in PARAMETERS
        if parameter.initial is not None
    }


def resolve_reflectivity_weight(
    data: dict[str, np.ndarray],
    requested_weight: str,
) -> tuple[float, dict[str, float]]:
    """Return a numeric reflectivity weight and setup diagnostics."""

    if requested_weight != "auto":
        return float(requested_weight), {}
    problem = make_problem(data, reflectivity_weight=1.0)
    evaluation = problem.evaluate(initial_values())
    raw = {contribution.name: contribution.raw for contribution in evaluation.contributions}
    weighted = {
        contribution.name: contribution.weighted
        for contribution in evaluation.contributions
    }
    rc_weighted = sum(weighted[name] for name in RC_COLUMN_BY_NAME)
    reflectivity_raw = raw["reflectivity"]
    if reflectivity_raw <= 0:
        return 1.0, {"reflectivity_raw": reflectivity_raw, "rc_weighted": rc_weighted}
    return rc_weighted / reflectivity_raw, {
        "reflectivity_raw": reflectivity_raw,
        "rc_weighted": rc_weighted,
    }


def fit_stages() -> tuple[FitStage, ...]:
    """Return a conservative staged fitting sequence."""

    return (
        FitStage(
            "period_and_offset",
            (
                PARAMETER_BY_NAME["lno_thickness"],
                PARAMETER_BY_NAME["sto_thickness"],
                PARAMETER_BY_NAME["angle_offset"],
            ),
        ),
        FitStage(
            "surface",
            (
                PARAMETER_BY_NAME["carbon_thickness"],
                PARAMETER_BY_NAME["carbon_roughness_fraction"],
            ),
        ),
        FitStage(
            "roughness",
            (
                PARAMETER_BY_NAME["superlattice_roughness"],
                PARAMETER_BY_NAME["substrate_roughness"],
            ),
        ),
        FitStage("final_all", PARAMETERS),
    )


def run_fit(args: argparse.Namespace) -> None:
    """Run BO and save diagnostics."""

    if args.regenerate_data or not DATA_FILE.exists():
        generate_synthetic_data(DATA_FILE, DATA_PLOT, angle_count=args.angle_count)
        print(f"Saved {DATA_FILE}")
        print(f"Saved {DATA_PLOT}")
    if args.generate_only:
        return

    fit_data = load_data(DATA_FILE, stride=args.stride)
    reflectivity_weight, diagnostics = resolve_reflectivity_weight(
        fit_data,
        args.reflectivity_weight,
    )
    if args.multi_seed:
        run_multi_seed_fit(args, fit_data, reflectivity_weight, diagnostics)
        return

    problem = make_problem(fit_data, reflectivity_weight)
    settings = settings_from_args(args, args.random_state)

    if args.staged:
        result = run_staged_multistart_bayesian_fit(
            problem,
            fit_stages(),
            settings,
            n_starts=args.n_starts,
            initial_values=initial_values(),
            random_seed=args.random_state,
        )
        best_parameters = result.best_parameters
        final_run = result.stages[-1].best_run.result
        history = final_run.history
        best_objective = result.best_objective
        save_staged_fit_summary_csv(
            RUNS_DIR / f"{args.output_prefix}_staged_summary.csv",
            result,
            PARAMETERS,
        )
    else:
        final_run = run_bayesian_fit(problem, settings)
        best_parameters = final_run.best_parameters
        history = final_run.history
        best_objective = final_run.best_objective

    save_result_artifacts(
        RUNS_DIR,
        args.output_prefix,
        final_run,
        best_parameters,
        reflectivity_weight,
    )
    print_summary(best_objective, best_parameters, reflectivity_weight, diagnostics)


def run_multi_seed_fit(
    args: argparse.Namespace,
    fit_data: dict[str, np.ndarray],
    reflectivity_weight: float,
    diagnostics: dict[str, float],
) -> None:
    """Run several direct BO seeds, summarize them, and promote the best."""

    seeds = parse_seed_list(args.seeds, args.seed_start, args.seed_count)
    run_dir = RUNS_DIR / args.multi_seed_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    full_problem = make_problem(load_data(DATA_FILE), reflectivity_weight)

    rows = []
    for index, seed in enumerate(seeds, start=1):
        print(f"Multi-seed run {index}/{len(seeds)}: seed={seed}", flush=True)
        seed_prefix = f"{args.output_prefix}_seed{seed}"
        seed_dir = run_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        problem = make_problem(fit_data, reflectivity_weight)
        result = run_bayesian_fit(problem, settings_from_args(args, seed))
        save_result_artifacts(
            seed_dir,
            seed_prefix,
            result,
            result.best_parameters,
            reflectivity_weight,
            full_problem=full_problem,
        )
        rows.append(summary_row(seed, seed_dir, seed_prefix, result))

    best = min(rows, key=lambda row: row["objective"])
    summary_path = run_dir / f"{args.output_prefix}_summary.csv"
    save_multi_seed_summary(summary_path, rows)
    promote_best_run(run_dir, args.output_prefix, best)
    print(f"Saved {summary_path}")
    print(f"Promoted best seed {best['seed']} to {run_dir / (args.output_prefix + '_best_*')}")
    print_summary(
        best["objective"],
        {parameter.name: best[parameter.name] for parameter in PARAMETERS},
        reflectivity_weight,
        diagnostics,
    )


def settings_from_args(
    args: argparse.Namespace,
    random_state: int,
) -> BayesianOptimizationSettings:
    """Return BO settings for one run."""

    return BayesianOptimizationSettings(
        n_calls=args.n_calls,
        n_initial_points=args.n_initial_points,
        acquisition_function=args.acquisition_function,
        random_state=random_state,
        show_progress=args.progress,
        progress_interval=args.progress_interval,
    )


def save_result_artifacts(
    output_dir: Path,
    output_prefix: str,
    result,
    best_parameters: dict[str, float],
    reflectivity_weight: float,
    full_problem: FittingProblem | None = None,
) -> None:
    """Save history, diagnostic plots, and stack schematic for one BO result."""

    full_problem = (
        make_problem(load_data(DATA_FILE), reflectivity_weight)
        if full_problem is None
        else full_problem
    )
    best_simulation = full_problem.simulate(best_parameters)
    save_fit_history_csv(output_dir / f"{output_prefix}_history.csv", result.history, PARAMETERS)
    plot_fit_convergence(output_dir / f"{output_prefix}_convergence.png", result.history)
    plot_colored_best_fit(
        output_dir / f"{output_prefix}_best_fit.png",
        full_problem.reflectivity,
        full_problem.rocking_curves,
        best_simulation,
    )
    plot_surrogate_slices(
        output_dir / f"{output_prefix}_surrogate_slices.png",
        result,
        PARAMETERS,
        TRUE_VALUES,
    )
    plot_stack_schematic(
        output_dir / f"{output_prefix}_stack_schematic.png",
        best_simulation.stack,
        title="Synthetic C/LNO/STO BO Stack",
        top_layers=4,
        bottom_layers=3,
    )


def parse_seed_list(
    seeds: str,
    seed_start: int,
    seed_count: int,
) -> tuple[int, ...]:
    """Return explicit seeds or a consecutive seed range."""

    if seeds:
        parsed = tuple(int(seed.strip()) for seed in seeds.split(",") if seed.strip())
        if not parsed:
            raise ValueError("--seeds did not contain any integers")
        return parsed
    if seed_count <= 0:
        raise ValueError("--seed-count must be positive")
    return tuple(seed_start + index for index in range(seed_count))


def summary_row(seed: int, seed_dir: Path, seed_prefix: str, result) -> dict[str, float | str]:
    """Return one CSV-ready summary row for a seed run."""

    best = result.best_evaluation
    row: dict[str, float | str] = {
        "seed": seed,
        "objective": result.best_objective,
        "best_evaluation": best_index(result.history.evaluations, best),
        "output_dir": seed_dir.name,
        "output_prefix": seed_prefix,
        "carbon_roughness": carbon_roughness_from_values(result.best_parameters),
        "total_seconds": result.timing.total_seconds,
        "objective_seconds": result.timing.objective_seconds,
        "optimizer_overhead_seconds": result.timing.optimizer_overhead_seconds,
    }
    row.update(result.best_parameters)
    for contribution in best.contributions:
        row[f"{contribution.name}_raw"] = contribution.raw
        row[f"{contribution.name}_weighted"] = contribution.weighted
    return row


def best_index(evaluations: Sequence, best) -> int:
    """Return the one-based index of a best evaluation object."""

    for index, evaluation in enumerate(evaluations, start=1):
        if evaluation is best:
            return index
    return min(
        range(1, len(evaluations) + 1),
        key=lambda index: evaluations[index - 1].objective,
    )


def save_multi_seed_summary(path: Path, rows: Sequence[dict[str, float | str]]) -> None:
    """Save multi-seed result rows to CSV."""

    columns = [
        "seed",
        "objective",
        "best_evaluation",
        "output_dir",
        "output_prefix",
        "total_seconds",
        "objective_seconds",
        "optimizer_overhead_seconds",
        *[parameter.name for parameter in PARAMETERS],
        "carbon_roughness",
        "reflectivity_raw",
        "reflectivity_weighted",
        *[
            item
            for name in RC_COLUMN_BY_NAME
            for item in (f"{name}_raw", f"{name}_weighted")
        ],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join(str(row.get(column, "")) for column in columns) + "\n")


def promote_best_run(
    run_dir: Path,
    output_prefix: str,
    best: dict[str, float | str],
) -> None:
    """Copy the best seed's artifacts to stable best-run filenames."""

    seed_dir = run_dir / str(best["output_dir"])
    seed_prefix = str(best["output_prefix"])
    artifact_map = (
        ("history.csv", "best_history.csv"),
        ("convergence.png", "best_convergence.png"),
        ("best_fit.png", "best_fit.png"),
        ("surrogate_slices.png", "best_surrogate_slices.png"),
        ("stack_schematic.png", "best_stack_schematic.png"),
    )
    for suffix, promoted_name in artifact_map:
        source = seed_dir / f"{seed_prefix}_{suffix}"
        destination = run_dir / f"{output_prefix}_{promoted_name}"
        shutil.copy2(source, destination)


def plot_colored_best_fit(
    path: Path,
    reflectivity_data: ReflectivityData,
    rocking_curve_data: tuple[RockingCurveData, ...],
    simulation,
) -> None:
    """Save fit overlays with distinct measured-data colors for each RC."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1 + len(rocking_curve_data),
        1,
        figsize=(7.6, 2.0 * (1 + len(rocking_curve_data))),
        sharex=True,
    )
    axes = np.asarray(axes).ravel()

    ax = axes[0]
    ax.semilogy(
        reflectivity_data.angles,
        reflectivity_data.reflectivity,
        "o",
        color=PLOT_COLORS["reflectivity"],
        markersize=3,
        alpha=0.55,
        label="data",
    )
    ax.semilogy(
        simulation.reflectivity.angle,
        simulation.reflectivity.reflectivity,
        color="tab:red",
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
        color = PLOT_COLORS[data.name]
        ax.plot(
            data.angles,
            data.intensity,
            "o",
            color=color,
            markersize=3,
            alpha=0.55,
            label=f"{data.name} data",
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


def print_summary(
    best_objective: float,
    best_parameters: dict[str, float],
    reflectivity_weight: float,
    diagnostics: dict[str, float],
) -> None:
    """Print the most important fit results."""

    print(f"Reflectivity weight: {reflectivity_weight:.6g}")
    if diagnostics:
        print(
            "Initial balance: "
            f"reflectivity raw={diagnostics['reflectivity_raw']:.6g}, "
            f"weighted RC sum={diagnostics['rc_weighted']:.6g}"
        )
    print(f"Best objective: {best_objective:.6g}")
    print("Best parameters:")
    for parameter in PARAMETERS:
        value = best_parameters[parameter.name]
        true_value = TRUE_VALUES[parameter.name]
        unit = f" {parameter.unit}" if parameter.unit else ""
        print(f"  {parameter.name}: {value:.6g}{unit} (true {true_value:.6g}{unit})")
    print(
        "  carbon_roughness: "
        f"{carbon_roughness_from_values(best_parameters):.6g} A "
        f"(true {carbon_roughness_from_values(TRUE_VALUES):.6g} A)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate-data", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--angle-count", type=int, default=161)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--reflectivity-weight", default="auto")
    parser.add_argument("--n-calls", type=int, default=80)
    parser.add_argument("--n-initial-points", type=int, default=24)
    parser.add_argument("--random-state", type=int, default=12)
    parser.add_argument("--acquisition-function", choices=("EI", "LCB", "PI"), default="EI")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-interval", type=int, default=5)
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--n-starts", type=int, default=2)
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument(
        "--multi-seed-dir",
        default="multi_seed_outputs",
        help="Folder inside the example directory for per-seed outputs.",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="Comma-separated explicit seeds. Overrides --seed-start/--seed-count.",
    )
    parser.add_argument("--seed-start", type=int, default=12)
    parser.add_argument("--seed-count", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    run_fit(parse_args())


if __name__ == "__main__":
    main()
