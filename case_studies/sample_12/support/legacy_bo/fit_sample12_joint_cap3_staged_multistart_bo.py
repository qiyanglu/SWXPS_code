"""Run staged multi-start BO for Sample#12 joint cap3 fitting.

Each stage/start writes diagnostics into its own folder. The best run from a
stage seeds the next stage. The final best result is promoted to stable files in
the output directory.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path
import shutil
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
    plot_fit_convergence,
    plot_stack_schematic,
    plot_surrogate_slices,
    run_bayesian_optimization,
    save_fit_history_csv,
)

import fit_sample12_joint_cap3_bo as joint  # noqa: E402


@dataclass(frozen=True)
class StageDefinition:
    """One staged BO step."""

    name: str
    parameter_names: tuple[str, ...]


STAGES = (
    StageDefinition(
        "geometry",
        (
            "reflectivity_angle_offset",
            "rc_angle_offset",
            "sto_thickness_start",
            "lno_thickness_start",
            "sto_thickness_delta",
            "lno_thickness_delta",
            "thickness_transition_repeat",
            "thickness_transition_width",
        ),
    ),
    StageDefinition(
        "cap",
        (
            "carbon_thickness",
            "carbon_roughness_fraction",
            "top_lno_total_thickness",
            "top_lno_layer1_thickness",
            "cap_roughness_fraction",
        ),
    ),
    StageDefinition(
        "roughness",
        (
            "sto_roughness_first",
            "sto_roughness_last",
            "lno_roughness_first",
            "lno_roughness_last",
            "substrate_roughness",
        ),
    ),
    StageDefinition(
        "final_all",
        tuple(parameter.name for parameter in joint.PARAMETERS),
    ),
)


@dataclass(frozen=True)
class RunRecord:
    """Summary for one stage/start run."""

    stage: str
    start_index: int
    random_state: int
    objective: float
    folder: Path
    best_parameters: dict[str, float]
    contributions: dict[str, float]


def stage_parameters(
    parameter_names: tuple[str, ...],
    current_values: dict[str, float],
):
    """Return active parameters with initials set from current best values."""

    selected = []
    for name in parameter_names:
        parameter = joint.PARAMETER_BY_NAME[name]
        selected.append(replace(parameter, initial=current_values.get(name, parameter.initial)))
    return tuple(selected)


def make_stage_problem(
    data,
    reflectivity_weight: float,
    active_parameters,
    current_values: dict[str, float],
) -> joint.JointCap3Problem:
    """Create a joint problem with only selected active parameters."""

    fixed_values = {
        **current_values,
        "carbon_roughness": joint.carbon_roughness_from_values(current_values),
        "cap_roughness": joint.cap_roughness_from_values(current_values),
        "top_lno_layer2_thickness": (
            current_values["top_lno_total_thickness"]
            - current_values["top_lno_layer1_thickness"]
        ),
        "bottom_lno_thickness": joint.BOTTOM_LNO_THICKNESS,
    }
    rc_data = tuple(
        joint.RockingCurveData(
            name,
            data.rc_angle,
            data.rc_normalized[name],
            weight=joint.DATASET_WEIGHTS[name],
        )
        for name in joint.sample12.RC_NAMES
    )
    reflectivity_problem = FittingProblem(
        parameters=active_parameters,
        stack_builder=joint.build_cap3_stack,
        photon_energy_ev=joint.sample12.PHOTON_ENERGY_EV,
        reflectivity=joint.ReflectivityData(
            name="reflectivity",
            angles=data.reflectivity_angle,
            reflectivity=data.reflectivity_raw,
            weight=reflectivity_weight,
            log_floor=1.0e-12,
        ),
        angle_offset_parameter="reflectivity_angle_offset",
        field_step=5.0,
        roughness_step=2.0,
        fixed_values=fixed_values,
    )
    rc_problem = FittingProblem(
        parameters=active_parameters,
        stack_builder=joint.build_cap3_stack,
        photon_energy_ev=joint.sample12.PHOTON_ENERGY_EV,
        rocking_curves=rc_data,
        core_levels=joint.core_level_requests(),
        angle_offset_parameter="rc_angle_offset",
        field_step=5.0,
        roughness_step=2.0,
        offpeak_mask=np.ones_like(data.rc_angle, dtype=bool),
        fixed_values=fixed_values,
    )
    return joint.JointCap3Problem(
        parameters=active_parameters,
        reflectivity_problem=reflectivity_problem,
        rc_problem=rc_problem,
    )


def save_run_artifacts(
    folder: Path,
    stage_problem: joint.JointCap3Problem,
    result,
    active_parameters,
    current_values: dict[str, float],
) -> None:
    """Save diagnostics for one stage/start run."""

    folder.mkdir(parents=True, exist_ok=True)
    best_values = {**current_values, **result.best_parameters}
    simulation = stage_problem.simulate(result.best_parameters)
    save_fit_history_csv(folder / "history.csv", result.history, active_parameters)
    plot_fit_convergence(folder / "convergence.png", result.history)
    joint.plot_joint_best_fit(
        folder / "best_fit.png",
        stage_problem.reflectivity,
        stage_problem.rocking_curves,
        simulation,
    )
    plot_surrogate_slices(
        folder / "surrogate_slices.png",
        result,
        active_parameters,
        best_values,
    )
    plot_stack_schematic(
        folder / "stack_schematic.png",
        simulation.stack,
        title=f"Sample#12 {folder.name} Stack",
        top_layers=6,
        bottom_layers=3,
    )


def contribution_summary(result) -> dict[str, float]:
    """Return raw and weighted contribution values from a BO result."""

    values = {}
    for contribution in result.best_evaluation.contributions:
        values[f"{contribution.name}_raw"] = contribution.raw
        values[f"{contribution.name}_weighted"] = contribution.weighted
    return values


def stage_seed(base_seed: int, stage_index: int, start_index: int) -> int:
    """Return a deterministic seed for one stage/start."""

    return base_seed + 1000 * stage_index + start_index


def run_staged_multistart(
    data,
    reflectivity_weight: float,
    output_dir: Path,
    n_calls: int,
    n_initial_points: int,
    n_starts: int,
    random_seed: int,
    show_progress: bool,
    progress_interval: int,
) -> tuple[list[RunRecord], dict[str, float]]:
    """Run all stages and return run records plus final best values."""

    output_dir.mkdir(parents=True, exist_ok=True)
    current_values = joint.initial_values()
    records: list[RunRecord] = []

    for stage_index, stage in enumerate(STAGES):
        stage_records: list[RunRecord] = []
        for start_index in range(n_starts):
            seed = stage_seed(random_seed, stage_index, start_index)
            active_parameters = stage_parameters(stage.parameter_names, current_values)
            problem = make_stage_problem(
                data,
                reflectivity_weight,
                active_parameters,
                current_values,
            )
            print(
                f"Stage {stage.name}, start {start_index}, seed {seed}: "
                f"{len(active_parameters)} parameters",
                flush=True,
            )
            result = run_bayesian_optimization(
                problem.objective(),
                BayesianOptimizationSettings(
                    n_calls=n_calls,
                    n_initial_points=n_initial_points,
                    random_state=seed,
                    show_progress=show_progress,
                    progress_interval=progress_interval,
                ),
            )
            folder = output_dir / f"{stage_index + 1:02d}_{stage.name}" / f"start_{start_index:02d}_seed_{seed}"
            save_run_artifacts(
                folder,
                problem,
                result,
                active_parameters,
                current_values,
            )
            record = RunRecord(
                stage=stage.name,
                start_index=start_index,
                random_state=seed,
                objective=result.best_objective,
                folder=folder,
                best_parameters=dict(result.best_parameters),
                contributions=contribution_summary(result),
            )
            stage_records.append(record)
            records.append(record)
            print(f"  best objective: {result.best_objective:.6g}", flush=True)

        best_record = min(stage_records, key=lambda record: record.objective)
        current_values.update(best_record.best_parameters)
        print(
            f"Selected stage {stage.name} start {best_record.start_index} "
            f"objective={best_record.objective:.6g}",
            flush=True,
        )

    return records, current_values


def save_summary(path: Path, records: list[RunRecord], final_values: dict[str, float]) -> None:
    """Write one CSV summary row per stage/start."""

    contribution_names = sorted(
        {
            key
            for record in records
            for key in record.contributions
        }
    )
    parameter_names = [parameter.name for parameter in joint.PARAMETERS]
    columns = [
        "stage",
        "start_index",
        "random_state",
        "objective",
        "folder",
        *contribution_names,
        *parameter_names,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = {
                "stage": record.stage,
                "start_index": record.start_index,
                "random_state": record.random_state,
                "objective": record.objective,
                "folder": str(record.folder),
            }
            row.update(record.contributions)
            row.update(record.best_parameters)
            writer.writerow(row)

    final_path = path.with_name("final_best_parameters.csv")
    with final_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("parameter", "value"))
        for name in parameter_names:
            writer.writerow((name, final_values[name]))
        writer.writerow(("carbon_roughness", joint.carbon_roughness_from_values(final_values)))
        writer.writerow(("cap_roughness", joint.cap_roughness_from_values(final_values)))
        writer.writerow(
            (
                "top_lno_layer2_thickness",
                final_values["top_lno_total_thickness"] - final_values["top_lno_layer1_thickness"],
            )
        )


def promote_final_best(
    output_dir: Path,
    data,
    reflectivity_weight: float,
    final_values: dict[str, float],
) -> None:
    """Save promoted final-best artifacts in the staged run root."""

    problem = joint.make_problem(data, reflectivity_weight)
    simulation = problem.simulate(final_values)
    joint.plot_joint_best_fit(
        output_dir / "best_fit.png",
        problem.reflectivity,
        problem.rocking_curves,
        simulation,
    )
    plot_stack_schematic(
        output_dir / "stack_schematic.png",
        simulation.stack,
        title="Sample#12 Staged Multi-Start Best Stack",
        top_layers=6,
        bottom_layers=3,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "sample_12" / "staged_multistart")
    parser.add_argument("--background-percent", type=float, default=10.0)
    parser.add_argument("--background-order", type=int, default=2)
    parser.add_argument("--reflectivity-min-angle", type=float, default=joint.sample12.RC_START_DEG)
    parser.add_argument("--reflectivity-max-angle", type=float, default=joint.sample12.RC_STOP_DEG)
    parser.add_argument("--reflectivity-weight", default="0.0503187")
    parser.add_argument("--n-calls", type=int, default=120)
    parser.add_argument("--n-initial-points", type=int, default=40)
    parser.add_argument("--n-starts", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=12)
    parser.add_argument("--run-fit", action="store_true")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-interval", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = joint.sample12.load_and_prepare_data(args.background_percent, args.background_order)
    data = joint.sample12.apply_reflectivity_window(
        data,
        args.reflectivity_min_angle,
        args.reflectivity_max_angle,
    )
    reflectivity_weight, diagnostics = joint.resolve_reflectivity_weight(
        data,
        args.reflectivity_weight,
    )
    print("Sample#12 staged multi-start BO setup")
    print(f"Output directory: {args.output_dir}")
    print(f"Reflectivity weight: {reflectivity_weight:g}")
    if diagnostics:
        print(f"Initial reflectivity raw log-MSE: {diagnostics['reflectivity_raw']:.6g}")
        print(f"Initial total weighted RC MSE: {diagnostics['rc_weighted']:.6g}")
    print(f"Stages: {', '.join(stage.name for stage in STAGES)}")
    print(f"Starts per stage: {args.n_starts}")
    print(f"Calls per start: {args.n_calls}")
    print(f"Initial points per start: {args.n_initial_points}")
    if not args.run_fit:
        print("BO fitting was not run. Re-run with --run-fit after checking this setup.")
        return

    records, final_values = run_staged_multistart(
        data,
        reflectivity_weight,
        args.output_dir,
        args.n_calls,
        args.n_initial_points,
        args.n_starts,
        args.random_seed,
        args.progress,
        args.progress_interval,
    )
    save_summary(args.output_dir / "summary.csv", records, final_values)
    promote_final_best(args.output_dir, data, reflectivity_weight, final_values)

    final_stage_records = [record for record in records if record.stage == STAGES[-1].name]
    best_final = min(final_stage_records, key=lambda record: record.objective)
    print("Final best:")
    print(f"  objective: {best_final.objective:.6g}")
    print(f"  folder: {best_final.folder}")
    for name, value in final_values.items():
        print(f"  {name}: {value:.6g}")
    print(f"  carbon_roughness: {joint.carbon_roughness_from_values(final_values):.6g}")
    print(f"  cap_roughness: {joint.cap_roughness_from_values(final_values):.6g}")


if __name__ == "__main__":
    main()
