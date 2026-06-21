"""Promote the longer Sample#13 TRF result when it beats the saved best result."""

from __future__ import annotations

import csv
from pathlib import Path
import shutil
import sys


FIT_DIR = Path(__file__).resolve().parent
CASE_DIR = FIT_DIR.parent
REPO_ROOT = CASE_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
RUN_DIR = REPO_ROOT / "runs" / "sample_13" / "jax_least_squares" / "long_run"
BEST_DIR = CASE_DIR / "best_results_so_far"
SUMMARY_PATH = RUN_DIR / "summary.csv"
OLD_SUMMARY_NAME = "sample13_joint_cap3_lnorough_jax_gradient_best_summary.csv"
CONCENTRATION_PREFIX = "sample13_trf_all_rcs_top30_vertical_concentration_profiles"

for path in (SRC_DIR, FIT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from swxps import plot_vertical_concentration_profiles  # noqa: E402
from swxps.result_exports import (  # noqa: E402
    save_fit_curve_data_csv,
    save_optimized_stack_csv,
)

import fit_sample13_reflectivity_all_rcs_least_squares as fit13  # noqa: E402


ARTIFACTS = {
    "best_fit.png": "sample13_trf_all_rcs_best_fit.png",
    "convergence.png": "sample13_trf_all_rcs_convergence.png",
    "stack_schematic.png": "sample13_trf_all_rcs_stack_schematic.png",
    "parameter_positions.png": "sample13_trf_all_rcs_parameter_positions.png",
    "summary.csv": "sample13_trf_all_rcs_summary.csv",
    "summary.txt": "sample13_trf_all_rcs_summary.txt",
    "jax_contributions.csv": "sample13_trf_all_rcs_jax_contributions.csv",
    "numpy_validation_contributions.csv": (
        "sample13_trf_all_rcs_validation_numpy_contributions.csv"
    ),
    "history.csv": "sample13_trf_all_rcs_history.csv",
    "covariance.csv": "sample13_trf_all_rcs_covariance.csv",
    "parameter_uncertainties.csv": "sample13_trf_all_rcs_parameter_uncertainties.csv",
    "parameter_positions.csv": "sample13_trf_all_rcs_parameter_positions.csv",
}


def main() -> None:
    candidate = read_row(SUMMARY_PATH)
    reference_weight, previous_score = existing_reference()
    candidate_score = common_objective(candidate, reference_weight)
    print(f"Previous common objective: {previous_score:.12g}")
    print(f"Candidate common objective: {candidate_score:.12g}")
    if candidate_score >= previous_score:
        print("Candidate was not promoted because it did not improve the saved best.")
        return

    fit13.fit13.patch_archived_context_paths()
    values = {
        parameter.name: float(candidate[parameter.name])
        for parameter in fit13.fit13.PARAMETERS
    }
    data = fit13.fit13.sample13.load_and_prepare_data(10.0, 2)
    data = fit13.fit13.sample13.apply_reflectivity_window(
        data,
        fit13.fit13.sample13.RC_START_DEG,
        fit13.fit13.sample13.RC_STOP_DEG,
    )
    problem = fit13.make_problem(data, float(candidate["reflectivity_weight"]))
    simulation = problem.simulate(values)

    clear_directory(BEST_DIR)
    save_fit_curve_data_csv(
        BEST_DIR / "best_fit_experiment_and_simulation.csv",
        problem.reflectivity,
        problem.rocking_curves,
        simulation,
    )
    save_optimized_stack_csv(
        BEST_DIR / "optimized_stack_layers.csv",
        simulation.stack,
    )
    save_concentration_plot(simulation.stack)
    for source_name, target_name in ARTIFACTS.items():
        source = RUN_DIR / source_name
        if source.exists():
            shutil.copy2(source, BEST_DIR / target_name)
    write_selection_summary(
        BEST_DIR / "selection_comparison.csv",
        reference_weight,
        previous_score,
        candidate_score,
    )
    print(f"Promoted longer TRF result to {BEST_DIR}")


def existing_reference() -> tuple[float, float]:
    selection_path = BEST_DIR / "selection_comparison.csv"
    if selection_path.exists():
        row = read_row(selection_path)
        return float(row["reference_reflectivity_weight"]), float(
            row["promoted_common_objective"]
        )
    old = read_row(BEST_DIR / OLD_SUMMARY_NAME)
    weight = float(old["reflectivity_weighted"]) / float(old["reflectivity_raw"])
    return weight, float(old["best_numpy_objective"])


def common_objective(row: dict[str, str], reflectivity_weight: float) -> float:
    return (
        reflectivity_weight * float(row["reflectivity_raw"])
        + 0.5 * float(row["C 1s_raw"])
        + 3.0 * float(row["Ni 3p_raw"])
        + 3.0 * float(row["La 4d_raw"])
    )


def clear_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve()
    if resolved.parent != CASE_DIR.resolve():
        raise ValueError("best-results directory is outside the Sample#13 case folder")
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def save_concentration_plot(stack) -> None:
    concentrations = concentration_by_layer(stack)
    profiles = plot_vertical_concentration_profiles(
        BEST_DIR / f"{CONCENTRATION_PREFIX}.png",
        stack,
        concentrations,
        max_depth=30.0,
        step=0.1,
        title="Sample#13 top 30 A concentration profiles",
        layer_labels={
            1: "C",
            2: "LNO-1 Ni-free",
            3: "LNO-2",
            4: "LNO-bottom",
        },
        show_layer_shading=False,
        layer_box_style=True,
        categorical_strips=True,
    )
    path = BEST_DIR / f"{CONCENTRATION_PREFIX}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["depth_A", *profiles.profiles])
        for index, depth in enumerate(profiles.depth):
            writer.writerow(
                [depth, *[values[index] for values in profiles.profiles.values()]]
            )


def concentration_by_layer(stack) -> dict[str, list[float]]:
    carbon = []
    lanthanum = []
    nickel = []
    for index, material in enumerate(stack.materials):
        carbon.append(1.0 if material == "C" else 0.0)
        lanthanum.append(1.0 if material == "LNO" else 0.0)
        nickel.append(1.0 if material == "LNO" and index != 2 else 0.0)
    return {"La": lanthanum, "Ni": nickel, "C": carbon}


def write_selection_summary(
    path: Path,
    reference_weight: float,
    previous_score: float,
    candidate_score: float,
) -> None:
    fields = [
        "reference_reflectivity_weight",
        "previous_common_objective",
        "promoted_common_objective",
        "improvement_percent",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "reference_reflectivity_weight": reference_weight,
                "previous_common_objective": previous_score,
                "promoted_common_objective": candidate_score,
                "improvement_percent": 100.0
                * (previous_score - candidate_score)
                / previous_score,
            }
        )


def read_row(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return next(csv.DictReader(handle))


if __name__ == "__main__":
    main()
