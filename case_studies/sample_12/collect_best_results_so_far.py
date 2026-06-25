"""Collect the current best Sample#12 fit artifacts into one folder."""

from __future__ import annotations

import csv
from pathlib import Path
import shutil
import sys

CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
FIT_DIR = CASE_DIR / "jax_gradient_fit"
RUN_DIR = REPO_ROOT / "runs" / "sample_12" / "jax_least_squares" / "single_run"
BEST_DIR = CASE_DIR / "best_results_so_far"
SUMMARY_PATH = RUN_DIR / "summary.csv"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(FIT_DIR) not in sys.path:
    sys.path.insert(0, str(FIT_DIR))

from swanx.stack import plot_vertical_concentration_profiles
from swanx.result_exports import (  # noqa: E402
    save_fit_curve_data_csv,
    save_optimized_stack_csv,
)

import fit_sample12_joint_cap3_jax_gradient as fit12  # noqa: E402


ARTIFACTS_TO_COPY = {
    "best_fit.png": "sample12_trf_least_squares_best_fit.png",
    "stack_schematic.png": "sample12_trf_least_squares_stack_schematic.png",
    "superlattice_profile.png": "sample12_trf_least_squares_superlattice_profile.png",
    "convergence.png": "sample12_trf_least_squares_convergence.png",
    "summary.csv": "sample12_trf_least_squares_summary.csv",
    "summary.txt": "sample12_trf_least_squares_summary.txt",
    "numpy_validation_contributions.csv": "sample12_trf_least_squares_validation_numpy_contributions.csv",
    "jax_contributions.csv": "sample12_trf_least_squares_jax_contributions.csv",
    "history.csv": "sample12_trf_least_squares_history.csv",
    "covariance.csv": "sample12_trf_least_squares_covariance.csv",
    "parameter_uncertainties.csv": "sample12_trf_least_squares_parameter_uncertainties.csv",
}
CONCENTRATION_PREFIX = "sample12_trf_top30_vertical_concentration_profiles"


def main() -> None:
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    fit12.patch_archived_context_paths()
    values = load_best_values(SUMMARY_PATH)
    data = fit12.joint.sample12.load_and_prepare_data(
        background_percent=10.0,
        background_order=2,
    )
    data = fit12.joint.sample12.apply_reflectivity_window(
        data,
        fit12.joint.sample12.RC_START_DEG,
        fit12.joint.sample12.RC_STOP_DEG,
    )
    summary = read_summary(SUMMARY_PATH)
    reflectivity_weight = float(summary["reflectivity_weight"])
    problem = fit12.joint.make_problem(data, reflectivity_weight)
    simulation = problem.simulate(values)

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
    profiles = plot_vertical_concentration_profiles(
        BEST_DIR / f"{CONCENTRATION_PREFIX}.png",
        simulation.stack,
        concentration_by_layer(simulation.stack),
        max_depth=30.0,
        step=0.2,
        title="Sample#12 top 30 A concentration profiles",
        layer_labels=layer_labels(simulation.stack),
        show_layer_shading=False,
        layer_box_style=True,
        categorical_strips=True,
    )
    save_concentration_profiles_csv(
        BEST_DIR / f"{CONCENTRATION_PREFIX}.csv",
        profiles.depth,
        profiles.profiles,
    )
    for source_name, target_name in ARTIFACTS_TO_COPY.items():
        source = RUN_DIR / source_name
        if source.exists():
            shutil.copy2(source, BEST_DIR / target_name)
    print(f"Collected best Sample#12 results in {BEST_DIR}")
    print("Wrote best_fit_experiment_and_simulation.csv")
    print("Wrote optimized_stack_layers.csv")
    print(f"Wrote {CONCENTRATION_PREFIX}.png/csv")


def load_best_values(path: Path) -> dict[str, float]:
    row = read_summary(path)
    return {
        parameter.name: float(row[parameter.name])
        for parameter in fit12.joint.PARAMETERS
    }


def read_summary(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return next(csv.DictReader(handle))


def concentration_by_layer(stack) -> dict[str, list[float]]:
    c = []
    la = []
    ni = []
    for material in stack.materials:
        c.append(1.0 if material == "C" else 0.0)
        la.append(1.0 if material == "LNO" else 0.0)
        ni.append(1.0 if material == "LNO" else 0.0)
    return {"C": c, "La": la, "Ni": ni}


def layer_labels(stack) -> dict[int, str]:
    labels = {
        1: "C",
        2: "LNO-1",
        3: "LNO-2",
        4: "LNO-bottom",
    }
    first_superlattice = 5
    for index in range(first_superlattice, min(len(stack.layers) - 1, first_superlattice + 8)):
        repeat = (index - first_superlattice) // 2 + 1
        labels[index] = f"{stack.layers[index].material} r{repeat}"
    return labels


def save_concentration_profiles_csv(path: Path, depth, profiles: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["depth_A", *profiles])
        for row_index, z in enumerate(depth):
            writer.writerow([z, *[values[row_index] for values in profiles.values()]])


if __name__ == "__main__":
    main()
