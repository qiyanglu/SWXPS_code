"""Fit the synthetic C/LNO/STO dataset with Bayesian optimization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swxps import (
    BayesianOptimizationSettings,
    FitParameter,
    FittingProblem,
    ReflectivityData,
    RockingCurveData,
    plot_best_fit,
    plot_fit_convergence,
    plot_surrogate_slices,
    run_bayesian_fit,
    save_fit_history_csv,
)

from generate_lno_sto_c_synthetic_data import core_level_requests, c_lno_sto_template


PHOTON_ENERGY_EV = 1000.0
TRUE_VALUES = {
    "carbon_thickness": 10.0,
    "carbon_roughness": 2.0,
    "lno_thickness": 20.0,
    "sto_thickness": 20.0,
    "superlattice_roughness": 3.0,
    "substrate_roughness": 3.0,
    "angle_offset": 0.0,
}
PARAMETERS = (
    FitParameter("carbon_thickness", 6.0, 14.0, "Angstrom", initial=8.5),
    FitParameter("carbon_roughness", 0.5, 4.0, "Angstrom", initial=1.0),
    FitParameter("lno_thickness", 18.0, 22.0, "Angstrom", initial=19.0),
    FitParameter("sto_thickness", 18.0, 22.0, "Angstrom", initial=21.0),
    FitParameter("superlattice_roughness", 1.0, 5.0, "Angstrom", initial=2.0),
    FitParameter("substrate_roughness", 1.0, 5.0, "Angstrom", initial=4.0),
    FitParameter("angle_offset", -0.35, 0.35, "deg", initial=0.15),
)
DATASET_WEIGHTS = {
    "reflectivity": 1.0,
    "La 4d": 10.0,
    "O 1s": 10.0,
    "Ti 2p": 10.0,
    "C 1s": 10.0,
}
STACK_TEMPLATE = c_lno_sto_template(PHOTON_ENERGY_EV)


def load_synthetic_data(path: Path, stride: int) -> dict[str, np.ndarray]:
    """Load and optionally downsample the synthetic CSV."""

    data = np.genfromtxt(path, delimiter=",", names=True)
    if stride <= 0:
        raise ValueError("stride must be positive")
    return {name: np.asarray(data[name][::stride], dtype=float) for name in data.dtype.names}


def full_data(path: Path) -> dict[str, np.ndarray]:
    """Load the full synthetic CSV for final plotting."""

    data = np.genfromtxt(path, delimiter=",", names=True)
    return {name: np.asarray(data[name], dtype=float) for name in data.dtype.names}


def build_stack(values: dict[str, float]):
    """Build a C/LNO/STO stack from fitted parameter values."""

    return STACK_TEMPLATE.build(values)


def make_fit_problem(data: dict[str, np.ndarray]) -> FittingProblem:
    """Create a weighted reflectivity plus SW-XPS fitting problem."""

    angles = data["angle_deg"]
    peak_angle = angles[np.argmax(data["reflectivity"])]
    offpeak_mask = np.abs(angles - peak_angle) > 1.25
    reflectivity_data = ReflectivityData(
        name="reflectivity",
        angles=angles,
        reflectivity=data["reflectivity"],
        weight=DATASET_WEIGHTS["reflectivity"],
        log_floor=1.0e-12,
    )
    rc_data = (
        RockingCurveData("La 4d", angles, data["la4d_rc"], weight=DATASET_WEIGHTS["La 4d"]),
        RockingCurveData("O 1s", angles, data["o1s_rc"], weight=DATASET_WEIGHTS["O 1s"]),
        RockingCurveData("Ti 2p", angles, data["ti2p_rc"], weight=DATASET_WEIGHTS["Ti 2p"]),
        RockingCurveData("C 1s", angles, data["c1s_rc"], weight=DATASET_WEIGHTS["C 1s"]),
    )
    return FittingProblem(
        parameters=PARAMETERS,
        stack_builder=build_stack,
        photon_energy_ev=PHOTON_ENERGY_EV,
        reflectivity=reflectivity_data,
        rocking_curves=rc_data,
        core_levels=core_level_requests(PHOTON_ENERGY_EV),
        angle_offset_parameter="angle_offset",
        field_step=1.0,
        roughness_step=1.0,
        offpeak_mask=offpeak_mask,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-calls", type=int, default=12, help="Total BO calls.")
    parser.add_argument("--n-initial-points", type=int, default=5, help="Random initial BO points.")
    parser.add_argument("--stride", type=int, default=4, help="Use every Nth data point for fitting.")
    parser.add_argument("--random-state", type=int, default=3, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(__file__).resolve().parent
    data_path = output_dir / "lno_sto_c_synthetic_data.csv"
    fit_data = load_synthetic_data(data_path, stride=args.stride)
    problem = make_fit_problem(fit_data)
    result = run_bayesian_fit(
        problem,
        BayesianOptimizationSettings(
            n_calls=args.n_calls,
            n_initial_points=args.n_initial_points,
            random_state=args.random_state,
        ),
    )

    full = full_data(data_path)
    full_problem = make_fit_problem(full)
    best_simulation = full_problem.simulate(result.best_parameters)
    history_path = output_dir / "lno_sto_c_bo_history.csv"
    convergence_path = output_dir / "lno_sto_c_bo_convergence.png"
    best_fit_path = output_dir / "lno_sto_c_bo_best_fit.png"
    surrogate_path = output_dir / "lno_sto_c_bo_surrogate_slices.png"
    save_fit_history_csv(history_path, result.history, PARAMETERS)
    plot_fit_convergence(convergence_path, result.history)
    plot_best_fit(
        best_fit_path,
        full_problem.reflectivity,
        full_problem.rocking_curves,
        best_simulation,
    )
    plot_surrogate_slices(surrogate_path, result, PARAMETERS, TRUE_VALUES)

    print(f"Best objective: {result.best_objective:.6g}")
    best_vector = [[result.best_parameters[parameter.name] for parameter in PARAMETERS]]
    surrogate_mean, surrogate_std = result.predict_objective(best_vector)
    print(
        "GP surrogate at best point: "
        f"mean={surrogate_mean[0]:.6g}, std={surrogate_std[0]:.6g}"
    )
    print("Best parameters:")
    for parameter in PARAMETERS:
        value = result.best_parameters[parameter.name]
        true_value = TRUE_VALUES[parameter.name]
        unit = f" {parameter.unit}" if parameter.unit else ""
        print(f"  {parameter.name}: {value:.6g}{unit} (true {true_value:g}{unit})")
    print("Best contribution terms:")
    for contribution in result.best_evaluation.contributions:
        print(f"  {contribution.name}: raw={contribution.raw:.6g}, weighted={contribution.weighted:.6g}")
    print(f"Saved {history_path}")
    print(f"Saved {convergence_path}")
    print(f"Saved {best_fit_path}")
    print(f"Saved {surrogate_path}")


if __name__ == "__main__":
    main()
