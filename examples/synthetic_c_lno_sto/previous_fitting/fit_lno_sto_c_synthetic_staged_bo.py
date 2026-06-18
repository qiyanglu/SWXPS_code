"""Run staged multi-start BO on the synthetic C/LNO/STO dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swxps import (
    BayesianOptimizationSettings,
    FitStage,
    plot_best_fit,
    plot_fit_convergence,
    run_staged_multistart_bayesian_fit,
    save_fit_history_csv,
    save_staged_fit_summary_csv,
)

from fit_lno_sto_c_synthetic_bo import (
    PARAMETERS,
    TRUE_VALUES,
    full_data,
    load_synthetic_data,
    make_fit_problem,
)


PARAMETER_BY_NAME = {parameter.name: parameter for parameter in PARAMETERS}
STAGES = (
    FitStage(
        "period_and_angle",
        (
            PARAMETER_BY_NAME["lno_thickness"],
            PARAMETER_BY_NAME["sto_thickness"],
            PARAMETER_BY_NAME["angle_offset"],
        ),
    ),
    FitStage(
        "surface_carbon",
        (
            PARAMETER_BY_NAME["carbon_thickness"],
            PARAMETER_BY_NAME["carbon_roughness"],
            PARAMETER_BY_NAME["angle_offset"],
        ),
    ),
    FitStage(
        "roughness",
        (
            PARAMETER_BY_NAME["superlattice_roughness"],
            PARAMETER_BY_NAME["substrate_roughness"],
            PARAMETER_BY_NAME["angle_offset"],
        ),
    ),
    FitStage("final_all", PARAMETERS),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-calls", type=int, default=12, help="BO calls per start/stage.")
    parser.add_argument("--n-initial-points", type=int, default=5, help="Random points per start/stage.")
    parser.add_argument("--n-starts", type=int, default=2, help="Independent starts per stage.")
    parser.add_argument("--stride", type=int, default=8, help="Use every Nth data point for fitting.")
    parser.add_argument("--random-seed", type=int, default=11, help="Base random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(__file__).resolve().parent
    data_path = output_dir / "lno_sto_c_synthetic_data.csv"
    fit_data = load_synthetic_data(data_path, stride=args.stride)
    problem = make_fit_problem(fit_data)

    staged_result = run_staged_multistart_bayesian_fit(
        problem,
        STAGES,
        BayesianOptimizationSettings(
            n_calls=args.n_calls,
            n_initial_points=args.n_initial_points,
        ),
        n_starts=args.n_starts,
        random_seed=args.random_seed,
    )

    summary_path = output_dir / "lno_sto_c_staged_bo_summary.csv"
    save_staged_fit_summary_csv(summary_path, staged_result, PARAMETERS)

    final_stage = staged_result.stages[-1]
    final_result = final_stage.best_run.result
    full = full_data(data_path)
    full_problem = make_fit_problem(full)
    best_simulation = full_problem.simulate(staged_result.best_parameters)
    history_path = output_dir / "lno_sto_c_staged_bo_final_history.csv"
    convergence_path = output_dir / "lno_sto_c_staged_bo_final_convergence.png"
    best_fit_path = output_dir / "lno_sto_c_staged_bo_best_fit.png"
    save_fit_history_csv(history_path, final_result.history, PARAMETERS)
    plot_fit_convergence(convergence_path, final_result.history)
    plot_best_fit(
        best_fit_path,
        full_problem.reflectivity,
        full_problem.rocking_curves,
        best_simulation,
    )

    print("Stage best objectives:")
    for stage_result in staged_result.stages:
        print(
            f"  {stage_result.stage.name}: "
            f"{stage_result.best_run.result.best_objective:.6g}"
        )
    print(f"Final objective: {staged_result.best_objective:.6g}")
    print("Final parameters:")
    for parameter in PARAMETERS:
        value = staged_result.best_parameters[parameter.name]
        true_value = TRUE_VALUES[parameter.name]
        unit = f" {parameter.unit}" if parameter.unit else ""
        print(f"  {parameter.name}: {value:.6g}{unit} (true {true_value:g}{unit})")
    print(f"Saved {summary_path}")
    print(f"Saved {history_path}")
    print(f"Saved {convergence_path}")
    print(f"Saved {best_fit_path}")


if __name__ == "__main__":
    main()
