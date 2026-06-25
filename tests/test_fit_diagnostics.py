import numpy as np

from swanx.diagnostics import (
    save_fit_history_csv,
    save_staged_fit_summary_csv,
)
from swanx.fitting import (
    BayesianOptimizationResult,
    FitContribution,
    FitEvaluation,
    FitHistory,
    FitParameter,
    FitStage,
    StageFitResult,
    StageRunResult,
    StagedFitResult,
)


def test_save_fit_history_csv_writes_objective_contributions_and_parameters(tmp_path):
    history = FitHistory(
        (
            FitEvaluation(
                parameters={"x": 1.0},
                objective=2.5,
                contributions=(
                    FitContribution("A", raw=1.0, weight=2.0),
                    FitContribution("B", raw=0.5, weight=1.0),
                ),
                timings={"rocking_curve_simulation_seconds": 0.25},
            ),
        )
    )
    path = tmp_path / "history.csv"

    save_fit_history_csv(path, history, (FitParameter("x", 0.0, 2.0),))

    data = np.genfromtxt(path, delimiter=",", names=True)
    assert data["evaluation"] == 1.0
    assert data["objective"] == 2.5
    assert data["A_raw"] == 1.0
    assert data["A_weighted"] == 2.0
    assert data["B_raw"] == 0.5
    assert data["B_weighted"] == 0.5
    assert data["rocking_curve_simulation_seconds"] == 0.25
    assert data["x"] == 1.0


def test_save_staged_fit_summary_csv_writes_each_start(tmp_path):
    evaluation = FitEvaluation(
        parameters={"x": 1.0},
        objective=0.5,
        contributions=(FitContribution("synthetic", raw=0.5, weight=1.0),),
    )
    bo_result = BayesianOptimizationResult(
        best_parameters={"x": 1.0},
        best_objective=0.5,
        best_evaluation=evaluation,
        history=FitHistory((evaluation,)),
        raw_result=object(),
    )
    run = StageRunResult("stage", 0, 7, bo_result)
    stage = FitStage("stage", (FitParameter("x", 0.0, 2.0),))
    staged = StagedFitResult(
        stages=(StageFitResult(stage, (run,), run, {"x": 1.0}),),
        best_parameters={"x": 1.0},
    )
    path = tmp_path / "staged.csv"

    save_staged_fit_summary_csv(path, staged, stage.parameters)

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    assert data["stage"] == "stage"
    assert data["start_index"] == 0
    assert data["random_state"] == 7
    assert data["objective"] == 0.5
    assert data["is_stage_best"] == 1.0
    assert data["x"] == 1.0
