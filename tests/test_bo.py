from dataclasses import dataclass

import numpy as np

from swxps import (
    BayesianOptimizationResult,
    BayesianOptimizationSettings,
    FitContribution,
    FitParameter,
    FitStage,
    FittingProblem,
    JointObjective,
    ReflectivityData,
    SimulationStack,
    StackLayer,
    evaluation_from_contributions,
    run_bayesian_optimization,
    run_staged_multistart_bayesian_fit,
)


@dataclass(frozen=True)
class FakeRawResult:
    x: list[float]
    fun: float


class FakeReal:
    def __init__(self, low, high, name):
        self.low = low
        self.high = high
        self.name = name


class FakeSpace:
    def transform(self, vectors):
        return vectors


class FakeModel:
    def predict(self, vectors, return_std):
        assert return_std is True
        return np.sum(vectors, axis=1), np.full(len(vectors), 0.25)


def test_bayesian_optimization_adapter_returns_native_result(monkeypatch):
    from swxps import bo

    def fake_gp_minimize(
        func,
        dimensions,
        x0,
        n_calls,
        n_initial_points,
        acq_func,
        random_state,
    ):
        assert [dimension.name for dimension in dimensions] == ["x"]
        assert x0 == [[0.25]]
        assert n_calls == 2
        assert n_initial_points == 1
        assert acq_func == "EI"
        first = x0[0]
        second = [0.0]
        func(first)
        best = func(second)
        return FakeRawResult(x=second, fun=best)

    monkeypatch.setattr(bo, "_load_skopt", lambda: (fake_gp_minimize, FakeReal))

    parameters = (FitParameter("x", -1.0, 1.0, initial=0.25),)

    def evaluate(values):
        contribution = FitContribution("synthetic", raw=values["x"] ** 2, weight=1.0)
        return evaluation_from_contributions(values, (contribution,))

    objective = JointObjective(parameters, evaluate)
    result = run_bayesian_optimization(
        objective,
        BayesianOptimizationSettings(n_calls=2, n_initial_points=1, random_state=1),
    )

    assert result.best_parameters == {"x": 0.0}
    assert result.best_objective == 0.0
    assert len(result.history.evaluations) == 2
    assert result.raw_result == FakeRawResult(x=[0.0], fun=0.0)


def test_bayesian_optimization_result_predicts_surrogate_mean_and_std():
    evaluation = evaluation_from_contributions(
        {"x": 0.0},
        (FitContribution("synthetic", raw=0.0, weight=1.0),),
    )
    result = BayesianOptimizationResult(
        best_parameters={"x": 0.0},
        best_objective=0.0,
        best_evaluation=evaluation,
        history=JointObjective((FitParameter("x", -1.0, 1.0),), lambda _: evaluation).history,
        raw_result=type(
            "RawResult",
            (),
            {"models": [FakeModel()], "space": FakeSpace()},
        )(),
    )

    mean, std = result.predict_objective([[1.0], [2.0]])

    np.testing.assert_allclose(mean, [1.0, 2.0])
    np.testing.assert_allclose(std, [0.25, 0.25])


def test_staged_multistart_fit_carries_best_parameters_forward(monkeypatch):
    from swxps import bo

    calls = []

    def fake_run_bayesian_fit(problem, settings):
        calls.append(
            (
                tuple(parameter.name for parameter in problem.parameters),
                dict(problem.fixed_values),
                settings.random_state,
            )
        )
        active_name = problem.parameters[0].name
        best_value = 1.0 if active_name == "x" else 2.0
        objective = 10.0 - best_value
        evaluation = evaluation_from_contributions(
            {active_name: best_value},
            (FitContribution("synthetic", raw=objective, weight=1.0),),
        )
        return BayesianOptimizationResult(
            best_parameters={active_name: best_value},
            best_objective=objective,
            best_evaluation=evaluation,
            history=JointObjective((problem.parameters[0],), lambda _: evaluation).history.append(evaluation),
            raw_result=FakeRawResult(x=[best_value], fun=objective),
        )

    monkeypatch.setattr(bo, "run_bayesian_fit", fake_run_bayesian_fit)

    parameters = (
        FitParameter("x", 0.0, 3.0, initial=0.5),
        FitParameter("y", 0.0, 3.0, initial=0.75),
        FitParameter("angle_offset", -1.0, 1.0, initial=0.1),
    )
    problem = FittingProblem(
        parameters=parameters,
        stack_builder=lambda values: SimulationStack(
            (
                StackLayer("vacuum", 0.0),
                StackLayer("film", values["x"] + values["y"]),
                StackLayer("substrate", 0.0),
            )
        ),
        photon_energy_ev=1000.0,
        reflectivity=ReflectivityData(
            "reflectivity",
            np.array([1.0]),
            np.array([1.0]),
        ),
    )
    stages = (
        FitStage("first", (parameters[0],)),
        FitStage("second", (parameters[1],)),
    )

    result = run_staged_multistart_bayesian_fit(
        problem,
        stages,
        BayesianOptimizationSettings(n_calls=2, n_initial_points=1),
        n_starts=2,
        random_seed=5,
    )

    assert result.best_parameters["x"] == 1.0
    assert result.best_parameters["y"] == 2.0
    assert result.best_parameters["angle_offset"] == 0.1
    assert calls[0] == (("x",), {"x": 0.5, "y": 0.75, "angle_offset": 0.1}, 5)
    assert calls[2] == (("y",), {"x": 1.0, "y": 0.75, "angle_offset": 0.1}, 1005)
