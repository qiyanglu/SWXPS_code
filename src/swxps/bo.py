"""Bayesian-optimization backend using scikit-optimize."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Literal

import numpy as np

from .fitting import (
    FitEvaluation,
    FitHistory,
    FitParameter,
    FittingProblem,
    JointObjective,
    initial_vector,
)


@dataclass(frozen=True)
class BayesianOptimizationSettings:
    """Settings for the first Gaussian-process BO backend."""

    n_calls: int = 40
    n_initial_points: int = 10
    acquisition_function: Literal["EI", "LCB", "PI"] = "EI"
    random_state: int | None = None
    use_initial: bool = True
    show_progress: bool = False
    progress_interval: int = 1

    def __post_init__(self) -> None:
        if self.n_calls <= 0:
            raise ValueError("n_calls must be positive")
        if self.n_initial_points <= 0:
            raise ValueError("n_initial_points must be positive")
        if self.n_initial_points > self.n_calls:
            raise ValueError("n_initial_points cannot exceed n_calls")
        if self.progress_interval <= 0:
            raise ValueError("progress_interval must be positive")


@dataclass(frozen=True)
class OptimizationTiming:
    """Wall-clock timing summary for one optimizer run."""

    total_seconds: float
    objective_seconds: float
    optimizer_overhead_seconds: float
    evaluations: int


@dataclass(frozen=True)
class BayesianOptimizationResult:
    """Package-native summary of a Bayesian-optimization run."""

    best_parameters: dict[str, float]
    best_objective: float
    best_evaluation: FitEvaluation
    history: FitHistory
    raw_result: object
    timing: OptimizationTiming = OptimizationTiming(0.0, 0.0, 0.0, 0)

    def predict_objective(
        self,
        vectors: list[list[float]] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return GP surrogate objective mean and standard deviation.

        These are uncertainties of the surrogate objective model, not posterior
        uncertainties of the physical fitting parameters.
        """

        models = getattr(self.raw_result, "models", None)
        space = getattr(self.raw_result, "space", None)
        if not models or space is None:
            raise ValueError("raw optimizer result does not contain a fitted surrogate model")
        transformed = space.transform(np.asarray(vectors, dtype=float))
        mean, std = models[-1].predict(transformed, return_std=True)
        return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)


@dataclass(frozen=True)
class FitStage:
    """One fitting stage with a selected subset of active parameters."""

    name: str
    parameters: tuple[FitParameter, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("fit stage name must be non-empty")
        if not self.parameters:
            raise ValueError("fit stage requires at least one parameter")


@dataclass(frozen=True)
class StageRunResult:
    """One multi-start run inside one fitting stage."""

    stage_name: str
    start_index: int
    random_state: int | None
    result: BayesianOptimizationResult


@dataclass(frozen=True)
class StageFitResult:
    """All multi-start results for one fitting stage."""

    stage: FitStage
    runs: tuple[StageRunResult, ...]
    best_run: StageRunResult
    parameters_after_stage: dict[str, float]


@dataclass(frozen=True)
class StagedFitResult:
    """Result of a staged multi-start fitting workflow."""

    stages: tuple[StageFitResult, ...]
    best_parameters: dict[str, float]

    @property
    def best_objective(self) -> float:
        """Return the objective of the final stage's best run."""

        return self.stages[-1].best_run.result.best_objective


def run_bayesian_optimization(
    objective: JointObjective,
    settings: BayesianOptimizationSettings | None = None,
) -> BayesianOptimizationResult:
    """Minimize a joint fitting objective with `scikit-optimize`."""

    settings = BayesianOptimizationSettings() if settings is None else settings
    gp_minimize, real_dimension = _load_skopt()
    dimensions = [
        _as_real_dimension(parameter, real_dimension)
        for parameter in objective.parameters
    ]

    func = _TimedObjective(
        objective,
        show_progress=settings.show_progress,
        interval=settings.progress_interval,
    )
    start = perf_counter()
    raw_result = gp_minimize(
        func=func,
        dimensions=dimensions,
        x0=[initial_vector(objective.parameters)] if settings.use_initial else None,
        n_calls=settings.n_calls,
        n_initial_points=settings.n_initial_points,
        acq_func=settings.acquisition_function,
        random_state=settings.random_state,
    )
    total_seconds = perf_counter() - start

    best = objective.history.best
    if best is None:
        raise RuntimeError("Bayesian optimization finished without evaluations")
    return BayesianOptimizationResult(
        best_parameters=best.parameters,
        best_objective=best.objective,
        best_evaluation=best,
        history=objective.history,
        raw_result=raw_result,
        timing=OptimizationTiming(
            total_seconds=total_seconds,
            objective_seconds=func.objective_seconds,
            optimizer_overhead_seconds=max(0.0, total_seconds - func.objective_seconds),
            evaluations=len(objective.history.evaluations),
        ),
    )


class _TimedObjective:
    """Timing and optional progress wrapper around a fitting objective."""

    def __init__(
        self,
        objective: JointObjective,
        show_progress: bool,
        interval: int,
    ) -> None:
        self.objective = objective
        self.show_progress = show_progress
        self.interval = interval
        self.best = np.inf
        self.objective_seconds = 0.0

    def __call__(self, vector: list[float] | np.ndarray) -> float:
        start = perf_counter()
        value = float(self.objective(vector))
        elapsed = perf_counter() - start
        self.objective_seconds += elapsed
        self.best = min(self.best, value)
        count = len(self.objective.history.evaluations)
        if self.show_progress and (count == 1 or count % self.interval == 0):
            print(
                "BO eval "
                f"{count}: objective={value:.6g}, best={self.best:.6g}, "
                f"eval_time={elapsed:.3f}s",
                flush=True,
            )
        return value


def run_bayesian_fit(
    problem: FittingProblem,
    settings: BayesianOptimizationSettings | None = None,
) -> BayesianOptimizationResult:
    """Run Bayesian optimization for a high-level fitting problem."""

    return run_bayesian_optimization(problem.objective(), settings)


def run_staged_multistart_bayesian_fit(
    problem: FittingProblem,
    stages: tuple[FitStage, ...],
    settings: BayesianOptimizationSettings,
    n_starts: int,
    initial_values: dict[str, float] | None = None,
    random_seed: int | None = None,
) -> StagedFitResult:
    """Run staged fitting, using multi-start BO within each stage.

    Parameters not active in a stage are held fixed at the best values from
    previous stages or from `initial_values` / `FitParameter.initial`.
    """

    if n_starts <= 0:
        raise ValueError("n_starts must be positive")
    if not stages:
        raise ValueError("at least one fitting stage is required")

    current_values = _default_parameter_values(problem.parameters)
    if initial_values is not None:
        current_values.update({name: float(value) for name, value in initial_values.items()})

    stage_results: list[StageFitResult] = []
    for stage_index, stage in enumerate(stages):
        runs: list[StageRunResult] = []
        for start_index in range(n_starts):
            seed = _stage_seed(random_seed, stage_index, start_index)
            stage_settings = replace(settings, random_state=seed)
            stage_problem = _stage_problem(problem, stage.parameters, current_values)
            result = run_bayesian_fit(stage_problem, stage_settings)
            runs.append(
                StageRunResult(
                    stage_name=stage.name,
                    start_index=start_index,
                    random_state=seed,
                    result=result,
                )
            )

        best_run = min(runs, key=lambda run: run.result.best_objective)
        current_values.update(best_run.result.best_parameters)
        stage_results.append(
            StageFitResult(
                stage=stage,
                runs=tuple(runs),
                best_run=best_run,
                parameters_after_stage=dict(current_values),
            )
        )

    return StagedFitResult(
        stages=tuple(stage_results),
        best_parameters=dict(current_values),
    )


def _load_skopt():
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except ImportError as error:
        raise ImportError(
            "scikit-optimize is required for Bayesian optimization; "
            "install the fitting extra or install scikit-optimize directly"
        ) from error
    return gp_minimize, Real


def _as_real_dimension(parameter: FitParameter, real_dimension):
    return real_dimension(
        low=parameter.lower,
        high=parameter.upper,
        name=parameter.name,
    )


def _default_parameter_values(parameters: tuple[FitParameter, ...]) -> dict[str, float]:
    return {
        parameter.name: (
            float(parameter.initial)
            if parameter.initial is not None
            else 0.5 * (parameter.lower + parameter.upper)
        )
        for parameter in parameters
    }


def _stage_problem(
    problem: FittingProblem,
    active_parameters: tuple[FitParameter, ...],
    fixed_values: dict[str, float],
) -> FittingProblem:
    staged_parameters = tuple(
        replace(parameter, initial=fixed_values.get(parameter.name, parameter.initial))
        if parameter.name in fixed_values
        else parameter
        for parameter in active_parameters
    )

    return replace(
        problem,
        parameters=staged_parameters,
        fixed_values=dict(fixed_values),
    )


def _stage_seed(
    random_seed: int | None,
    stage_index: int,
    start_index: int,
) -> int | None:
    if random_seed is None:
        return None
    return random_seed + 1000 * stage_index + start_index
