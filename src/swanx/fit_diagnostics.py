"""Utilities for saving and visualizing fitting results."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .bo import BayesianOptimizationResult, StagedFitResult
from .fitting.core import (
    FitHistory,
    FitParameter,
    FitSimulation,
    ReflectivityData,
    RockingCurveData,
)


def save_fit_history_csv(
    path: str | Path,
    history: FitHistory,
    parameters: Sequence[FitParameter],
) -> None:
    """Save objective history, contribution terms, and parameter values."""

    contribution_names = sorted(
        {
            contribution.name
            for evaluation in history.evaluations
            for contribution in evaluation.contributions
        }
    )
    timing_names = sorted(
        {
            name
            for evaluation in history.evaluations
            for name in evaluation.timings
        }
    )
    columns = [
        "evaluation",
        "objective",
        *[f"{name}_raw" for name in contribution_names],
        *[f"{name}_weighted" for name in contribution_names],
        *timing_names,
        *[parameter.name for parameter in parameters],
    ]
    rows = []
    for index, evaluation in enumerate(history.evaluations, start=1):
        by_name = {
            contribution.name: contribution
            for contribution in evaluation.contributions
        }
        row = [index, evaluation.objective]
        row.extend(
            by_name[name].raw if name in by_name else np.nan
            for name in contribution_names
        )
        row.extend(
            by_name[name].weighted if name in by_name else np.nan
            for name in contribution_names
        )
        row.extend(evaluation.timings.get(name, np.nan) for name in timing_names)
        row.extend(evaluation.parameters[parameter.name] for parameter in parameters)
        rows.append(row)
    np.savetxt(
        path,
        np.asarray(rows, dtype=float),
        delimiter=",",
        header=",".join(columns),
        comments="",
    )


def save_staged_fit_summary_csv(
    path: str | Path,
    staged_result: StagedFitResult,
    parameters: Sequence[FitParameter],
) -> None:
    """Save one summary row per stage/start in a staged fit."""

    columns = [
        "stage",
        "start_index",
        "random_state",
        "objective",
        "is_stage_best",
        *[parameter.name for parameter in parameters],
    ]
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for stage_result in staged_result.stages:
            for run in stage_result.runs:
                is_best = run is stage_result.best_run
                row = [
                    stage_result.stage.name,
                    run.start_index,
                    np.nan if run.random_state is None else run.random_state,
                    run.result.best_objective,
                    1.0 if is_best else 0.0,
                ]
                row.extend(
                    run.result.best_parameters.get(parameter.name, np.nan)
                    for parameter in parameters
                )
                handle.write(",".join(str(value) for value in row) + "\n")


def plot_fit_convergence(
    path: str | Path,
    history: FitHistory,
) -> None:
    """Plot objective and best-so-far objective versus evaluation."""

    plt = _load_pyplot()
    objective = np.array(
        [evaluation.objective for evaluation in history.evaluations],
        dtype=float,
    )
    best_so_far = np.minimum.accumulate(objective)
    evaluations = np.arange(1, len(objective) + 1)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.semilogy(
        evaluations,
        objective,
        marker="o",
        linestyle="",
        alpha=0.45,
        label="evaluated",
    )
    ax.semilogy(
        evaluations,
        best_so_far,
        color="black",
        linewidth=1.8,
        label="best so far",
    )
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Objective")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_best_fit(
    path: str | Path,
    reflectivity_data: ReflectivityData | None,
    rocking_curve_data: Sequence[RockingCurveData],
    simulation: FitSimulation,
) -> None:
    """Plot measured data against one simulated fit result."""

    plt = _load_pyplot()
    n_panels = (1 if reflectivity_data is not None else 0) + len(rocking_curve_data)
    if n_panels == 0:
        raise ValueError("at least one dataset is required for a best-fit plot")

    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(7.6, max(3.0, 2.0 * n_panels)),
        sharex=True,
    )
    axes = np.asarray([axes]).ravel()
    axis_index = 0

    if reflectivity_data is not None:
        if simulation.reflectivity is None:
            raise ValueError("simulation does not include reflectivity")
        ax = axes[axis_index]
        ax.semilogy(
            reflectivity_data.angles,
            reflectivity_data.reflectivity,
            "o",
            markersize=3,
            label="data",
        )
        ax.semilogy(
            simulation.reflectivity.angle,
            simulation.reflectivity.reflectivity,
            color="black",
            linewidth=1.4,
            label="fit",
        )
        ax.set_ylabel(reflectivity_data.name)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        axis_index += 1

    simulated_curves = {}
    if rocking_curve_data:
        if simulation.rocking_curves is None:
            raise ValueError("simulation does not include rocking curves")
        simulated_curves = {
            core.name: core.curve.intensity
            for core in simulation.rocking_curves.core_levels
        }

    for data in rocking_curve_data:
        if data.name not in simulated_curves:
            raise ValueError(f"missing simulated curve {data.name!r}")
        ax = axes[axis_index]
        ax.plot(
            data.angles,
            data.intensity,
            "o",
            markersize=3,
            alpha=0.5,
            label="data",
        )
        ax.plot(
            data.angles,
            simulated_curves[data.name],
            color="black",
            linewidth=1.4,
            label="fit",
        )
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0)
        ax.set_ylabel(data.name)
        ax.grid(True, alpha=0.25)
        ax.legend()
        axis_index += 1

    axes[-1].set_xlabel("Incident angle (deg)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_surrogate_slices(
    path: str | Path,
    result: BayesianOptimizationResult,
    parameters: Sequence[FitParameter],
    reference_values: dict[str, float] | None = None,
) -> None:
    """Plot 1D GP surrogate mean/std slices through the best point."""

    plt = _load_pyplot()
    best_vector = np.array(
        [result.best_parameters[parameter.name] for parameter in parameters],
        dtype=float,
    )
    ncols = 2
    nrows = int(np.ceil(len(parameters) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.0, 2.8 * nrows))
    axes = np.asarray(axes).ravel()
    evaluated_values = {
        parameter.name: np.array(
            [
                evaluation.parameters[parameter.name]
                for evaluation in result.history.evaluations
            ],
            dtype=float,
        )
        for parameter in parameters
    }
    evaluated_objective = np.array(
        [evaluation.objective for evaluation in result.history.evaluations],
        dtype=float,
    )

    for index, parameter in enumerate(parameters):
        ax = axes[index]
        grid = np.linspace(parameter.lower, parameter.upper, 160)
        vectors = np.repeat(best_vector[None, :], len(grid), axis=0)
        vectors[:, index] = grid
        mean, std = result.predict_objective(vectors)

        ax.plot(grid, mean, color="black", linewidth=1.5, label="GP mean")
        ax.fill_between(
            grid,
            mean - std,
            mean + std,
            color="tab:blue",
            alpha=0.2,
            label="GP mean +/- 1 std",
        )
        ax.scatter(
            evaluated_values[parameter.name],
            evaluated_objective,
            s=12,
            color="tab:orange",
            alpha=0.45,
            label="evaluations",
        )
        ax.axvline(
            result.best_parameters[parameter.name],
            color="tab:red",
            linestyle="-",
            linewidth=1.0,
            label="BO best",
        )
        if reference_values is not None and parameter.name in reference_values:
            ax.axvline(
                reference_values[parameter.name],
                color="tab:green",
                linestyle="--",
                linewidth=1.0,
                label="reference",
            )
        unit = f" ({parameter.unit})" if parameter.unit else ""
        ax.set_xlabel(f"{parameter.name}{unit}")
        ax.set_ylabel("Objective")
        ax.grid(True, alpha=0.25)

    for ax in axes[len(parameters) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _load_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("matplotlib is required for fitting diagnostic plots") from error
    return plt
