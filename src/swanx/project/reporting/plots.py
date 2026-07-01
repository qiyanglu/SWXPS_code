"""Plot report outputs for YAML project runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..builder import BuiltProject


def write_plots(output: Path, built: BuiltProject, simulation) -> list[str]:
    names = _plot_names(built)
    expected = (*names.curve_plots, "plots/stack_schematic.png")
    if not built.spec.save_plots:
        return [f"{name} skipped because run.outputs.plots/report.save_plots is false" for name in expected]
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return [f"{name} skipped because matplotlib is not installed" for name in expected]

    notes: list[str] = []
    (output / "plots").mkdir(exist_ok=True)
    notes.append(_write_fit_overview_plot(output, built, simulation, plt, names))
    notes.append(_write_reflectivity_plot(output, built, simulation, plt, names))
    notes.append(_write_rocking_curve_plot(output, built, simulation, plt, names))
    notes.append(_write_stack_schematic_plot(output, built, simulation))
    return notes


class _PlotNames:
    def __init__(self, *, overview: str, reflectivity: str, rocking_curves: str, model_label: str) -> None:
        self.overview = overview
        self.reflectivity = reflectivity
        self.rocking_curves = rocking_curves
        self.model_label = model_label

    @property
    def curve_plots(self) -> tuple[str, str, str]:
        return (self.overview, self.reflectivity, self.rocking_curves)


def _plot_names(built: BuiltProject) -> _PlotNames:
    if built.spec.fit_method == "simulate_only":
        return _PlotNames(
            overview="plots/simulation_overview.png",
            reflectivity="plots/reflectivity_simulation.png",
            rocking_curves="plots/rocking_curves_simulation.png",
            model_label="simulation",
        )
    return _PlotNames(
        overview="plots/fit_overview.png",
        reflectivity="plots/reflectivity_fit.png",
        rocking_curves="plots/rocking_curves_fit.png",
        model_label="fit",
    )


def _write_fit_overview_plot(output: Path, built: BuiltProject, simulation, plt, names: _PlotNames) -> str:
    if simulation.reflectivity is None and simulation.rocking_curves is None:
        return f"{names.overview} skipped because no simulated curves are available"
    rocking_datasets = list(built.rocking_curve_data)
    simulated_rc = _simulated_rocking_curves(simulation)
    if not rocking_datasets and simulated_rc:
        rocking_datasets = [type("DatasetName", (), {"name": name, "angles": simulation.rocking_curves.angle, "intensity": None})() for name in simulated_rc]
    row_count = (1 if simulation.reflectivity is not None else 0) + max(len(rocking_datasets), 0)
    row_count = max(row_count, 1)
    fig, axes = plt.subplots(
        row_count,
        1,
        figsize=(8.2, max(3.4, 1.85 * row_count + 0.8)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).ravel()
    axis_index = 0
    overlays: list[str] = []
    if simulation.reflectivity is not None:
        ax = axes[axis_index]
        color = _plot_color("reflectivity")
        if built.reflectivity_data is not None:
            ax.semilogy(
                built.reflectivity_data.angles,
                built.reflectivity_data.reflectivity,
                "o",
                color=color,
                markersize=3.0,
                alpha=0.58,
                label="reflectivity data",
            )
            overlays.append("reflectivity")
        ax.semilogy(
            simulation.reflectivity.angle,
            simulation.reflectivity.reflectivity,
            color="tab:red",
            linewidth=1.55,
            label=names.model_label,
        )
        ax.set_ylabel("Reflectivity")
        ax.legend(frameon=False, loc="best")
        _style_axis(ax, semilog=True)
        axis_index += 1
    for data in rocking_datasets:
        ax = axes[axis_index]
        color = _plot_color(data.name)
        if getattr(data, "intensity", None) is not None:
            ax.plot(
                data.angles,
                data.intensity,
                "o",
                color=color,
                markersize=3.0,
                alpha=0.58,
                label=f"{data.name} data",
            )
            overlays.append(str(data.name))
        if data.name in simulated_rc:
            model_color = "black" if getattr(data, "intensity", None) is not None else color
            ax.plot(
                simulation.rocking_curves.angle,
                simulated_rc[data.name],
                color=model_color,
                linewidth=1.45,
                label=names.model_label,
            )
        ax.axhline(1.0, color="0.35", linestyle=":", linewidth=0.9, alpha=0.6)
        ax.set_ylabel(data.name)
        ax.legend(frameon=False, loc="best")
        _style_axis(ax)
        axis_index += 1
    axes[-1].set_xlabel("Incident angle (deg)")
    fig.savefig(output / names.overview, dpi=220)
    plt.close(fig)
    if overlays:
        return f"{names.overview} written with experimental overlays: " + ", ".join(overlays)
    return f"{names.overview} written without experimental overlay because no matching datasets were provided"

def _write_reflectivity_plot(output: Path, built: BuiltProject, simulation, plt, names: _PlotNames) -> str:
    if simulation.reflectivity is None:
        return f"{names.reflectivity} skipped because no simulated reflectivity is available"
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    has_overlay = built.reflectivity_data is not None
    if has_overlay:
        ax.semilogy(
            built.reflectivity_data.angles,
            built.reflectivity_data.reflectivity,
            "o",
            color=_plot_color("reflectivity"),
            markersize=3.0,
            alpha=0.58,
            label="experimental",
        )
    ax.semilogy(
        simulation.reflectivity.angle,
        simulation.reflectivity.reflectivity,
        color="tab:red",
        linewidth=1.6,
        label="simulated",
    )
    ax.set_xlabel("Incident angle (deg)")
    ax.set_ylabel("Reflectivity")
    ax.legend(frameon=False, loc="best")
    _style_axis(ax, semilog=True)
    fig.savefig(output / names.reflectivity, dpi=220)
    plt.close(fig)
    if has_overlay:
        return f"{names.reflectivity} written with experimental overlay"
    return f"{names.reflectivity} written without experimental overlay because no reflectivity dataset was provided"

def _write_rocking_curve_plot(output: Path, built: BuiltProject, simulation, plt, names: _PlotNames) -> str:
    if simulation.rocking_curves is None:
        return f"{names.rocking_curves} skipped because no simulated rocking curves are available"
    fig, ax = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
    simulated_by_name = _simulated_rocking_curves(simulation)
    overlaid = []
    for data in built.rocking_curve_data:
        if data.name not in simulated_by_name:
            continue
        color = _plot_color(data.name)
        ax.plot(data.angles, data.intensity, "o", color=color, markersize=3.0, alpha=0.58, label=f"{data.name} data")
        ax.plot(
            simulation.rocking_curves.angle,
            simulated_by_name[data.name],
            color=color,
            linewidth=1.45,
            label=f"{data.name} {names.model_label}",
        )
        overlaid.append(data.name)
    if not overlaid:
        for core in simulation.rocking_curves.core_levels:
            color = _plot_color(core.name)
            ax.plot(simulation.rocking_curves.angle, core.curve.intensity, color=color, linewidth=1.45, label=core.name)
    ax.axhline(1.0, color="0.35", linestyle=":", linewidth=0.9, alpha=0.6)
    ax.set_xlabel("Incident angle (deg)")
    ax.set_ylabel("Normalized intensity")
    ax.legend(frameon=False, loc="best", ncols=2 if len(simulated_by_name) > 2 else 1)
    _style_axis(ax)
    fig.savefig(output / names.rocking_curves, dpi=220)
    plt.close(fig)
    if overlaid:
        return f"{names.rocking_curves} written with experimental overlays: " + ", ".join(overlaid)
    return f"{names.rocking_curves} written without experimental overlay because no matching rocking-curve dataset was provided"

def _write_stack_schematic_plot(output: Path, built: BuiltProject, simulation) -> str:
    try:
        from swanx.diagnostics import plot_stack_schematic
    except ImportError:
        return "plots/stack_schematic.png skipped because matplotlib is not installed"
    try:
        plot_stack_schematic(
            output / "plots" / "stack_schematic.png",
            simulation.stack,
            title=f"{built.spec.name} stack",
            top_layers=4,
            bottom_layers=3,
        )
    except ValueError as error:
        return f"plots/stack_schematic.png skipped because stack schematic is unavailable: {error}"
    return "plots/stack_schematic.png written from the final stack"

def _write_least_squares_plot_outputs(directory: Path, result: Any, built: BuiltProject | None) -> list[str]:
    if built is None or built.fitting_problem is None:
        return []
    try:
        from swanx.diagnostics import plot_correlation_matrix, plot_parameter_estimates
        import matplotlib.pyplot as plt
    except ImportError:
        return [
            "plots/parameter_uncertainty.png skipped because matplotlib is not installed",
            "plots/parameter_correlation.png skipped because matplotlib is not installed",
            "plots/convergence.png skipped because matplotlib is not installed",
        ]
    directory.mkdir(exist_ok=True)
    notes: list[str] = []
    if getattr(result, "final_residuals", None) is None or getattr(result, "final_jacobian", None) is None:
        notes.extend([
            "plots/parameter_uncertainty.png skipped because least-squares residuals or Jacobian are unavailable",
            "plots/parameter_correlation.png skipped because least-squares residuals or Jacobian are unavailable",
        ])
    else:
        try:
            diagnostics = _least_squares_diagnostics_for_plots(result, built)
        except ValueError as error:
            notes.extend([
                f"plots/parameter_uncertainty.png skipped because least-squares diagnostics are unavailable: {error}",
                f"plots/parameter_correlation.png skipped because least-squares diagnostics are unavailable: {error}",
            ])
        else:
            uncertainty_figure, _ = plot_parameter_estimates(diagnostics)
            uncertainty_figure.savefig(directory / "parameter_uncertainty.png", dpi=200, bbox_inches="tight")
            plt.close(uncertainty_figure)
            notes.append("plots/parameter_uncertainty.png written from least-squares covariance diagnostics")
            correlation_figure, _ = plot_correlation_matrix(diagnostics)
            correlation_figure.savefig(directory / "parameter_correlation.png", dpi=200, bbox_inches="tight")
            plt.close(correlation_figure)
            notes.append("plots/parameter_correlation.png written from least-squares covariance diagnostics")
    notes.append(_write_least_squares_convergence_plot(directory, result, plt))
    return notes

def _write_least_squares_convergence_plot(directory: Path, result: Any, plt) -> str:
    history = tuple(getattr(result, "history", ()) or ())
    if not history:
        return "plots/convergence.png skipped because least-squares convergence history is unavailable"
    iterations = [getattr(record, "iteration", index) for index, record in enumerate(history, start=1)]
    costs = np.asarray([getattr(record, "cost", np.nan) for record in history], dtype=float)
    finite = np.isfinite(costs) & (costs > 0.0)
    if not np.any(finite):
        return "plots/convergence.png skipped because least-squares costs are unavailable"
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.semilogy(np.asarray(iterations)[finite], costs[finite], marker="o", markersize=3, linewidth=1.4)
    ax.set_xlabel("Jacobian evaluation")
    ax.set_ylabel("Least-squares cost")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(directory / "convergence.png", dpi=200)
    plt.close(fig)
    return "plots/convergence.png written from least-squares convergence history"

def _write_bayesian_plot_outputs(directory: Path, result: Any, built: BuiltProject | None) -> list[str]:
    notes: list[str] = []
    try:
        from swanx.diagnostics import plot_fit_convergence, plot_surrogate_slices
    except ImportError:
        return [
            "plots/convergence.png skipped because matplotlib is not installed",
            "plots/surrogate_slices.png skipped because matplotlib is not installed",
        ]
    directory.mkdir(exist_ok=True)
    history = getattr(result, "history", None)
    if history is None or not getattr(history, "evaluations", ()):
        notes.append("plots/convergence.png skipped because Bayesian optimization history is unavailable")
    else:
        plot_fit_convergence(directory / "convergence.png", history)
        notes.append("plots/convergence.png written from Bayesian optimization history")
    if built is None or built.fitting_problem is None:
        notes.append("plots/surrogate_slices.png skipped because fitting parameter declarations are unavailable")
    elif not hasattr(result, "predict_objective"):
        notes.append("plots/surrogate_slices.png skipped because Bayesian surrogate prediction is unavailable")
    else:
        try:
            plot_surrogate_slices(
                directory / "surrogate_slices.png",
                result,
                built.fitting_problem.parameters,
            )
        except Exception as error:  # optional diagnostic plot; keep the report run alive
            notes.append(f"plots/surrogate_slices.png skipped because Bayesian surrogate slices are unavailable: {error}")
        else:
            notes.append("plots/surrogate_slices.png written from the Bayesian surrogate model")
    return notes

def _least_squares_diagnostics_for_plots(result: Any, built: BuiltProject):
    from swanx.diagnostics import ParameterDiagnostics

    if built.fitting_problem is None:
        raise ValueError("no fitting problem was provided")
    parameters = built.fitting_problem.parameters
    if not parameters:
        raise ValueError("no varying parameters were provided")
    names = tuple(parameter.name for parameter in parameters)
    values = np.asarray([result.best_parameters[name] for name in names], dtype=float)
    bounds = tuple((float(parameter.lower), float(parameter.upper)) for parameter in parameters)
    residuals = np.asarray(getattr(result, "final_residuals", ()), dtype=float)
    jacobian = np.asarray(getattr(result, "final_jacobian", np.empty((0, len(parameters)))), dtype=float)
    covariance = getattr(result, "covariance", None)
    if covariance is None:
        if residuals.ndim != 1 or jacobian.ndim != 2 or jacobian.shape[1] != len(parameters):
            raise ValueError("residuals and Jacobian do not match the parameter vector")
        dof = residuals.size - len(parameters)
        if dof <= 0:
            raise ValueError("not enough residual degrees of freedom")
        residual_variance = float(np.dot(residuals, residuals) / dof)
        covariance = residual_variance * np.linalg.pinv(jacobian.T @ jacobian, rcond=1.0e-12)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (len(parameters), len(parameters)):
        raise ValueError("covariance shape does not match the parameter vector")
    covariance = 0.5 * (covariance + covariance.T)
    diagonal = np.diag(covariance)
    stderr = np.sqrt(np.where(diagonal >= 0.0, diagonal, np.nan))
    denominator = np.outer(stderr, stderr)
    correlation = np.full_like(covariance, np.nan, dtype=float)
    np.divide(covariance, denominator, out=correlation, where=denominator != 0.0)
    finite = np.isfinite(correlation)
    correlation[finite] = np.clip(correlation[finite], -1.0, 1.0)
    singular_values = np.linalg.svd(jacobian, compute_uv=False) if jacobian.ndim == 2 and jacobian.size else np.array([], dtype=float)
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0
        else float("inf")
    )
    dof = int(residuals.size - len(parameters)) if residuals.ndim == 1 else 0
    residual_variance = float(np.dot(residuals, residuals) / dof) if dof > 0 else float("nan")
    return ParameterDiagnostics(
        names=names,
        values=values,
        bounds=bounds,
        residuals=residuals,
        jacobian=jacobian,
        covariance=covariance,
        stderr=stderr,
        correlation=correlation,
        singular_values=singular_values,
        condition_number=condition_number,
        dof=dof,
        residual_variance=residual_variance,
    )

def _simulated_rocking_curves(simulation) -> dict[str, np.ndarray]:
    if simulation.rocking_curves is None:
        return {}
    return {core.name: core.curve.intensity for core in simulation.rocking_curves.core_levels}

def _plot_color(name: str) -> str:
    colors = {
        "reflectivity": "black",
        "La 4d": "tab:purple",
        "O 1s": "tab:green",
        "Ti 2p": "tab:orange",
        "C 1s": "tab:brown",
    }
    fallback = ("tab:blue", "tab:cyan", "tab:pink", "tab:olive", "tab:gray")
    if name in colors:
        return colors[name]
    return fallback[abs(hash(name)) % len(fallback)]

def _style_axis(ax, *, semilog: bool = False) -> None:
    ax.grid(True, which="both" if semilog else "major", alpha=0.25, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=10)
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(11)
