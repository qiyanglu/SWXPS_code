"""Report writers for YAML project runs."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .builder import BuiltProject
from .yaml_io import write_yaml


def prepare_output_dir(spec) -> Path:
    if spec.project.get("output_dir"):
        output = Path(str(spec.project["output_dir"]))
        if not output.is_absolute():
            output = spec.root_dir / output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = spec.root_dir / "runs" / f"{spec.name}_{timestamp}"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("input", "resolved", "simulation", "fit"):
        (output / name).mkdir(exist_ok=True)
    return output


def write_input_files(output: Path, built: BuiltProject, timestamp: str) -> None:
    shutil.copyfile(built.spec.path, output / "input" / "project_original.yaml")
    write_yaml(
        output / "input" / "project_resolved.yaml",
        built.spec.to_resolved_mapping(built.values),
    )
    _write_json(
        output / "input" / "run_metadata.json",
        {
            "project_name": built.spec.name,
            "timestamp": timestamp,
            "source": str(built.spec.path),
            "roughness_convention": (
                "roughness_A on layer j means roughness/interdiffusion at the upper "
                "interface of layer j, i.e. interface between layer j-1 and layer j."
            ),
            "repeat_index_convention": "repeat_index is 1-based inside repeat blocks.",
        },
    )


def write_resolved_files(output: Path, built: BuiltProject) -> None:
    layers = built.spec.layer_specs_for_values(built.values)
    _write_csv(output / "resolved" / "stack_resolved.csv", [
        ["layer_index", "id", "material", "tags", "thickness_A", "roughness_A"],
        *[
            [
                index,
                layer["id"],
                layer["material"],
                ";".join(layer["tags"]),
                layer["thickness"],
                layer["roughness"],
            ]
            for index, layer in enumerate(layers)
        ],
    ])
    _write_csv(output / "resolved" / "materials_resolved.csv", [
        ["material", "has_optical_constants", "has_imfp"],
        *[
            [
                material,
                material in built.material_tables.optical_constants,
                material in built.material_tables.imfp,
            ]
            for material in sorted(built.spec.materials)
        ],
    ])
    _write_csv(output / "resolved" / "core_levels_resolved.csv", [
        ["name", "binding_energy_ev", "emission_angle_deg", "emitting_layer_indices"],
        *[
            [
                core.name,
                core.binding_energy_ev,
                core.emission_angle_deg,
                "" if core.emitting_layer_indices is None else ";".join(map(str, core.emitting_layer_indices)),
            ]
            for core in built.core_levels
        ],
    ])
    _write_csv(output / "resolved" / "parameters_resolved.csv", [
        ["name", "value", "vary", "initial", "lower", "upper", "resolved_value"],
        *[
            [
                name,
                parameter.value,
                parameter.vary,
                "" if parameter.initial is None else parameter.initial,
                "" if parameter.lower is None else parameter.lower,
                "" if parameter.upper is None else parameter.upper,
                built.values.get(name, parameter.value),
            ]
            for name, parameter in built.spec.parameters.items()
        ],
    ])
    dataset_rows = [["type", "name", "points"]]
    if built.reflectivity_data is not None:
        dataset_rows.append(["reflectivity", built.reflectivity_data.name, len(built.reflectivity_data.angles)])
    for data in built.rocking_curve_data:
        dataset_rows.append(["rocking_curve", data.name, len(data.angles)])
    _write_csv(output / "resolved" / "datasets_resolved.csv", dataset_rows)


def write_simulation_files(output: Path, simulation) -> None:
    if simulation.reflectivity is not None:
        _write_csv(output / "simulation" / "reflectivity_simulated.csv", [
            ["angle_deg", "calculation_angle_deg", "reflectivity"],
            *[
                [angle, calc, value]
                for angle, calc, value in zip(
                    simulation.reflectivity.angle,
                    simulation.reflectivity.calculation_angle,
                    simulation.reflectivity.reflectivity,
                )
            ],
        ])
    if simulation.rocking_curves is not None:
        rows = [["core_level", "angle_deg", "calculation_angle_deg", "intensity", "raw_intensity"]]
        for core in simulation.rocking_curves.core_levels:
            for angle, calc, intensity, raw in zip(
                simulation.rocking_curves.angle,
                simulation.rocking_curves.calculation_angle,
                core.curve.intensity,
                core.curve.raw_intensity,
            ):
                rows.append([core.name, angle, calc, intensity, raw])
        _write_csv(output / "simulation" / "rocking_curves_simulated.csv", rows)


def write_experimental_files(output: Path, built: BuiltProject) -> None:
    if built.reflectivity_data is not None or built.rocking_curve_data:
        (output / "data").mkdir(exist_ok=True)
    if built.reflectivity_data is not None:
        rows = [["angle_deg", "reflectivity", "sigma"]]
        sigma = _sigma_or_empty(built.reflectivity_data.sigma, len(built.reflectivity_data.angles))
        rows.extend(
            [angle, value, uncertainty]
            for angle, value, uncertainty in zip(
                built.reflectivity_data.angles,
                built.reflectivity_data.reflectivity,
                sigma,
            )
        )
        _write_csv(output / "data" / "reflectivity_experimental.csv", rows)
    if built.rocking_curve_data:
        rows = [["name", "angle_deg", "intensity", "sigma"]]
        for data in built.rocking_curve_data:
            sigma = _sigma_or_empty(data.sigma, len(data.angles))
            rows.extend(
                [data.name, angle, value, uncertainty]
                for angle, value, uncertainty in zip(data.angles, data.intensity, sigma)
            )
        _write_csv(output / "data" / "rocking_curves_experimental.csv", rows)


def write_fit_files(output: Path, built: BuiltProject, simulation, evaluation, result: Any) -> None:
    if built.spec.fit_method == "simulate_only":
        _write_residuals(output, built, simulation)
        return
    if evaluation is not None:
        _write_csv(output / "fit" / "fit_contributions.csv", [
            ["name", "raw", "weight", "weighted"],
            *[
                [contribution.name, contribution.raw, contribution.weight, contribution.weighted]
                for contribution in evaluation.contributions
            ],
        ])
    best = dict(built.values)
    if result is not None and hasattr(result, "best_parameters"):
        best.update({name: float(value) for name, value in result.best_parameters.items()})
    _write_csv(output / "fit" / "best_parameters.csv", [
        ["name", "initial", "lower", "upper", "best_value"],
        *[
            [
                parameter.name,
                parameter.initial,
                parameter.lower,
                parameter.upper,
                best.get(parameter.name, parameter.value),
            ]
            for parameter in built.spec.varying_parameters()
        ],
    ])
    _write_residuals(output, built, simulation)


def write_fit_summary(
    output: Path,
    built: BuiltProject,
    *,
    timestamp: str,
    result: Any = None,
    evaluation: Any = None,
) -> None:
    final_objective = None if evaluation is None else evaluation.objective
    for attr in ("final_cost", "best_loss", "best_objective"):
        if final_objective is None and result is not None and hasattr(result, attr):
            final_objective = getattr(result, attr)
    _write_json(
        output / "fit" / "fit_summary.json",
        {
            "project_name": built.spec.name,
            "timestamp": timestamp,
            "fit_method": built.spec.fit_method,
            "photon_energy_ev": built.spec.photon_energy_ev,
            "polarization": built.spec.settings.get("polarization", "s"),
            "layer_count": len(built.spec.stack),
            "core_level_count": len(built.core_levels),
            "dataset_count": (1 if built.reflectivity_data is not None else 0) + len(built.rocking_curve_data),
            "final_objective": final_objective,
            "backend_status": None if result is None else getattr(result, "status", None),
            "backend_message": None if result is None else getattr(result, "message", None),
        },
    )


def write_method_outputs(output: Path, method: str, result: Any, built: BuiltProject | None = None) -> list[str]:
    if result is None:
        return []
    notes: list[str] = []
    if method == "jax_least_squares":
        _write_least_squares_outputs(output / "optimizer" / "least_squares", result, built)
        notes.extend(_write_least_squares_plot_outputs(output / "plots", result, built))
    elif method == "jax_gradient":
        _write_gradient_outputs(output / "optimizer" / "gradient", result)
    elif method == "bayesian_optimization":
        _write_bayesian_outputs(output / "optimizer" / "bayesian", result)
    return notes


def write_plots(output: Path, built: BuiltProject, simulation) -> list[str]:
    expected = (
        "plots/fit_overview.png",
        "plots/reflectivity_fit.png",
        "plots/rocking_curves_fit.png",
    )
    if not built.spec.report.get("save_plots", False):
        return [f"{name} skipped because report.save_plots is false" for name in expected]
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return [f"{name} skipped because matplotlib is not installed" for name in expected]

    notes: list[str] = []
    (output / "plots").mkdir(exist_ok=True)
    notes.append(_write_fit_overview_plot(output, built, simulation, plt))
    notes.append(_write_reflectivity_plot(output, built, simulation, plt))
    notes.append(_write_rocking_curve_plot(output, built, simulation, plt))
    return notes


def _write_fit_overview_plot(output: Path, built: BuiltProject, simulation, plt) -> str:
    if simulation.reflectivity is None and simulation.rocking_curves is None:
        return "plots/fit_overview.png skipped because no simulated curves are available"
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
            label="fit",
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
            ax.plot(
                simulation.rocking_curves.angle,
                simulated_rc[data.name],
                color="black",
                linewidth=1.45,
                label="fit",
            )
        ax.axhline(1.0, color="0.35", linestyle=":", linewidth=0.9, alpha=0.6)
        ax.set_ylabel(data.name)
        ax.legend(frameon=False, loc="best")
        _style_axis(ax)
        axis_index += 1
    axes[-1].set_xlabel("Incident angle (deg)")
    fig.savefig(output / "plots" / "fit_overview.png", dpi=220)
    plt.close(fig)
    if overlays:
        return "plots/fit_overview.png written with experimental overlays: " + ", ".join(overlays)
    return "plots/fit_overview.png written without experimental overlay because no matching datasets were provided"


def _write_reflectivity_plot(output: Path, built: BuiltProject, simulation, plt) -> str:
    if simulation.reflectivity is None:
        return "plots/reflectivity_fit.png skipped because no simulated reflectivity is available"
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
    fig.savefig(output / "plots" / "reflectivity_fit.png", dpi=220)
    plt.close(fig)
    if has_overlay:
        return "plots/reflectivity_fit.png written with experimental overlay"
    return "plots/reflectivity_fit.png written without experimental overlay because no reflectivity dataset was provided"


def _write_rocking_curve_plot(output: Path, built: BuiltProject, simulation, plt) -> str:
    if simulation.rocking_curves is None:
        return "plots/rocking_curves_fit.png skipped because no simulated rocking curves are available"
    fig, ax = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
    simulated_by_name = _simulated_rocking_curves(simulation)
    overlaid = []
    for data in built.rocking_curve_data:
        if data.name not in simulated_by_name:
            continue
        color = _plot_color(data.name)
        ax.plot(data.angles, data.intensity, "o", color=color, markersize=3.0, alpha=0.58, label=f"{data.name} data")
        ax.plot(simulation.rocking_curves.angle, simulated_by_name[data.name], color=color, linewidth=1.45, label=f"{data.name} fit")
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
    fig.savefig(output / "plots" / "rocking_curves_fit.png", dpi=220)
    plt.close(fig)
    if overlaid:
        return "plots/rocking_curves_fit.png written with experimental overlays: " + ", ".join(overlaid)
    return "plots/rocking_curves_fit.png written without experimental overlay because no matching rocking-curve dataset was provided"


def _write_least_squares_plot_outputs(directory: Path, result: Any, built: BuiltProject | None) -> list[str]:
    if built is None or built.fitting_problem is None:
        return []
    if getattr(result, "final_residuals", None) is None or getattr(result, "final_jacobian", None) is None:
        return ["plots/parameter_uncertainty.png skipped because least-squares residuals or Jacobian are unavailable", "plots/parameter_correlation.png skipped because least-squares residuals or Jacobian are unavailable"]
    try:
        from swanx.diagnostics import plot_correlation_matrix, plot_parameter_estimates
        import matplotlib.pyplot as plt
    except ImportError:
        return ["plots/parameter_uncertainty.png skipped because matplotlib is not installed", "plots/parameter_correlation.png skipped because matplotlib is not installed"]
    directory.mkdir(exist_ok=True)
    try:
        diagnostics = _least_squares_diagnostics_for_plots(result, built)
    except ValueError as error:
        return [
            f"plots/parameter_uncertainty.png skipped because least-squares diagnostics are unavailable: {error}",
            f"plots/parameter_correlation.png skipped because least-squares diagnostics are unavailable: {error}",
        ]
    notes = []
    uncertainty_figure, _ = plot_parameter_estimates(diagnostics)
    uncertainty_figure.savefig(directory / "parameter_uncertainty.png", dpi=200, bbox_inches="tight")
    plt.close(uncertainty_figure)
    notes.append("plots/parameter_uncertainty.png written from least-squares covariance diagnostics")
    correlation_figure, _ = plot_correlation_matrix(diagnostics)
    correlation_figure.savefig(directory / "parameter_correlation.png", dpi=200, bbox_inches="tight")
    plt.close(correlation_figure)
    notes.append("plots/parameter_correlation.png written from least-squares covariance diagnostics")
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


def write_markdown_report(
    output: Path,
    built: BuiltProject,
    *,
    timestamp: str,
    result: Any = None,
    evaluation: Any = None,
    skipped_outputs: list[str] | None = None,
) -> None:
    plot_notes = [] if skipped_outputs is None else list(skipped_outputs)
    skipped_notes = [item for item in plot_notes if " skipped " in f" {item} "]
    generated = sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "report.md"
    )
    generated.append("report.md")
    lines = [
        f"# SWANX Project Report: {built.spec.name}",
        "",
        f"- Project name: {built.spec.name}",
        f"- Timestamp: {timestamp}",
        f"- Fit method: {built.spec.fit_method}",
        f"- Photon energy: {built.spec.photon_energy_ev} eV",
        f"- Polarization: {built.spec.settings.get('polarization', 's')}",
        f"- Layers: {len(built.spec.stack)}",
        f"- Core levels: {len(built.core_levels)}",
        f"- Datasets: {(1 if built.reflectivity_data is not None else 0) + len(built.rocking_curve_data)}",
        "",
        "## Core Levels",
        "",
    ]
    if built.core_levels:
        for core in built.core_levels:
            indices = "all matching layers" if core.emitting_layer_indices is None else ", ".join(map(str, core.emitting_layer_indices))
            lines.append(f"- {core.name}: {core.binding_energy_ev} eV, emitting layer indices: {indices}")
    else:
        lines.append("- none")
    lines.extend(["", "## Datasets", ""])
    if built.reflectivity_data is not None:
        lines.append(f"- Reflectivity: {built.reflectivity_data.name} ({len(built.reflectivity_data.angles)} points)")
    for data in built.rocking_curve_data:
        lines.append(f"- Rocking curve: {data.name} ({len(data.angles)} points)")
    if built.reflectivity_data is None and not built.rocking_curve_data:
        lines.append("- none")
    lines.extend(["", "## Run Summary", ""])
    if built.spec.fit_method == "simulate_only":
        lines.append("No fitting was performed; this was a simulation-only run.")
        lines.extend(["", "Used parameter values:", ""])
        if built.spec.parameters:
            for name, parameter in built.spec.parameters.items():
                lines.append(f"- {name}: {built.values.get(name, parameter.value)}")
        else:
            lines.append("- none")
    else:
        objective = None if evaluation is None else evaluation.objective
        for attr in ("final_cost", "best_loss", "best_objective"):
            if objective is None and result is not None and hasattr(result, attr):
                objective = getattr(result, attr)
        lines.append(f"Final objective: {objective if objective is not None else 'not available'}")
        best = dict(built.values)
        if result is not None and hasattr(result, "best_parameters"):
            best.update({name: float(value) for name, value in result.best_parameters.items()})
        varying = built.spec.varying_parameters()
        if varying:
            lines.extend(["", "Best parameters:", ""])
            for parameter in varying:
                lines.append(f"- {parameter.name}: {best.get(parameter.name, parameter.value)}")
    lines.extend(["", "## Plot Notes", ""])
    if plot_notes:
        lines.extend(f"- {item}" for item in plot_notes)
    else:
        lines.append("- none")
    lines.extend(["", "## Output Files", ""])
    lines.extend(f"- `{item}`" for item in generated)
    lines.extend(["", "## Warnings / Skipped Optional Outputs", ""])
    if skipped_notes:
        lines.extend(f"- {item}" for item in skipped_notes)
    else:
        lines.append("- none")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def _write_least_squares_outputs(directory: Path, result: Any, built: BuiltProject | None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "status.json", _status_dict(result, objective_attr="final_cost"))
    _write_array(directory / "residual_vector.csv", getattr(result, "final_residuals", None), "residual")
    _write_array(directory / "jacobian.csv", getattr(result, "final_jacobian", None), "value")
    _write_array(directory / "covariance.csv", getattr(result, "covariance", None), "value")
    covariance = getattr(result, "covariance", None)
    if covariance is not None:
        covariance = np.asarray(covariance, dtype=float)
        sigma = np.sqrt(np.diag(covariance))
        denom = np.outer(sigma, sigma)
        correlation = np.divide(covariance, denom, out=np.zeros_like(covariance), where=denom != 0)
        _write_array(directory / "correlation.csv", correlation, "value")
        _write_csv(directory / "parameter_uncertainty.csv", _least_squares_uncertainty_rows(result, built, sigma))
    history = getattr(result, "history", ())
    _write_csv(directory / "convergence_history.csv", [
        ["iteration", "cost", "gradient_norm", "parameters_json"],
        *[
            [record.iteration, record.cost, record.gradient_norm, json.dumps(record.parameters, sort_keys=True)]
            for record in history
        ],
    ])
    active_bounds = [["parameter", "active_bound"]]
    if built is not None:
        best = getattr(result, "best_parameters", {})
        for parameter in built.spec.varying_parameters():
            value = float(best.get(parameter.name, parameter.value))
            bound = ""
            if parameter.lower is not None and np.isclose(value, parameter.lower):
                bound = "lower"
            elif parameter.upper is not None and np.isclose(value, parameter.upper):
                bound = "upper"
            active_bounds.append([parameter.name, bound])
    _write_csv(directory / "active_bounds.csv", active_bounds)


def _least_squares_uncertainty_rows(result: Any, built: BuiltProject | None, sigma: np.ndarray) -> list[list[Any]]:
    rows: list[list[Any]] = [["parameter", "best_value", "stderr", "ci95_low", "ci95_high", "lower", "upper"]]
    if built is None:
        for index, stderr in enumerate(sigma):
            best = np.nan
            rows.append([f"parameter_{index}", best, stderr, best, best, "", ""])
        return rows
    best_parameters = getattr(result, "best_parameters", {})
    for parameter, stderr in zip(built.spec.varying_parameters(), sigma):
        best_value = float(best_parameters.get(parameter.name, parameter.value))
        ci = 1.96 * float(stderr)
        rows.append([
            parameter.name,
            best_value,
            float(stderr),
            best_value - ci,
            best_value + ci,
            parameter.lower,
            parameter.upper,
        ])
    return rows


def _write_gradient_outputs(directory: Path, result: Any) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "status.json", _status_dict(result, objective_attr="best_loss"))
    history = getattr(result, "history", ())
    _write_csv(directory / "objective_history.csv", [
        ["iteration", "loss"],
        *[[record.iteration, record.loss] for record in history],
    ])
    _write_csv(directory / "parameter_history.csv", [
        ["iteration", "parameters_json"],
        *[[record.iteration, json.dumps(record.parameters, sort_keys=True)] for record in history],
    ])
    _write_csv(directory / "gradient_norm_history.csv", [
        ["iteration", "gradient_norm"],
        *[[record.iteration, record.gradient_norm] for record in history],
    ])
    _write_array(directory / "final_gradient.csv", getattr(result, "final_gradient", None), "gradient")


def _write_bayesian_outputs(directory: Path, result: Any) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "status.json", _status_dict(result, objective_attr="best_objective"))
    history = getattr(getattr(result, "history", None), "evaluations", ())
    best_objective = float("inf")
    best_parameters: dict[str, float] = {}
    evaluations = [["evaluation", "objective", "parameters_json"]]
    best_so_far = [["evaluation", "best_objective", "best_parameters_json"]]
    samples = [["evaluation", "parameters_json"]]
    for index, evaluation in enumerate(history, start=1):
        parameters_json = json.dumps(evaluation.parameters, sort_keys=True)
        objective = float(evaluation.objective)
        evaluations.append([index, objective, parameters_json])
        samples.append([index, parameters_json])
        if objective <= best_objective:
            best_objective = objective
            best_parameters = dict(evaluation.parameters)
        best_so_far.append([index, best_objective, json.dumps(best_parameters, sort_keys=True)])
    _write_csv(directory / "evaluations.csv", evaluations)
    _write_csv(directory / "best_so_far.csv", best_so_far)
    _write_csv(directory / "parameter_samples.csv", samples)
    _write_json(directory / "stage_summary.json", {"stages": []})


def _write_residuals(output: Path, built: BuiltProject, simulation) -> None:
    rows = _residual_rows(built, simulation)
    if len(rows) > 1:
        _write_csv(output / "fit" / "residuals.csv", rows)


def _residual_rows(built: BuiltProject, simulation) -> list[list[Any]]:
    rows: list[list[Any]] = [["dataset", "angle_deg", "experimental", "simulated", "residual"]]
    if built.reflectivity_data is not None and simulation.reflectivity is not None:
        for angle, observed, simulated in zip(
            built.reflectivity_data.angles,
            built.reflectivity_data.reflectivity,
            simulation.reflectivity.reflectivity,
        ):
            rows.append([built.reflectivity_data.name, angle, observed, simulated, observed - simulated])
    if built.rocking_curve_data and simulation.rocking_curves is not None:
        simulated_by_name = {
            core.name: core.curve.intensity for core in simulation.rocking_curves.core_levels
        }
        for data in built.rocking_curve_data:
            if data.name not in simulated_by_name:
                continue
            for angle, observed, simulated in zip(data.angles, data.intensity, simulated_by_name[data.name]):
                rows.append([data.name, angle, observed, simulated, observed - simulated])
    return rows


def _status_dict(result: Any, *, objective_attr: str) -> dict[str, Any]:
    return {
        "status": getattr(result, "status", None),
        "message": getattr(result, "message", None),
        "success": getattr(result, "success", None),
        "objective": getattr(result, objective_attr, None),
    }


def _write_array(path: Path, value: Any, column_name: str) -> None:
    if value is None:
        return
    array = np.asarray(value)
    if array.ndim == 1:
        _write_csv(path, [["index", column_name], *[[index, item] for index, item in enumerate(array)]])
    elif array.ndim == 2:
        _write_csv(path, [["row", "column", column_name], *[
            [row, column, array[row, column]]
            for row in range(array.shape[0])
            for column in range(array.shape[1])
        ]])


def _sigma_or_empty(sigma, count: int) -> list[Any]:
    return [""] * count if sigma is None else list(sigma)


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=_json_default)


def _write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
