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
        output = Path("runs") / f"{spec.name}_{timestamp}"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("input", "resolved", "simulation", "data", "fit", "plots", "optimizer"):
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
        ["name", "initial", "lower", "upper", "resolved_value"],
        *[
            [
                name,
                parameter.initial,
                parameter.lower,
                parameter.upper,
                built.values.get(name, parameter.initial),
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
    if evaluation is not None:
        _write_csv(output / "fit" / "fit_contributions.csv", [
            ["name", "raw", "weight", "weighted"],
            *[
                [contribution.name, contribution.raw, contribution.weight, contribution.weighted]
                for contribution in evaluation.contributions
            ],
        ])
    best = getattr(result, "best_parameters", built.values)
    _write_csv(output / "fit" / "best_parameters.csv", [
        ["name", "value"],
        *[[name, value] for name, value in sorted(best.items())],
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


def write_method_outputs(output: Path, method: str, result: Any) -> None:
    if result is None:
        return
    if method == "jax_least_squares":
        _write_least_squares_outputs(output / "optimizer" / "least_squares", result)
    elif method == "jax_gradient":
        _write_gradient_outputs(output / "optimizer" / "gradient", result)
    elif method == "bayesian_optimization":
        _write_bayesian_outputs(output / "optimizer" / "bayesian", result)


def write_plots(output: Path, built: BuiltProject, simulation) -> None:
    if not built.spec.report.get("save_plots", False):
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if simulation.reflectivity is not None:
        fig, ax = plt.subplots()
        ax.semilogy(simulation.reflectivity.angle, simulation.reflectivity.reflectivity, label="simulated")
        if built.reflectivity_data is not None:
            ax.semilogy(built.reflectivity_data.angles, built.reflectivity_data.reflectivity, "o", label="experimental")
        ax.set_xlabel("Grazing angle (deg)")
        ax.set_ylabel("Reflectivity")
        ax.legend()
        fig.savefig(output / "plots" / "reflectivity_fit.png", dpi=150)
        plt.close(fig)
    if simulation.rocking_curves is not None:
        fig, ax = plt.subplots()
        for core in simulation.rocking_curves.core_levels:
            ax.plot(simulation.rocking_curves.angle, core.curve.intensity, label=core.name)
        ax.set_xlabel("Grazing angle (deg)")
        ax.set_ylabel("Normalized intensity")
        ax.legend()
        fig.savefig(output / "plots" / "rocking_curves_fit.png", dpi=150)
        plt.close(fig)


def _write_least_squares_outputs(directory: Path, result: Any) -> None:
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
        _write_array(directory / "parameter_uncertainty.csv", sigma, "uncertainty")
    history = getattr(result, "history", ())
    _write_csv(directory / "convergence_history.csv", [
        ["iteration", "cost", "gradient_norm", "parameters_json"],
        *[
            [record.iteration, record.cost, record.gradient_norm, json.dumps(record.parameters, sort_keys=True)]
            for record in history
        ],
    ])
    _write_csv(directory / "active_bounds.csv", [["parameter", "active_bound"]])


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
    best = float("inf")
    rows = [["evaluation", "objective", "best_so_far", "parameters_json"]]
    samples = [["evaluation", "parameters_json"]]
    for index, evaluation in enumerate(history, start=1):
        best = min(best, float(evaluation.objective))
        rows.append([index, evaluation.objective, best, json.dumps(evaluation.parameters, sort_keys=True)])
        samples.append([index, json.dumps(evaluation.parameters, sort_keys=True)])
    _write_csv(directory / "evaluations.csv", rows)
    _write_csv(directory / "best_so_far.csv", rows)
    _write_csv(directory / "parameter_samples.csv", samples)
    _write_json(directory / "stage_summary.json", {"stages": []})


def _write_residuals(output: Path, built: BuiltProject, simulation) -> None:
    rows = [["dataset", "angle_deg", "experimental", "simulated", "residual"]]
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
    if len(rows) > 1:
        _write_csv(output / "fit" / "residuals.csv", rows)


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
