"""CSV and JSON report outputs for YAML project runs."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..builder import BuiltProject
from ..yaml_io import write_yaml
from ._shared import _sigma_or_empty, _write_csv, _write_json


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
