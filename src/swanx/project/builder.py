"""Build existing SWANX objects from a ProjectSpec."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from swanx.fitting import FitParameter, FittingProblem, ReflectivityData, RockingCurveData
from swanx.io import (
    MaterialTables,
    core_level_from_tables,
    load_material_tables,
    read_reflectivity_data,
    read_rocking_curve_data,
    stack_from_layer_specs,
)
from swanx.stack import SimulationStack
from swanx.workflows.simulate import CoreLevelRequest

from .spec import ProjectSpec, ProjectValidationError


@dataclass(frozen=True)
class BuiltProject:
    spec: ProjectSpec
    values: dict[str, float]
    material_tables: MaterialTables
    stack: SimulationStack
    core_levels: tuple[CoreLevelRequest, ...]
    reflectivity_data: ReflectivityData | None
    rocking_curve_data: tuple[RockingCurveData, ...]
    fitting_problem: FittingProblem | None


def build_project(spec: ProjectSpec, values: dict[str, float] | None = None) -> BuiltProject:
    values = spec.default_parameter_values() if values is None else dict(values)
    tables = _load_tables(spec)
    stack = _build_stack(spec, values, tables)
    core_levels = _build_core_levels(spec, stack, tables)
    reflectivity_data, rocking_curve_data = _read_datasets(spec)
    problem = None
    if reflectivity_data is not None or rocking_curve_data:
        parameters = tuple(
            FitParameter(
                name=parameter.name,
                lower=parameter.lower,
                upper=parameter.upper,
                initial=parameter.initial,
            )
            for parameter in spec.parameters.values()
        )
        problem = FittingProblem(
            parameters=parameters,
            stack_builder=lambda trial_values: _build_stack(spec, trial_values, tables),
            photon_energy_ev=spec.photon_energy_ev,
            reflectivity=reflectivity_data,
            rocking_curves=rocking_curve_data,
            core_levels=core_levels,
            field_step=float(spec.settings.get("field_step", 1.0)),
            roughness_step=float(spec.settings.get("roughness_step", 1.0)),
            roughness_profile=str(spec.settings.get("roughness_profile", "erf")),
            polarization=project_polarization(str(spec.settings.get("polarization", "s"))),
            rocking_curve_normalization=str(spec.settings.get("normalization", "mean")),
            simulation_backend=str(spec.settings.get("simulation_backend", "numpy")),
        )
    return BuiltProject(
        spec=spec,
        values=values,
        material_tables=tables,
        stack=stack,
        core_levels=core_levels,
        reflectivity_data=reflectivity_data,
        rocking_curve_data=rocking_curve_data,
        fitting_problem=problem,
    )


def project_polarization(value: str) -> str | dict[str, float]:
    if value == "unpolarized":
        return {"s": 0.5, "p": 0.5}
    if value in {"s", "p"}:
        return value
    raise ProjectValidationError("polarization must be 's', 'p', or 'unpolarized'")


def angles_from_settings(spec: ProjectSpec) -> np.ndarray:
    if "angles_deg" in spec.settings:
        return np.asarray(spec.settings["angles_deg"], dtype=float)
    try:
        start = float(spec.settings["angle_start_deg"])
        stop = float(spec.settings["angle_stop_deg"])
        count = int(spec.settings["angle_count"])
    except KeyError as error:
        raise ProjectValidationError(
            "simulate_only without datasets requires settings.angles_deg or "
            "settings.angle_start_deg/angle_stop_deg/angle_count"
        ) from error
    return np.linspace(start, stop, count)


def _load_tables(spec: ProjectSpec) -> MaterialTables:
    opc_files: dict[str, Path] = {}
    imfp_files: dict[str, Path] = {}
    for material, fields in spec.materials.items():
        if "opc_file" in fields:
            opc_files[material] = _resolve(spec, fields["opc_file"])
        if "imfp_file" in fields:
            imfp_files[material] = _resolve(spec, fields["imfp_file"])
    return load_material_tables(opc_files=opc_files, imfp_files=imfp_files)


def _build_stack(
    spec: ProjectSpec,
    values: dict[str, float],
    tables: MaterialTables,
) -> SimulationStack:
    return stack_from_layer_specs(
        spec.layer_specs_for_values(values),
        optical_constants=tables.optical_constants,
        energy_ev=spec.photon_energy_ev,
    )


def _build_core_levels(
    spec: ProjectSpec,
    stack: SimulationStack,
    tables: MaterialTables,
) -> tuple[CoreLevelRequest, ...]:
    cores = []
    for raw in spec.core_levels:
        name = str(raw["name"])
        indices = resolve_emitting_layer_indices(spec, raw.get("emit_from", {}))
        concentration = float(raw.get("concentration", 1.0))
        candidate_indices = indices if indices is not None else tuple(range(len(stack.layers)))
        materials = {
            stack.layers[index].material: concentration
            for index in candidate_indices
            if stack.layers[index].material.lower() != "vacuum"
        }
        cores.append(
            core_level_from_tables(
                name=name,
                binding_energy_ev=float(raw["binding_energy_ev"]),
                photon_energy_ev=spec.photon_energy_ev,
                concentration_by_material=materials,
                imfp_tables=tables.imfp,
                emission_angle_deg=float(raw.get("emission_angle_deg", 0.0)),
                emitting_layer_indices=indices,
            )
        )
    return tuple(cores)


def resolve_emitting_layer_indices(
    spec: ProjectSpec,
    emit_from: dict[str, Any],
) -> tuple[int, ...] | None:
    if not emit_from:
        return None
    selected: list[int] = []
    layer_ids = set(emit_from.get("layer_ids", ()) or ())
    tags = set(emit_from.get("tags", ()) or ())
    for layer in spec.stack:
        if layer.id in layer_ids or tags.intersection(layer.tags):
            selected.append(layer.layer_index)
    if not selected:
        raise ProjectValidationError("core-level emit_from selector did not match any layers")
    return tuple(selected)


def _read_datasets(spec: ProjectSpec) -> tuple[ReflectivityData | None, tuple[RockingCurveData, ...]]:
    reflectivity = None
    if spec.datasets.get("reflectivity"):
        fields = spec.datasets["reflectivity"]
        reflectivity = read_reflectivity_data(
            _resolve(spec, fields["path"]),
            name=fields.get("name"),
            angle_column=fields.get("angle_column", "angle_deg"),
            intensity_column=fields.get("intensity_column", "reflectivity"),
            sigma_column=fields.get("sigma_column"),
        )
    rocking = []
    for fields in spec.datasets.get("rocking_curves", ()) or ():
        rocking.append(
            read_rocking_curve_data(
                _resolve(spec, fields["path"]),
                name=fields.get("name"),
                angle_column=fields.get("angle_column", "angle_deg"),
                intensity_column=fields.get("intensity_column", "intensity"),
                sigma_column=fields.get("sigma_column"),
                normalization_mode=fields.get("normalization", spec.settings.get("normalization")),
            )
        )
    return reflectivity, tuple(rocking)


def _resolve(spec: ProjectSpec, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else spec.root_dir / path
