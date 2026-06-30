"""Build existing SWANX objects from a ProjectSpec."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
from swanx.preprocessing import normalize_rocking_curve
from swanx.stack import SimulationStack
from swanx.stack.slicing import LayerSlicingPolicy, fixed_layer_grid_plan
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
    offpeak_mask = _offpeak_mask_from_settings(spec, reflectivity_data, rocking_curve_data)
    problem = None
    if reflectivity_data is not None or rocking_curve_data:
        parameters = tuple(
            FitParameter(
                name=parameter.name,
                lower=parameter.lower,
                upper=parameter.upper,
                initial=parameter.initial,
            )
            for parameter in spec.varying_parameters()
        )
        problem = FittingProblem(
            parameters=parameters,
            stack_builder=lambda trial_values: _build_stack(
                spec, {**spec.default_parameter_values(), **trial_values}, tables
            ),
            photon_energy_ev=spec.photon_energy_ev,
            reflectivity=reflectivity_data,
            rocking_curves=rocking_curve_data,
            core_levels=core_levels,
            field_step=float(spec.settings.get("field_step", 1.0)),
            roughness_step=float(spec.settings.get("roughness_step", 1.0)),
            roughness_profile=str(spec.settings.get("roughness_profile", "erf")),
            polarization=project_polarization(str(spec.settings.get("polarization", "s"))),
            slicing=_slicing_from_settings(spec, tables),
            offpeak_mask=offpeak_mask,
            angle_offset_parameter=_optional_string(spec.settings.get("angle_offset_parameter", "angle_offset")),
            reflectivity_angle_offset_parameter=_optional_string(
                spec.settings.get("reflectivity_angle_offset_parameter")
            ),
            rocking_curve_angle_offset_parameter=_optional_string(
                spec.settings.get("rocking_curve_angle_offset_parameter")
            ),
            rocking_curve_normalization=str(spec.settings.get("normalization", "mean")),
            normalization_edge_fraction=float(spec.settings.get("normalization_edge_fraction", 0.10)),
            normalization_polynomial_order=int(spec.settings.get("normalization_polynomial_order", 2)),
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
        indices = resolve_emitting_layer_indices(spec, raw["emit_from"])
        concentration = float(raw.get("concentration", 1.0))
        candidate_indices = indices if indices is not None else tuple(range(len(stack.layers)))
        materials = {
            stack.layers[index].material: concentration
            for index in candidate_indices
            if stack.layers[index].material.lower() != "vacuum"
        }
        extra_imfp = _core_extra_imfp(raw, tables, spec.photon_energy_ev, float(raw["binding_energy_ev"]))
        cores.append(
            core_level_from_tables(
                name=name,
                binding_energy_ev=float(raw["binding_energy_ev"]),
                photon_energy_ev=spec.photon_energy_ev,
                concentration_by_material=materials,
                imfp_tables=tables.imfp,
                emission_angle_deg=float(raw.get("emission_angle_deg", 0.0)),
                emitting_layer_indices=indices,
                extra_imfp_by_material=extra_imfp,
            )
        )
    return tuple(cores)


def _core_extra_imfp(
    raw: dict[str, Any],
    tables: MaterialTables,
    photon_energy_ev: float,
    binding_energy_ev: float,
) -> dict[str, float] | None:
    source = raw.get("vacuum_imfp_from_material")
    if source is None:
        return None
    material = str(source)
    if material not in tables.imfp:
        raise ProjectValidationError(
            f"core level {raw.get('name', '<unnamed>')!r} requested "
            f"vacuum_imfp_from_material={material!r}, but that material has no IMFP table"
        )
    kinetic_energy_ev = float(photon_energy_ev) - float(binding_energy_ev)
    return {"vacuum": tables.imfp[material].at_kinetic_energy(kinetic_energy_ev)}


def resolve_emitting_layer_indices(
    spec: ProjectSpec,
    emit_from: dict[str, Any],
) -> tuple[int, ...] | None:
    if emit_from.get("all", False):
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
        reflectivity = replace(
            reflectivity,
            weight=float(fields.get("weight", reflectivity.weight)),
            log_floor=float(fields.get("log_floor", reflectivity.log_floor)),
        )
    rocking = []
    for fields in spec.datasets.get("rocking_curves", ()) or ():
        curve = read_rocking_curve_data(
            _resolve(spec, fields["path"]),
            name=fields.get("name"),
            angle_column=fields.get("angle_column", "angle_deg"),
            intensity_column=fields.get("intensity_column", "intensity"),
            sigma_column=fields.get("sigma_column"),
        )
        rocking.append(replace(curve, weight=float(fields.get("weight", curve.weight))))
    rocking_tuple = tuple(rocking)
    offpeak_mask = _offpeak_mask_from_settings(spec, reflectivity, rocking_tuple)
    return reflectivity, _normalize_rocking_datasets(spec, rocking_tuple, offpeak_mask)


def _normalize_rocking_datasets(
    spec: ProjectSpec,
    rocking: tuple[RockingCurveData, ...],
    offpeak_mask: np.ndarray | None,
) -> tuple[RockingCurveData, ...]:
    normalized = []
    for fields, curve in zip(spec.datasets.get("rocking_curves", ()) or (), rocking):
        mode = fields.get("normalization", spec.settings.get("normalization"))
        if mode is None:
            normalized.append(curve)
            continue
        values, _ = normalize_rocking_curve(
            curve.angles,
            curve.intensity,
            mode=mode,
            offpeak_mask=offpeak_mask,
            edge_fraction=float(spec.settings.get("normalization_edge_fraction", 0.10)),
            polynomial_order=int(spec.settings.get("normalization_polynomial_order", 2)),
        )
        normalized.append(replace(curve, intensity=values))
    return tuple(normalized)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _slicing_from_settings(spec: ProjectSpec, tables: MaterialTables):
    raw = spec.settings.get("slicing")
    if raw is None:
        return LayerSlicingPolicy()
    if isinstance(raw, str):
        if raw in {"adaptive", "unified"}:
            return LayerSlicingPolicy()
        if raw == "legacy":
            return None
        raise ProjectValidationError("settings.slicing must be 'adaptive', 'unified', 'legacy', or a mapping")
    if not isinstance(raw, dict):
        raise ProjectValidationError("settings.slicing must be 'adaptive', 'unified', 'legacy', or a mapping")
    mode = str(raw.get("mode", "adaptive"))
    if mode in {"adaptive", "unified"}:
        return LayerSlicingPolicy(
            min_slices=int(raw.get("min_slices", 3)),
            max_slice_thickness=float(raw.get("max_slice_thickness_A", raw.get("max_slice_thickness", 1.0))),
        )
    if mode in {"legacy", "none"}:
        return None
    if mode not in {"fixed", "fixed_grid"}:
        raise ProjectValidationError("settings.slicing.mode must be adaptive, unified, legacy, fixed, or fixed_grid")
    reference_values = spec.default_parameter_values()
    reference_values.update({name: float(value) for name, value in (raw.get("reference_values", {}) or {}).items()})
    capacity_stack = _build_stack(spec, reference_values, tables)
    policy = LayerSlicingPolicy(
        min_slices=int(raw.get("min_slices", 3)),
        max_slice_thickness=float(raw.get("max_slice_thickness_A", raw.get("max_slice_thickness", 1.0))),
    )
    return fixed_layer_grid_plan(capacity_stack.optical_layers, policy)


def _offpeak_mask_from_settings(
    spec: ProjectSpec,
    reflectivity: ReflectivityData | None,
    rocking: tuple[RockingCurveData, ...],
) -> np.ndarray | None:
    raw = spec.settings.get("rocking_curve_offpeak_mask", spec.settings.get("offpeak_mask"))
    if raw is None or not rocking:
        return None
    if not isinstance(raw, dict):
        raise ProjectValidationError("settings.rocking_curve_offpeak_mask must be a mapping")
    mode = str(raw.get("mode", "exclude_reflectivity_peak"))
    if mode != "exclude_reflectivity_peak":
        raise ProjectValidationError("settings.rocking_curve_offpeak_mask.mode must be 'exclude_reflectivity_peak'")
    if reflectivity is None:
        raise ProjectValidationError("settings.rocking_curve_offpeak_mask requires a reflectivity dataset")
    half_width = float(raw.get("half_width_deg", 1.25))
    if half_width <= 0:
        raise ProjectValidationError("settings.rocking_curve_offpeak_mask.half_width_deg must be positive")
    peak_angle = float(reflectivity.angles[int(np.argmax(reflectivity.reflectivity))])
    return np.abs(np.asarray(rocking[0].angles, dtype=float) - peak_angle) > half_width


def _resolve(spec: ProjectSpec, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else spec.root_dir / path

