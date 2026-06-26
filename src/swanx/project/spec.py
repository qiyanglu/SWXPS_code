"""ProjectSpec v1 data model and validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .expressions import ExpressionError, evaluate_number, names_in_expression
from .yaml_io import read_yaml


class ProjectValidationError(ValueError):
    """Raised when a YAML ProjectSpec is invalid."""


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    initial: float
    lower: float
    upper: float


@dataclass(frozen=True)
class ResolvedLayerSpec:
    id: str
    material: str
    tags: tuple[str, ...]
    thickness_expr: Any
    roughness_expr: Any
    layer_index: int
    repeat_index: int | None = None


@dataclass(frozen=True)
class ProjectSpec:
    """Validated YAML ProjectSpec v1.

    ``roughness_A`` on layer j means roughness/interdiffusion at the upper
    interface of layer j, i.e. interface between layer j-1 and layer j.
    """

    path: Path
    raw: dict[str, Any]
    project: dict[str, Any]
    settings: dict[str, Any]
    materials: dict[str, Any]
    parameters: dict[str, ParameterSpec]
    stack: tuple[ResolvedLayerSpec, ...]
    core_levels: tuple[dict[str, Any], ...]
    datasets: dict[str, Any]
    report: dict[str, Any]

    @property
    def root_dir(self) -> Path:
        return self.path.parent

    @property
    def name(self) -> str:
        return str(self.project.get("name", "swanx_project"))

    @property
    def photon_energy_ev(self) -> float:
        try:
            return float(self.settings["photon_energy_ev"])
        except KeyError as error:
            raise ProjectValidationError("settings.photon_energy_ev is required") from error

    @property
    def fit_method(self) -> str:
        return str(self.settings.get("fit_method", "simulate_only"))

    def default_parameter_values(self) -> dict[str, float]:
        return {name: parameter.initial for name, parameter in self.parameters.items()}

    def expanded_layer_ids(self) -> tuple[str, ...]:
        return tuple(layer.id for layer in self.stack)

    def layer_specs_for_values(self, values: Mapping[str, float]) -> list[dict[str, Any]]:
        specs = []
        for layer in self.stack:
            variables = dict(values)
            variables["repeat_index"] = 0.0 if layer.repeat_index is None else float(layer.repeat_index)
            variables["layer_index"] = float(layer.layer_index)
            specs.append(
                {
                    "id": layer.id,
                    "material": layer.material,
                    "tags": list(layer.tags),
                    "thickness": evaluate_number(
                        layer.thickness_expr,
                        variables,
                        label=f"stack layer {layer.id!r} thickness_A",
                    ),
                    "roughness": evaluate_number(
                        layer.roughness_expr,
                        variables,
                        label=f"stack layer {layer.id!r} roughness_A",
                    ),
                }
            )
        return specs

    def to_resolved_mapping(self, values: Mapping[str, float]) -> dict[str, Any]:
        resolved = dict(self.raw)
        resolved["stack"] = [
            {
                "id": layer["id"],
                "material": layer["material"],
                "tags": layer["tags"],
                "thickness_A": layer["thickness"],
                "roughness_A": layer["roughness"],
            }
            for layer in self.layer_specs_for_values(values)
        ]
        resolved["parameters"] = {
            name: {
                "initial": parameter.initial,
                "lower": parameter.lower,
                "upper": parameter.upper,
                "resolved_value": float(values.get(name, parameter.initial)),
            }
            for name, parameter in self.parameters.items()
        }
        return resolved


def load_project_spec(path: str | Path) -> ProjectSpec:
    project_path = Path(path)
    raw = read_yaml(project_path)
    if not isinstance(raw, dict):
        raise ProjectValidationError("project YAML must contain a mapping at the top level")
    return ProjectSpecFactory(project_path, raw).build()


class ProjectSpecFactory:
    def __init__(self, path: Path, raw: dict[str, Any]) -> None:
        self.path = path
        self.raw = raw

    def build(self) -> ProjectSpec:
        _require_sections(
            self.raw,
            (
                "project",
                "settings",
                "materials",
                "parameters",
                "stack",
                "core_levels",
                "datasets",
                "report",
            ),
        )
        project = _as_mapping(self.raw["project"], "project")
        settings = _as_mapping(self.raw["settings"], "settings")
        materials = _as_mapping(self.raw["materials"], "materials")
        parameters = _parse_parameters(_as_mapping(self.raw["parameters"], "parameters"))
        stack = _expand_stack(
            _as_sequence(self.raw["stack"], "stack"),
            parameter_names=set(parameters),
        )
        core_levels = tuple(
            dict(_as_mapping(item, "core_levels item"))
            for item in _as_sequence(self.raw["core_levels"], "core_levels")
        )
        datasets = _as_mapping(self.raw["datasets"], "datasets")
        report = _as_mapping(self.raw["report"], "report")
        spec = ProjectSpec(
            path=self.path,
            raw=dict(self.raw),
            project=dict(project),
            settings=dict(settings),
            materials=dict(materials),
            parameters=parameters,
            stack=stack,
            core_levels=core_levels,
            datasets=dict(datasets),
            report=dict(report),
        )
        _validate_settings(spec)
        _validate_materials(spec)
        _validate_core_levels(spec)
        _validate_datasets(spec)
        spec.layer_specs_for_values(spec.default_parameter_values())
        return spec


def _require_sections(raw: Mapping[str, Any], sections: Sequence[str]) -> None:
    missing = [section for section in sections if section not in raw]
    if missing:
        raise ProjectValidationError(f"missing ProjectSpec section(s): {', '.join(missing)}")


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProjectValidationError(f"{label} must be a mapping")
    return value


def _as_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ProjectValidationError(f"{label} must be a list")
    return value


def _parse_parameters(raw: Mapping[str, Any]) -> dict[str, ParameterSpec]:
    parsed: dict[str, ParameterSpec] = {}
    for name, value in raw.items():
        fields = _as_mapping(value, f"parameters.{name}")
        try:
            parameter = ParameterSpec(
                name=str(name),
                initial=float(fields["initial"]),
                lower=float(fields["lower"]),
                upper=float(fields["upper"]),
            )
        except KeyError as error:
            raise ProjectValidationError(
                f"parameters.{name} requires initial, lower, and upper"
            ) from error
        if parameter.lower >= parameter.upper:
            raise ProjectValidationError(f"parameter {name!r} lower must be smaller than upper")
        if not parameter.lower <= parameter.initial <= parameter.upper:
            raise ProjectValidationError(f"parameter {name!r} initial must be inside bounds")
        parsed[parameter.name] = parameter
    return parsed


def _expand_stack(
    raw_stack: Sequence[Any],
    *,
    parameter_names: set[str],
) -> tuple[ResolvedLayerSpec, ...]:
    layers: list[ResolvedLayerSpec] = []
    for item in raw_stack:
        mapping = _as_mapping(item, "stack item")
        if "repeat" in mapping:
            repeat = _as_mapping(mapping["repeat"], "stack repeat")
            times = int(repeat.get("times", 0))
            if times <= 0:
                raise ProjectValidationError("repeat.times must be positive")
            templates = _as_sequence(repeat.get("layers"), "repeat.layers")
            for repeat_index in range(1, times + 1):
                for template in templates:
                    layers.append(
                        _parse_layer(
                            _as_mapping(template, "repeat layer"),
                            layer_index=len(layers),
                            repeat_index=repeat_index,
                            parameter_names=parameter_names,
                        )
                    )
        else:
            layers.append(
                _parse_layer(
                    mapping,
                    layer_index=len(layers),
                    repeat_index=None,
                    parameter_names=parameter_names,
                )
            )
    ids = [layer.id for layer in layers]
    duplicates = sorted({layer_id for layer_id in ids if ids.count(layer_id) > 1})
    if duplicates:
        raise ProjectValidationError(f"duplicate layer id(s): {', '.join(duplicates)}")
    return tuple(layers)


def _parse_layer(
    raw: Mapping[str, Any],
    *,
    layer_index: int,
    repeat_index: int | None,
    parameter_names: set[str],
) -> ResolvedLayerSpec:
    try:
        raw_id = str(raw["id"])
        material = str(raw["material"])
    except KeyError as error:
        raise ProjectValidationError("each concrete stack layer requires id and material") from error
    try:
        layer_id = raw_id.format(
            repeat_index="" if repeat_index is None else repeat_index,
            layer_index=layer_index,
        )
    except KeyError as error:
        raise ProjectValidationError(f"unknown layer id format key in {raw_id!r}") from error
    tags = tuple(str(tag) for tag in raw.get("tags", ()))
    thickness = raw.get("thickness_A", 0.0)
    roughness = raw.get("roughness_A", 0.0)
    _validate_expression_names(thickness, parameter_names, f"layer {layer_id!r} thickness_A")
    _validate_expression_names(roughness, parameter_names, f"layer {layer_id!r} roughness_A")
    return ResolvedLayerSpec(
        id=layer_id,
        material=material,
        tags=tags,
        thickness_expr=thickness,
        roughness_expr=roughness,
        layer_index=layer_index,
        repeat_index=repeat_index,
    )


def _validate_expression_names(value: Any, parameter_names: set[str], label: str) -> None:
    allowed = set(parameter_names) | {"repeat_index", "layer_index"}
    try:
        names = names_in_expression(value)
    except ExpressionError as error:
        raise ProjectValidationError(str(error)) from error
    unknown = sorted(names - allowed)
    if unknown:
        raise ProjectValidationError(f"unknown parameter(s) in {label}: {', '.join(unknown)}")


def _validate_settings(spec: ProjectSpec) -> None:
    spec.photon_energy_ev
    method = spec.fit_method
    allowed_methods = {
        "simulate_only",
        "jax_least_squares",
        "jax_gradient",
        "bayesian_optimization",
    }
    if method not in allowed_methods:
        raise ProjectValidationError(
            f"settings.fit_method must be one of {sorted(allowed_methods)}"
        )
    polarization = spec.settings.get("polarization", "s")
    if polarization not in {"s", "p", "unpolarized"}:
        raise ProjectValidationError("settings.polarization must be 's', 'p', or 'unpolarized'")


def _validate_materials(spec: ProjectSpec) -> None:
    material_names = set(spec.materials)
    for layer in spec.stack:
        if layer.material.lower() == "vacuum":
            continue
        if layer.material not in material_names:
            raise ProjectValidationError(f"missing material definition for {layer.material!r}")
    for material, fields in spec.materials.items():
        material_fields = _as_mapping(fields, f"materials.{material}")
        for key in ("opc_file", "imfp_file"):
            if key in material_fields:
                _resolve_existing_path(spec, material_fields[key], f"materials.{material}.{key}")


def _validate_core_levels(spec: ProjectSpec) -> None:
    layer_ids = set(spec.expanded_layer_ids())
    tags = {tag for layer in spec.stack for tag in layer.tags}
    for core in spec.core_levels:
        name = str(core.get("name", "<unnamed>"))
        emit_from = _as_mapping(core.get("emit_from", {}), f"core_levels.{name}.emit_from")
        unknown_layers = sorted(set(emit_from.get("layer_ids", ()) or ()) - layer_ids)
        if unknown_layers:
            raise ProjectValidationError(
                f"unknown layer id(s) for core level {name!r}: {', '.join(unknown_layers)}"
            )
        unknown_tags = sorted(set(emit_from.get("tags", ()) or ()) - tags)
        if unknown_tags:
            raise ProjectValidationError(
                f"unknown tag(s) for core level {name!r}: {', '.join(unknown_tags)}"
            )


def _validate_datasets(spec: ProjectSpec) -> None:
    reflectivity = spec.datasets.get("reflectivity")
    if reflectivity:
        fields = _as_mapping(reflectivity, "datasets.reflectivity")
        _resolve_existing_path(spec, fields["path"], "datasets.reflectivity.path")
    for index, dataset in enumerate(spec.datasets.get("rocking_curves", ()) or ()):
        fields = _as_mapping(dataset, f"datasets.rocking_curves[{index}]")
        _resolve_existing_path(spec, fields["path"], f"datasets.rocking_curves[{index}].path")


def _resolve_existing_path(spec: ProjectSpec, value: Any, label: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = spec.root_dir / path
    if not path.exists():
        raise ProjectValidationError(f"missing data file for {label}: {path}")
    return path
