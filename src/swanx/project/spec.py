"""YAML ProjectSpec data model and validation."""

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
    value: float
    vary: bool
    initial: float | None = None
    lower: float | None = None
    upper: float | None = None

    def require_bounds(self) -> tuple[float, float, float]:
        if self.initial is None or self.lower is None or self.upper is None:
            raise ProjectValidationError(f"varying parameter {self.name!r} requires initial, lower, and upper")
        return self.initial, self.lower, self.upper


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
    """Validated YAML ProjectSpec.

    ``roughness_A`` on layer j means roughness/interdiffusion at the upper
    interface of layer j, i.e. interface between layer j-1 and layer j.
    """

    path: Path
    raw: dict[str, Any]
    project: dict[str, Any]
    run: dict[str, Any]
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

    @property
    def optimizer_settings(self) -> dict[str, Any]:
        raw = self.settings.get("optimizer", {}) or {}
        if not isinstance(raw, Mapping):
            raise ProjectValidationError("settings.optimizer must be a mapping")
        return dict(raw)

    @property
    def save_plots(self) -> bool:
        return bool(self.report.get("save_plots", False))

    @property
    def identifiability_options(self) -> dict[str, Any]:
        raw = self.report.get("identifiability", False)
        if raw is True:
            return {"enabled": True}
        if raw in (False, None):
            return {"enabled": False}
        if not isinstance(raw, Mapping):
            raise ProjectValidationError(
                "run.outputs.identifiability/report.identifiability must be a boolean or mapping"
            )
        options = dict(raw)
        options.setdefault("enabled", True)
        return options

    def default_parameter_values(self) -> dict[str, float]:
        return {name: parameter.value for name, parameter in self.parameters.items()}

    def varying_parameters(self) -> tuple[ParameterSpec, ...]:
        return tuple(parameter for parameter in self.parameters.values() if parameter.vary)

    def expanded_layer_ids(self) -> tuple[str, ...]:
        return tuple(layer.id for layer in self.stack)

    def layer_specs_for_values(self, values: Mapping[str, float]) -> list[dict[str, Any]]:
        specs = []
        for layer in self.stack:
            variables = dict(values)
            variables["repeat_index"] = 0.0 if layer.repeat_index is None else float(layer.repeat_index)
            variables["repeat_index0"] = 0.0 if layer.repeat_index is None else float(layer.repeat_index - 1)
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
                "value": parameter.value,
                "vary": parameter.vary,
                "initial": parameter.initial,
                "lower": parameter.lower,
                "upper": parameter.upper,
                "resolved_value": float(values.get(name, parameter.value)),
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
                "stack",
                "core_levels",
            ),
        )
        raw_with_defaults = dict(self.raw)
        raw_with_defaults.setdefault("parameters", {})
        raw_with_defaults.setdefault("datasets", {})
        raw_with_defaults.setdefault("report", {})
        raw_with_defaults.setdefault("run", {})
        project = _as_mapping(raw_with_defaults["project"], "project")
        run = _as_mapping(raw_with_defaults["run"], "run")
        settings = _as_mapping(raw_with_defaults["settings"], "settings")
        report = _as_mapping(raw_with_defaults["report"], "report")
        settings, report = _merge_run_controls(settings, report, run)
        raw_with_defaults["settings"] = dict(settings)
        raw_with_defaults["report"] = dict(report)
        materials = _as_mapping(raw_with_defaults["materials"], "materials")
        parameters = _parse_parameters(_as_mapping(raw_with_defaults["parameters"], "parameters"))
        stack = _expand_stack(
            _as_sequence(raw_with_defaults["stack"], "stack"),
            parameter_names=set(parameters),
        )
        core_levels = tuple(
            dict(_as_mapping(item, "core_levels item"))
            for item in _as_sequence(raw_with_defaults["core_levels"], "core_levels")
        )
        datasets = _as_mapping(raw_with_defaults["datasets"], "datasets")
        spec = ProjectSpec(
            path=self.path,
            raw=raw_with_defaults,
            project=dict(project),
            run=dict(run),
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


def _merge_run_controls(
    settings: Mapping[str, Any],
    report: Mapping[str, Any],
    run: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    effective_settings = dict(settings)
    effective_report = dict(report)

    if "mode" in run:
        run_mode = str(run["mode"])
        if "fit_method" in settings and str(settings["fit_method"]) != run_mode:
            raise ProjectValidationError(
                "run.mode conflicts with settings.fit_method; use one value"
            )
        effective_settings["fit_method"] = run_mode

    if "optimizer" in run:
        run_optimizer = _as_mapping(run["optimizer"], "run.optimizer")
        legacy_optimizer = settings.get("optimizer", {}) or {}
        legacy_optimizer = _as_mapping(legacy_optimizer, "settings.optimizer")
        effective_settings["optimizer"] = _merge_mappings_no_conflict(
            legacy_optimizer,
            run_optimizer,
            "settings.optimizer",
            "run.optimizer",
        )

    if "outputs" in run:
        outputs = _as_mapping(run["outputs"], "run.outputs")
        if "plots" in outputs:
            run_plots = bool(outputs["plots"])
            if "save_plots" in report and bool(report["save_plots"]) != run_plots:
                raise ProjectValidationError(
                    "run.outputs.plots conflicts with report.save_plots; use one value"
                )
            effective_report["save_plots"] = run_plots
        if "identifiability" in outputs:
            run_identifiability = outputs["identifiability"]
            if "identifiability" in report:
                effective_report["identifiability"] = _merge_identifiability_options(
                    report["identifiability"],
                    run_identifiability,
                )
            else:
                effective_report["identifiability"] = run_identifiability

    return effective_settings, effective_report


def _merge_mappings_no_conflict(
    legacy: Mapping[str, Any],
    modern: Mapping[str, Any],
    legacy_label: str,
    modern_label: str,
) -> dict[str, Any]:
    result = dict(legacy)
    for key, value in modern.items():
        if key in result and result[key] != value:
            raise ProjectValidationError(
                f"{modern_label}.{key} conflicts with {legacy_label}.{key}; use one value"
            )
        result[key] = value
    return result


def _merge_identifiability_options(legacy: Any, modern: Any) -> Any:
    if _identifiability_enabled(legacy) != _identifiability_enabled(modern):
        raise ProjectValidationError(
            "run.outputs.identifiability conflicts with report.identifiability; use one value"
        )
    if isinstance(legacy, Mapping) and isinstance(modern, Mapping):
        return _merge_mappings_no_conflict(
            legacy,
            modern,
            "report.identifiability",
            "run.outputs.identifiability",
        )
    if isinstance(modern, Mapping):
        return dict(modern)
    if isinstance(legacy, Mapping):
        return dict(legacy)
    return bool(modern)


def _identifiability_enabled(raw: Any) -> bool:
    if isinstance(raw, Mapping):
        return bool(raw.get("enabled", True))
    return bool(raw)


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
        has_bounds = any(key in fields for key in ("initial", "lower", "upper"))
        vary = bool(fields.get("vary", True if has_bounds else False))
        if vary:
            missing = [key for key in ("initial", "lower", "upper") if key not in fields]
            if missing:
                raise ProjectValidationError(
                    f"parameters.{name} with vary: true requires initial, lower, and upper"
                )
            initial = float(fields["initial"])
            lower = float(fields["lower"])
            upper = float(fields["upper"])
            if lower >= upper:
                raise ProjectValidationError(f"parameter {name!r} lower must be smaller than upper")
            if not lower <= initial <= upper:
                raise ProjectValidationError(f"parameter {name!r} initial must be inside bounds")
            parameter = ParameterSpec(
                name=str(name),
                value=initial,
                vary=True,
                initial=initial,
                lower=lower,
                upper=upper,
            )
        else:
            if "value" in fields:
                value_float = float(fields["value"])
            elif "initial" in fields:
                value_float = float(fields["initial"])
            else:
                raise ProjectValidationError(
                    f"parameters.{name} with vary: false requires value or initial"
                )
            parameter = ParameterSpec(
                name=str(name),
                value=value_float,
                vary=False,
                initial=float(fields["initial"]) if "initial" in fields else None,
                lower=float(fields["lower"]) if "lower" in fields else None,
                upper=float(fields["upper"]) if "upper" in fields else None,
            )
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
    allowed = set(parameter_names) | {"repeat_index", "repeat_index0", "layer_index"}
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
            f"run.mode/settings.fit_method must be one of {sorted(allowed_methods)}"
        )
    polarization = spec.settings.get("polarization", "s")
    if polarization not in {"s", "p", "unpolarized"}:
        raise ProjectValidationError("settings.polarization must be 's', 'p', or 'unpolarized'")
    _validate_slicing_setting(spec)
    _validate_offpeak_mask_setting(spec)
    spec.identifiability_options
    optimizer = spec.optimizer_settings
    has_datasets = bool(spec.datasets.get("reflectivity") or spec.datasets.get("rocking_curves"))
    if method == "jax_least_squares":
        _validate_jax_least_squares_optimizer(spec, optimizer, has_datasets)
    if method == "jax_gradient" and has_datasets and not optimizer.get("value_and_grad_factory"):
        raise ProjectValidationError(
            "run.mode='jax_gradient' requires "
            "run.optimizer.value_and_grad_factory='module:function' for the "
            "fixed-shape JAX value-and-gradient callback. Install with python -m pip install -e "
            "\".[project,gradient]\" and provide a factory, or use "
            "run.mode: \"simulate_only\" for simulation-only "
            "projects. Bayesian optimization is not used as a fallback."
        )


def _validate_jax_least_squares_optimizer(
    spec: ProjectSpec,
    optimizer: Mapping[str, Any],
    has_datasets: bool,
) -> None:
    residual = str(optimizer.get("residual", "auto_fixed_grid"))
    factory = optimizer.get("residual_function_factory")
    if factory and "residual" in optimizer:
        raise ProjectValidationError(
            "run.optimizer.residual_function_factory conflicts with "
            "run.optimizer.residual; use one residual source"
        )
    if residual not in {"auto", "auto_fixed_grid"}:
        raise ProjectValidationError(
            "run.optimizer.residual must be 'auto_fixed_grid' or use "
            "run.optimizer.residual_function_factory='module:function'"
        )
    if not has_datasets:
        return
    if factory:
        return
    raw_slicing = spec.settings.get("slicing")
    fixed_grid = isinstance(raw_slicing, Mapping) and str(raw_slicing.get("mode", "adaptive")) in {
        "fixed",
        "fixed_grid",
    }
    if not fixed_grid:
        raise ProjectValidationError(
            "run.optimizer.residual='auto_fixed_grid' requires "
            "settings.slicing.mode: 'fixed_grid'"
        )

def _validate_slicing_setting(spec: ProjectSpec) -> None:
    raw = spec.settings.get("slicing")
    if raw is None:
        return
    if isinstance(raw, str):
        if raw not in {"adaptive", "unified", "legacy"}:
            raise ProjectValidationError("settings.slicing must be 'adaptive', 'unified', 'legacy', or a mapping")
        return
    if not isinstance(raw, Mapping):
        raise ProjectValidationError("settings.slicing must be 'adaptive', 'unified', 'legacy', or a mapping")
    mode = str(raw.get("mode", "adaptive"))
    if mode not in {"adaptive", "unified", "legacy", "none", "fixed", "fixed_grid"}:
        raise ProjectValidationError("settings.slicing.mode must be adaptive, unified, legacy, fixed, or fixed_grid")
    if "min_slices" in raw and int(raw["min_slices"]) <= 0:
        raise ProjectValidationError("settings.slicing.min_slices must be positive")
    max_slice = raw.get("max_slice_thickness_A", raw.get("max_slice_thickness"))
    if max_slice is not None and float(max_slice) <= 0:
        raise ProjectValidationError("settings.slicing.max_slice_thickness_A must be positive")
    reference_values = raw.get("reference_values", {}) or {}
    if not isinstance(reference_values, Mapping):
        raise ProjectValidationError("settings.slicing.reference_values must be a mapping")
    unknown = sorted(set(reference_values) - set(spec.parameters))
    if unknown:
        raise ProjectValidationError("unknown parameter(s) in settings.slicing.reference_values: " + ", ".join(unknown))
    for name, value in reference_values.items():
        try:
            float(value)
        except (TypeError, ValueError) as error:
            raise ProjectValidationError(f"settings.slicing.reference_values.{name} must be numeric") from error


def _validate_offpeak_mask_setting(spec: ProjectSpec) -> None:
    raw = spec.settings.get("rocking_curve_offpeak_mask", spec.settings.get("offpeak_mask"))
    if raw is None:
        return
    if not isinstance(raw, Mapping):
        raise ProjectValidationError("settings.rocking_curve_offpeak_mask must be a mapping")
    mode = str(raw.get("mode", "exclude_reflectivity_peak"))
    if mode != "exclude_reflectivity_peak":
        raise ProjectValidationError("settings.rocking_curve_offpeak_mask.mode must be 'exclude_reflectivity_peak'")
    if float(raw.get("half_width_deg", 1.25)) <= 0:
        raise ProjectValidationError("settings.rocking_curve_offpeak_mask.half_width_deg must be positive")
    if not spec.datasets.get("reflectivity"):
        raise ProjectValidationError("settings.rocking_curve_offpeak_mask requires datasets.reflectivity")


def _validate_materials(spec: ProjectSpec) -> None:
    material_names = set(spec.materials)
    for layer in spec.stack:
        if layer.material.lower() == "vacuum":
            continue
        if layer.material not in material_names:
            raise ProjectValidationError(f"missing material definition for {layer.material!r}")
        material_fields = _as_mapping(spec.materials[layer.material], f"materials.{layer.material}")
        if "opc_file" not in material_fields:
            raise ProjectValidationError(
                f"non-vacuum stack material {layer.material!r} requires materials.{layer.material}.opc_file"
            )
    for material, fields in spec.materials.items():
        material_fields = _as_mapping(fields, f"materials.{material}")
        for key in ("opc_file", "imfp_file"):
            if key in material_fields:
                _resolve_existing_path(spec, material_fields[key], f"materials.{material}.{key}")
    emitting_materials = _emitting_materials(spec)
    for material in sorted(emitting_materials):
        if material.lower() == "vacuum":
            continue
        material_fields = _as_mapping(spec.materials.get(material, {}), f"materials.{material}")
        if "imfp_file" not in material_fields:
            raise ProjectValidationError(
                f"emitting material {material!r} requires materials.{material}.imfp_file"
            )


def _validate_core_levels(spec: ProjectSpec) -> None:
    for core in spec.core_levels:
        name = str(core.get("name", "<unnamed>"))
        _validate_emit_from(spec, core.get("emit_from"), name)


def _validate_emit_from(spec: ProjectSpec, raw_emit_from: Any, core_name: str) -> Mapping[str, Any]:
    if raw_emit_from is None:
        raise ProjectValidationError(
            f"core level {core_name!r} requires emit_from.layer_ids, emit_from.tags, or emit_from.all: true"
        )
    emit_from = _as_mapping(raw_emit_from, f"core_levels.{core_name}.emit_from")
    layer_ids = set(spec.expanded_layer_ids())
    tags = {tag for layer in spec.stack for tag in layer.tags}
    all_layers = bool(emit_from.get("all", False))
    has_layer_ids = bool(emit_from.get("layer_ids"))
    has_tags = bool(emit_from.get("tags"))
    if all_layers and (has_layer_ids or has_tags):
        raise ProjectValidationError(
            f"core level {core_name!r} emit_from.all cannot be combined with layer_ids or tags"
        )
    if not all_layers and not (has_layer_ids or has_tags):
        raise ProjectValidationError(
            f"core level {core_name!r} requires emit_from.layer_ids, emit_from.tags, or emit_from.all: true"
        )
    unknown_layers = sorted(set(emit_from.get("layer_ids", ()) or ()) - layer_ids)
    if unknown_layers:
        raise ProjectValidationError(
            f"unknown layer id(s) for core level {core_name!r}: {', '.join(unknown_layers)}"
        )
    unknown_tags = sorted(set(emit_from.get("tags", ()) or ()) - tags)
    if unknown_tags:
        raise ProjectValidationError(
            f"unknown tag(s) for core level {core_name!r}: {', '.join(unknown_tags)}"
        )
    return emit_from


def _emitting_materials(spec: ProjectSpec) -> set[str]:
    materials: set[str] = set()
    for core in spec.core_levels:
        name = str(core.get("name", "<unnamed>"))
        emit_from = _validate_emit_from(spec, core.get("emit_from"), name)
        if bool(emit_from.get("all", False)):
            selected = spec.stack
        else:
            layer_ids = set(emit_from.get("layer_ids", ()) or ())
            tags = set(emit_from.get("tags", ()) or ())
            selected = tuple(
                layer for layer in spec.stack
                if layer.id in layer_ids or tags.intersection(layer.tags)
            )
        materials.update(layer.material for layer in selected)
    return materials


def _validate_datasets(spec: ProjectSpec) -> None:
    reflectivity = spec.datasets.get("reflectivity")
    if reflectivity:
        fields = _as_mapping(reflectivity, "datasets.reflectivity")
        _resolve_existing_path(spec, fields["path"], "datasets.reflectivity.path")
        _validate_optional_positive_float(fields, "weight", "datasets.reflectivity.weight", allow_zero=True)
        _validate_optional_positive_float(fields, "log_floor", "datasets.reflectivity.log_floor")
    for index, dataset in enumerate(spec.datasets.get("rocking_curves", ()) or ()):
        fields = _as_mapping(dataset, f"datasets.rocking_curves[{index}]")
        _resolve_existing_path(spec, fields["path"], f"datasets.rocking_curves[{index}].path")
        _validate_optional_positive_float(
            fields,
            "weight",
            f"datasets.rocking_curves[{index}].weight",
            allow_zero=True,
        )


def _validate_optional_positive_float(
    fields: Mapping[str, Any],
    key: str,
    label: str,
    *,
    allow_zero: bool = False,
) -> None:
    if key not in fields:
        return
    value = float(fields[key])
    invalid = value < 0 if allow_zero else value <= 0
    if invalid:
        comparator = "non-negative" if allow_zero else "positive"
        raise ProjectValidationError(f"{label} must be {comparator}")


def _resolve_existing_path(spec: ProjectSpec, value: Any, label: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = spec.root_dir / path
    if not path.exists():
        raise ProjectValidationError(
            f"missing data file for {label}: {path}. "
            f"Relative ProjectSpec paths are resolved from {spec.root_dir}. "
            "Check the path, run `swanx inspect project.yaml`, or regenerate starter data with "
            "`swanx init --copy-example-data`."
        )
    return path
