"""Inspection helpers for YAML ProjectSpec files."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from .spec import ProjectSpec, ProjectValidationError, load_project_spec
from .yaml_io import read_yaml


def inspect_project(path: str | Path) -> str:
    """Return a human-readable summary of a YAML ProjectSpec without running it."""

    try:
        spec = load_project_spec(path)
    except ProjectValidationError as error:
        return _inspect_invalid_project(path, error)
    lines: list[str] = []
    _section(lines, "Project")
    lines.append(f"name: {spec.name}")
    lines.append(f"source: {spec.path}")
    lines.append(f"fit_method: {spec.fit_method}")
    lines.append(f"photon_energy_ev: {spec.photon_energy_ev}")
    lines.append(f"polarization: {spec.settings.get('polarization', 's')}")

    _section(lines, "Output")
    lines.append(f"default_output_dir: {_output_dir_preview(spec)}")
    lines.append(f"plots: {spec.save_plots}")
    lines.append(f"identifiability: {spec.identifiability_options.get('enabled', False)}")

    _section(lines, "Materials")
    for name, fields in spec.materials.items():
        if not isinstance(fields, dict):
            continue
        opc = _resolve(spec, fields.get("opc_file")) if fields.get("opc_file") else "not set"
        imfp = _resolve(spec, fields.get("imfp_file")) if fields.get("imfp_file") else "not set"
        lines.append(f"{name}: opc_file={opc}; imfp_file={imfp}")
    if not spec.materials:
        lines.append("none")

    _section(lines, "Stack")
    lines.append(f"layer_count: {len(spec.stack)}")
    lines.append("layer_ids: " + ", ".join(spec.expanded_layer_ids()))

    _section(lines, "Core Levels")
    if spec.core_levels:
        for core in spec.core_levels:
            emit_from = core.get("emit_from", {})
            selector = _selector_text(emit_from if isinstance(emit_from, dict) else {})
            lines.append(f"{core.get('name', '<unnamed>')}: {selector}")
    else:
        lines.append("none")

    _section(lines, "Datasets")
    reflectivity = spec.datasets.get("reflectivity")
    if isinstance(reflectivity, dict):
        lines.append(f"reflectivity: {reflectivity.get('name', 'Reflectivity')} -> {_resolve(spec, reflectivity.get('path'))}")
    rocking = spec.datasets.get("rocking_curves", ()) or ()
    for dataset in rocking:
        if isinstance(dataset, dict):
            lines.append(f"rocking_curve: {dataset.get('name', '<unnamed>')} -> {_resolve(spec, dataset.get('path'))}")
    if not reflectivity and not rocking:
        lines.append("none")

    _section(lines, "Varying Parameters")
    varying = spec.varying_parameters()
    if varying:
        for parameter in varying:
            lines.append(
                f"{parameter.name}: initial={parameter.initial}, lower={parameter.lower}, upper={parameter.upper}"
            )
    else:
        lines.append("none")

    _section(lines, "Optional Dependencies")
    for module, label in (
        ("yaml", "PyYAML project"),
        ("matplotlib", "matplotlib plots"),
        ("jax", "JAX"),
        ("scipy", "SciPy optimizers"),
        ("skopt", "Bayesian optimization"),
    ):
        lines.append(f"{label}: {'available' if importlib.util.find_spec(module) else 'missing'}")

    _section(lines, "Fitting Callbacks")
    lines.extend(_callback_status(spec))

    _section(lines, "Doctor")
    lines.extend(_doctor_lines(spec))
    return "\n".join(lines) + "\n"


def _section(lines: list[str], title: str) -> None:
    if lines:
        lines.append("")
    lines.append(f"[{title}]")


def _inspect_invalid_project(path: str | Path, error: ProjectValidationError) -> str:
    project_path = Path(path)
    raw = read_yaml(project_path)
    if not isinstance(raw, dict):
        raw = {}
    lines: list[str] = []
    _section(lines, "Project")
    lines.append(f"source: {project_path}")
    lines.append(f"validation_error: {error}")
    _section(lines, "Doctor")
    lines.extend(_doctor_lines_from_raw(project_path, raw))
    return "\n".join(lines) + "\n"


def _resolve(spec: ProjectSpec, value: Any) -> str:
    if value is None:
        return "not set"
    path = Path(str(value))
    if not path.is_absolute():
        path = spec.root_dir / path
    return str(path)


def _output_dir_preview(spec: ProjectSpec) -> str:
    if spec.project.get("output_dir"):
        output = Path(str(spec.project["output_dir"]))
        if not output.is_absolute():
            output = spec.root_dir / output
        return str(output)
    return str(spec.root_dir / "runs" / f"{spec.name}_<timestamp>")


def _selector_text(emit_from: dict[str, Any]) -> str:
    if emit_from.get("all"):
        return "emit_from=all"
    parts = []
    if emit_from.get("layer_ids"):
        parts.append("layer_ids=" + ",".join(str(item) for item in emit_from.get("layer_ids", ())))
    if emit_from.get("tags"):
        parts.append("tags=" + ",".join(str(item) for item in emit_from.get("tags", ())))
    return "; ".join(parts) if parts else "emit_from=missing"


def _callback_status(spec: ProjectSpec) -> list[str]:
    optimizer = spec.optimizer_settings
    if spec.fit_method == "jax_least_squares":
        factory = optimizer.get("residual_function_factory")
        if factory:
            return [f"residual_function_factory: {factory}"]
        return [f"residual: {optimizer.get('residual', 'auto_fixed_grid')}"]
    if spec.fit_method == "jax_gradient":
        factory = optimizer.get("value_and_grad_factory")
        return [f"value_and_grad_factory: {factory if factory else 'missing'}"]
    if spec.fit_method == "bayesian_optimization":
        return ["callback_required: no"]
    return ["callback_required: no"]


def _doctor_lines(spec: ProjectSpec) -> list[str]:
    lines: list[str] = []
    material_paths = _material_paths(spec)
    dataset_paths = _dataset_paths(spec)
    lines.extend(_file_status_lines(material_paths, dataset_paths))
    lines.extend(_dependency_status_lines())

    mode_ready = spec.fit_method == "jax_least_squares"
    residual_ready = _auto_fixed_grid_residual_selected(spec)
    datasets_ready = bool(dataset_paths) and all(path.exists() for _, path in dataset_paths)
    slicing_ready = _fixed_grid_slicing_selected(spec)
    lines.extend(_auto_fixed_grid_readiness_lines(mode_ready, residual_ready, datasets_ready, slicing_ready))
    return lines


def _doctor_lines_from_raw(project_path: Path, raw: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    material_paths = _raw_material_paths(project_path, raw)
    dataset_paths = _raw_dataset_paths(project_path, raw)
    lines.extend(_file_status_lines(material_paths, dataset_paths))
    lines.extend(_dependency_status_lines())

    run = raw.get("run", {})
    settings = raw.get("settings", {})
    optimizer = {}
    if isinstance(settings, dict) and isinstance(settings.get("optimizer"), dict):
        optimizer.update(settings["optimizer"])
    if isinstance(run, dict) and isinstance(run.get("optimizer"), dict):
        optimizer.update(run["optimizer"])
    mode = None
    if isinstance(run, dict):
        mode = run.get("mode")
    if mode is None and isinstance(settings, dict):
        mode = settings.get("fit_method", "simulate_only")
    slicing = settings.get("slicing") if isinstance(settings, dict) else None
    mode_ready = str(mode) == "jax_least_squares"
    residual_ready = not optimizer.get("residual_function_factory") and str(
        optimizer.get("residual", "auto_fixed_grid")
    ) in {"auto", "auto_fixed_grid"}
    datasets_ready = bool(dataset_paths) and all(path.exists() for _, path in dataset_paths)
    slicing_ready = isinstance(slicing, dict) and str(slicing.get("mode", "")) in {"fixed", "fixed_grid"}
    lines.extend(_auto_fixed_grid_readiness_lines(mode_ready, residual_ready, datasets_ready, slicing_ready))
    return lines


def _file_status_lines(
    material_paths: list[tuple[str, Path]],
    dataset_paths: list[tuple[str, Path]],
) -> list[str]:
    missing_materials = [label for label, path in material_paths if not path.exists()]
    missing_datasets = [label for label, path in dataset_paths if not path.exists()]
    lines = [f"material files: {_ok_missing(missing_materials)}"]
    lines.extend(f"  MISSING {label}" for label in missing_materials)
    lines.append(f"dataset files: {_ok_missing(missing_datasets)}")
    lines.extend(f"  MISSING {label}" for label in missing_datasets)
    return lines


def _dependency_status_lines() -> list[str]:
    lines: list[str] = []
    matplotlib_available = _available("matplotlib")
    lines.append(f"matplotlib: {_available_text(matplotlib_available)}")
    if not matplotlib_available:
        lines.append("plot consequence: plots will be skipped and noted in report.md")
    else:
        lines.append("plot consequence: plots can be written when run.outputs.plots is true")

    jax_available = _available("jax")
    scipy_available = _available("scipy")
    lines.append(
        "jax_least_squares deps: "
        f"JAX={_available_text(jax_available)}; SciPy={_available_text(scipy_available)}"
    )
    skopt_available = _available("skopt")
    lines.append(f"bayesian_optimization deps: scikit-optimize={_available_text(skopt_available)}")

    if not (matplotlib_available and jax_available and scipy_available and skopt_available):
        lines.append('suggested install: python -m pip install -e ".[project,least-squares,plot]"')
    return lines


def _auto_fixed_grid_readiness_lines(
    mode_ready: bool,
    residual_ready: bool,
    datasets_ready: bool,
    slicing_ready: bool,
) -> list[str]:
    return [
        "auto_fixed_grid readiness:",
        f"  mode is jax_least_squares: {_yes_no(mode_ready)}",
        f"  residual is auto_fixed_grid: {_yes_no(residual_ready)}",
        f"  datasets exist: {_yes_no(datasets_ready)}",
        f"  settings.slicing.mode is fixed_grid: {_yes_no(slicing_ready)}",
    ]


def _material_paths(spec: ProjectSpec) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for material, fields in spec.materials.items():
        if not isinstance(fields, dict):
            continue
        for key in ("opc_file", "imfp_file"):
            if fields.get(key):
                paths.append(
                    (
                        f"materials.{material}.{key}: {_resolve(spec, fields[key])}",
                        _path(spec, fields[key]),
                    )
                )
    return paths


def _dataset_paths(spec: ProjectSpec) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    reflectivity = spec.datasets.get("reflectivity")
    if isinstance(reflectivity, dict) and reflectivity.get("path"):
        paths.append(
            (
                f"datasets.reflectivity.path: {_resolve(spec, reflectivity['path'])}",
                _path(spec, reflectivity["path"]),
            )
        )
    for index, dataset in enumerate(spec.datasets.get("rocking_curves", ()) or ()):
        if isinstance(dataset, dict) and dataset.get("path"):
            paths.append(
                (
                    f"datasets.rocking_curves[{index}].path: {_resolve(spec, dataset['path'])}",
                    _path(spec, dataset["path"]),
                )
            )
    return paths


def _raw_material_paths(project_path: Path, raw: dict[str, Any]) -> list[tuple[str, Path]]:
    materials = raw.get("materials", {})
    if not isinstance(materials, dict):
        return []
    paths: list[tuple[str, Path]] = []
    for material, fields in materials.items():
        if not isinstance(fields, dict):
            continue
        for key in ("opc_file", "imfp_file"):
            if fields.get(key):
                value = fields[key]
                paths.append(
                    (
                        f"materials.{material}.{key}: {_resolve_from_root(project_path.parent, value)}",
                        _path_from_root(project_path.parent, value),
                    )
                )
    return paths


def _raw_dataset_paths(project_path: Path, raw: dict[str, Any]) -> list[tuple[str, Path]]:
    datasets = raw.get("datasets", {})
    if not isinstance(datasets, dict):
        return []
    paths: list[tuple[str, Path]] = []
    reflectivity = datasets.get("reflectivity")
    if isinstance(reflectivity, dict) and reflectivity.get("path"):
        value = reflectivity["path"]
        paths.append(
            (
                f"datasets.reflectivity.path: {_resolve_from_root(project_path.parent, value)}",
                _path_from_root(project_path.parent, value),
            )
        )
    for index, dataset in enumerate(datasets.get("rocking_curves", ()) or ()):
        if isinstance(dataset, dict) and dataset.get("path"):
            value = dataset["path"]
            paths.append(
                (
                    f"datasets.rocking_curves[{index}].path: {_resolve_from_root(project_path.parent, value)}",
                    _path_from_root(project_path.parent, value),
                )
            )
    return paths


def _path(spec: ProjectSpec, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else spec.root_dir / path


def _path_from_root(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _resolve_from_root(root: Path, value: Any) -> str:
    return str(_path_from_root(root, value))


def _available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _available_text(available: bool) -> str:
    return "available" if available else "missing"


def _ok_missing(missing: list[str]) -> str:
    return "OK" if not missing else "MISSING"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _auto_fixed_grid_residual_selected(spec: ProjectSpec) -> bool:
    optimizer = spec.optimizer_settings
    if optimizer.get("residual_function_factory"):
        return False
    return str(optimizer.get("residual", "auto_fixed_grid")) in {"auto", "auto_fixed_grid"}


def _fixed_grid_slicing_selected(spec: ProjectSpec) -> bool:
    slicing = spec.settings.get("slicing")
    return isinstance(slicing, dict) and str(slicing.get("mode", "")) in {"fixed", "fixed_grid"}
