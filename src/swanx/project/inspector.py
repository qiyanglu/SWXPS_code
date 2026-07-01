"""Inspection helpers for YAML ProjectSpec files."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from .spec import ProjectSpec, load_project_spec


def inspect_project(path: str | Path) -> str:
    """Return a human-readable summary of a YAML ProjectSpec without running it."""

    spec = load_project_spec(path)
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
    return "\n".join(lines) + "\n"


def _section(lines: list[str], title: str) -> None:
    if lines:
        lines.append("")
    lines.append(f"[{title}]")


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
