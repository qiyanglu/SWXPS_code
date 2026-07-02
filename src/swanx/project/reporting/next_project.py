"""Follow-up ProjectSpec YAML outputs for fitting runs."""

from __future__ import annotations

import csv
from copy import deepcopy
import os
from pathlib import Path
from typing import Any

from ..builder import BuiltProject
from ..spec import ParameterSpec
from ..yaml_io import read_yaml, write_yaml


def write_next_project_outputs(output: Path, result: Any, built: BuiltProject) -> list[str]:
    """Write optional follow-up ProjectSpec YAML files for a completed fit."""

    options = built.spec.next_project_options
    if not options.get("enabled", False):
        return []
    if built.spec.fit_method == "simulate_only":
        return ["next_project skipped because no fitting was performed"]
    best_parameters = {
        name: float(value)
        for name, value in getattr(result, "best_parameters", {}).items()
    }
    if not best_parameters:
        return ["next_project skipped because best-fit parameters are unavailable"]

    directory = output / "next_project"
    directory.mkdir(parents=True, exist_ok=True)
    original = read_yaml(built.spec.path)
    if not isinstance(original, dict):
        return ["next_project skipped because the original ProjectSpec is not a mapping"]

    notes = [
        f"# SWANX Next Project Suggestions: {built.spec.name}",
        "",
        "Generated follow-up ProjectSpec files are advisory. The original project YAML was not modified.",
        "",
    ]
    report_notes: list[str] = []

    best_mapping = _best_start_mapping(
        original,
        built=built,
        best_parameters=best_parameters,
        target_root=directory,
        suffix="best_start",
    )
    if options.get("best_start", True):
        write_yaml(directory / "project_best_start.yaml", best_mapping)
        report_notes.append("next_project/project_best_start.yaml written with best-fit initial values")
        notes.extend([
            "## Best-Start YAML",
            "",
            "- `project_best_start.yaml` updates varied parameter `initial` values to the fitted best values.",
            "- `run.outputs.next_project` is disabled in the generated YAML to avoid recursive output folders.",
            "",
        ])

    if options.get("reduced", True):
        threshold = float(options.get("low_sensitivity_threshold", 0.02))
        low = _low_sensitivity_parameters(
            output / "identifiability_analysis" / "parameter_identifiability.csv",
            threshold=threshold,
        )
        if low is None:
            report_notes.append(
                "next_project/project_reduced.yaml skipped because identifiability_analysis/parameter_identifiability.csv is unavailable"
            )
            notes.extend([
                "## Reduced YAML",
                "",
                "- `project_reduced.yaml` was not written because identifiability diagnostics were unavailable.",
                "- Enable `run.outputs.identifiability: true` for JAX least-squares runs before using reduced YAML generation.",
                "",
            ])
        else:
            reduced_mapping = deepcopy(best_mapping)
            project = reduced_mapping.get("project")
            if isinstance(project, dict):
                project["name"] = f"{built.spec.name}_reduced"
            fixed = _fix_low_sensitivity_parameters(
                reduced_mapping,
                built.spec.parameters,
                best_parameters,
                low,
            )
            write_yaml(directory / "project_reduced.yaml", reduced_mapping)
            report_notes.append(
                "next_project/project_reduced.yaml written with low-sensitivity parameters fixed"
            )
            notes.extend(_reduction_note_lines(threshold, fixed, low))

    (directory / "reduction_notes.md").write_text("\n".join(notes).rstrip() + "\n", encoding="utf-8")
    report_notes.append("next_project/reduction_notes.md written")
    return report_notes


def _best_start_mapping(
    original: dict[str, Any],
    *,
    built: BuiltProject,
    best_parameters: dict[str, float],
    target_root: Path,
    suffix: str,
) -> dict[str, Any]:
    mapping = deepcopy(original)
    project = mapping.setdefault("project", {})
    if isinstance(project, dict):
        project["name"] = f"{built.spec.name}_{suffix}"
        project.pop("output_dir", None)
    _rewrite_paths(mapping, built.spec.root_dir, target_root)
    _disable_recursive_next_project(mapping)

    parameters = mapping.setdefault("parameters", {})
    if isinstance(parameters, dict):
        for parameter in built.spec.varying_parameters():
            if parameter.name not in best_parameters:
                continue
            fields = parameters.setdefault(parameter.name, {})
            if not isinstance(fields, dict):
                continue
            fields["initial"] = float(best_parameters[parameter.name])
            fields["vary"] = True
            fields.pop("value", None)
    return mapping


def _rewrite_paths(mapping: dict[str, Any], source_root: Path, target_root: Path) -> None:
    materials = mapping.get("materials")
    if isinstance(materials, dict):
        for fields in materials.values():
            if isinstance(fields, dict):
                for key in ("opc_file", "imfp_file"):
                    if fields.get(key):
                        fields[key] = _relative_to_target(fields[key], source_root, target_root)

    datasets = mapping.get("datasets")
    if not isinstance(datasets, dict):
        return
    reflectivity = datasets.get("reflectivity")
    if isinstance(reflectivity, dict) and reflectivity.get("path"):
        reflectivity["path"] = _relative_to_target(reflectivity["path"], source_root, target_root)
    for dataset in datasets.get("rocking_curves", ()) or ():
        if isinstance(dataset, dict) and dataset.get("path"):
            dataset["path"] = _relative_to_target(dataset["path"], source_root, target_root)


def _relative_to_target(value: Any, source_root: Path, target_root: Path) -> str:
    path = Path(str(value))
    if not path.is_absolute():
        path = source_root / path
    try:
        return Path(os.path.relpath(path, target_root)).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _disable_recursive_next_project(mapping: dict[str, Any]) -> None:
    run = mapping.setdefault("run", {})
    if not isinstance(run, dict):
        return
    outputs = run.setdefault("outputs", {})
    if isinstance(outputs, dict):
        outputs["next_project"] = False
    report = mapping.get("report")
    if isinstance(report, dict):
        report.pop("next_project", None)


def _low_sensitivity_parameters(path: Path, *, threshold: float) -> dict[str, dict[str, str]] | None:
    if not path.is_file():
        return None
    low: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row.get("parameter")
            if not name:
                continue
            try:
                sensitivity = float(row.get("relative_sensitivity", "nan"))
            except ValueError:
                continue
            if sensitivity <= threshold:
                low[name] = row
    return low


def _fix_low_sensitivity_parameters(
    mapping: dict[str, Any],
    specs: dict[str, ParameterSpec],
    best_parameters: dict[str, float],
    low: dict[str, dict[str, str]],
) -> list[str]:
    parameters = mapping.setdefault("parameters", {})
    if not isinstance(parameters, dict):
        return []
    fixed: list[str] = []
    for name in sorted(low):
        spec = specs.get(name)
        if spec is None or not spec.vary or name not in best_parameters:
            continue
        fields = parameters.setdefault(name, {})
        if not isinstance(fields, dict):
            continue
        fields.clear()
        fields["value"] = float(best_parameters[name])
        fields["vary"] = False
        fixed.append(name)
    return fixed


def _reduction_note_lines(
    threshold: float,
    fixed: list[str],
    low: dict[str, dict[str, str]],
) -> list[str]:
    lines = [
        "## Reduced YAML",
        "",
        f"- Low-sensitivity threshold: `{threshold}`.",
    ]
    if fixed:
        lines.append("- `project_reduced.yaml` fixes these low-sensitivity parameters at fitted best values:")
        for name in fixed:
            row = low[name]
            lines.append(
                f"  - `{name}`: relative sensitivity `{row.get('relative_sensitivity', '')}`, "
                f"suggestion `{row.get('suggestion', '')}`"
            )
    else:
        lines.append("- `project_reduced.yaml` was written, but no varied parameters met the reduction threshold.")
    lines.extend([
        "",
        "Review this YAML before running it. Low sensitivity is a reduction signal, not proof that a parameter is physically irrelevant.",
        "",
    ])
    return lines


__all__ = ["write_next_project_outputs"]
