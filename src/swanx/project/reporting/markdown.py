"""Markdown report writer for YAML project runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..builder import BuiltProject


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
