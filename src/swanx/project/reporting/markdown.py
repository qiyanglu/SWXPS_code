"""Markdown report writer for YAML project runs."""

from __future__ import annotations

import csv
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
        lines.extend(_fit_interpretation_lines(output, built, result, evaluation))
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


def _fit_interpretation_lines(
    output: Path,
    built: BuiltProject,
    result: Any,
    evaluation: Any,
) -> list[str]:
    if built.spec.fit_method == "bayesian_optimization":
        return [
            "",
            "## Fit Interpretation",
            "",
            "Bayesian optimization was used as a global black-box search. "
            "Covariance and correlation interpretation is not reported for BO runs.",
        ]
    lines = ["", "## Fit Interpretation", ""]
    cost = getattr(result, "final_cost", None)
    objective = None if evaluation is None else getattr(evaluation, "objective", None)
    if built.spec.fit_method == "jax_least_squares":
        lines.append(f"- Final least-squares cost: {_format_report_value(cost)}")
        lines.append(f"- Final objective from maintained evaluator: {_format_report_value(objective)}")
    else:
        lines.append(f"- Final objective: {_format_report_value(objective)}")

    near_bounds = _near_bound_parameters(built, result)
    if near_bounds:
        lines.append("- Near-bound parameters: " + ", ".join(near_bounds))
    else:
        lines.append("- Near-bound parameters: none using the default report tolerance")

    ident_dir = output / "identifiability_analysis"
    if built.spec.fit_method == "jax_least_squares" and (ident_dir / "summary.md").is_file():
        lines.extend(_identifiability_summary_lines(ident_dir))
    return lines


def _near_bound_parameters(
    built: BuiltProject,
    result: Any,
    tolerance: float = 0.02,
) -> list[str]:
    best = dict(built.values)
    best.update(
        {
            name: float(value)
            for name, value in getattr(result, "best_parameters", {}).items()
        }
    )
    near = []
    for parameter in built.spec.varying_parameters():
        initial, lower, upper = parameter.require_bounds()
        del initial
        width = float(upper) - float(lower)
        if width <= 0.0:
            continue
        value = float(best.get(parameter.name, lower))
        scaled = (value - float(lower)) / width
        if scaled <= tolerance:
            near.append(f"{parameter.name} near lower bound")
        elif scaled >= 1.0 - tolerance:
            near.append(f"{parameter.name} near upper bound")
    return near


def _identifiability_summary_lines(directory: Path) -> list[str]:
    lines = [
        "- Identifiability analysis: see `identifiability_analysis/summary.md`.",
    ]
    parameter_rows = _csv_dicts(directory / "parameter_identifiability.csv")
    if parameter_rows:
        weak = sorted(
            parameter_rows,
            key=lambda row: _float(row.get("relative_sensitivity")),
        )[:3]
        lines.append(
            "- Weakly identifiable parameters: "
            + ", ".join(row["parameter"] for row in weak if row.get("parameter"))
        )
        weak_mode = sorted(
            parameter_rows,
            key=lambda row: _float(row.get("weak_mode_participation")),
            reverse=True,
        )[:3]
        lines.append(
            "- Highest weak-mode participation: "
            + ", ".join(row["parameter"] for row in weak_mode if row.get("parameter"))
        )

    correlation_rows = _csv_dicts(directory / "strong_correlation_pairs.csv")
    if correlation_rows:
        strongest = sorted(
            correlation_rows,
            key=lambda row: _float(row.get("abs_correlation")),
            reverse=True,
        )[:3]
        lines.append(
            "- Strongest correlations: "
            + ", ".join(
                f"{row.get('parameter_1')} vs {row.get('parameter_2')}"
                for row in strongest
            )
        )
    else:
        lines.append("- Strongest correlations: none above the configured threshold")

    if (directory / "dataset_sensitivity.csv").is_file():
        lines.append(
            "- Dataset sensitivity caveat: this uses the final weighted residual "
            "Jacobian, so it is a scaling/weighting audit signal rather than "
            "proof that one physical data type was scaled incorrectly."
        )
    return lines


def _csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: str | None) -> float:
    try:
        return float(value) if value not in (None, "") else float("nan")
    except ValueError:
        return float("nan")


def _format_report_value(value: Any) -> str:
    if value is None:
        return "not available"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)
