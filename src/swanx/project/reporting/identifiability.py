"""Identifiability report outputs for YAML ProjectSpec runs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from swanx.diagnostics import (
    IdentifiabilityAnalysis,
    IdentifiabilityParameter,
    IdentifiabilitySettings,
    analyze_identifiability,
)

from ..builder import BuiltProject
from ._shared import _write_csv


def write_identifiability_outputs(output: Path, result: Any, built: BuiltProject) -> list[str]:
    options = built.spec.identifiability_options
    if not options.get("enabled", False):
        return []
    if built.spec.fit_method != "jax_least_squares":
        return [
            "identifiability_analysis skipped because run.outputs.identifiability "
            "requires fit_method jax_least_squares"
        ]
    residuals = getattr(result, "final_residuals", None)
    jacobian = getattr(result, "final_jacobian", None)
    if result is None or residuals is None or jacobian is None:
        return ["identifiability_analysis skipped because least-squares residuals or Jacobian are unavailable"]

    settings = IdentifiabilitySettings.from_mapping(options)
    parameters = _identifiability_parameters(built, result)
    dataset_labels = _dataset_labels(output / "fit" / "residuals.csv")
    try:
        analysis = analyze_identifiability(
            parameters,
            residuals,
            jacobian,
            covariance=getattr(result, "covariance", None),
            dataset_labels=dataset_labels,
            settings=settings,
        )
    except ValueError as error:
        return [f"identifiability_analysis skipped because {error}"]

    directory = output / "identifiability_analysis"
    directory.mkdir(parents=True, exist_ok=True)
    _write_parameter_identifiability(directory / "parameter_identifiability.csv", analysis)
    _write_singular_values(directory / "singular_values.csv", analysis)
    _write_weak_modes(directory / "weak_modes.csv", analysis)
    _write_strong_correlation_pairs(
        directory / "strong_correlation_pairs.csv",
        analysis,
        threshold=settings.high_correlation_threshold,
    )
    _write_csv(directory / "dataset_sensitivity.csv", analysis.dataset_sensitivity_rows)
    plot_notes = _write_plots(directory, analysis)
    _write_summary(directory / "summary.md", built, result, analysis, plot_notes)
    return ["identifiability_analysis/summary.md written from scaled least-squares Jacobian", *plot_notes]


def _identifiability_parameters(
    built: BuiltProject,
    result: Any,
) -> tuple[IdentifiabilityParameter, ...]:
    best = dict(built.values)
    best.update(
        {
            name: float(value)
            for name, value in getattr(result, "best_parameters", {}).items()
        }
    )
    parameters = []
    for parameter in built.spec.varying_parameters():
        initial, lower, upper = parameter.require_bounds()
        parameters.append(
            IdentifiabilityParameter(
                name=parameter.name,
                initial=float(initial),
                lower=float(lower),
                upper=float(upper),
                best=float(best.get(parameter.name, initial)),
            )
        )
    return tuple(parameters)


def _dataset_labels(path: Path) -> tuple[str, ...] | None:
    if not path.is_file():
        return None
    with path.open("r", newline="", encoding="utf-8") as handle:
        return tuple(row["dataset"] for row in csv.DictReader(handle))


def _write_parameter_identifiability(path: Path, analysis: IdentifiabilityAnalysis) -> None:
    rows = [[
        "parameter",
        "initial",
        "best",
        "lower",
        "upper",
        "scaled_position",
        "bound_flag",
        "scaled_sensitivity_norm",
        "scaled_sensitivity_rms",
        "relative_sensitivity",
        "scaled_gradient",
        "abs_scaled_gradient",
        "stderr",
        "scaled_stderr_fraction_of_range",
        "max_abs_correlation",
        "weak_mode_participation",
        "suggestion",
    ]]
    for index, parameter in enumerate(analysis.parameters):
        rows.append([
            parameter.name,
            parameter.initial,
            parameter.best,
            parameter.lower,
            parameter.upper,
            parameter.scaled_position,
            analysis.active_bounds[index],
            analysis.scaled_sensitivity_norm[index],
            analysis.scaled_sensitivity_rms[index],
            analysis.relative_sensitivity[index],
            analysis.scaled_gradient[index],
            abs(analysis.scaled_gradient[index]),
            analysis.stderr[index],
            analysis.scaled_stderr[index],
            analysis.max_abs_correlation[index],
            analysis.weak_mode_participation[index],
            analysis.suggestions[index],
        ])
    _write_csv(path, rows)


def _write_singular_values(path: Path, analysis: IdentifiabilityAnalysis) -> None:
    singular_values = analysis.singular_values
    largest = singular_values[0] if singular_values.size else np.nan
    rows = [["index", "singular_value", "relative_to_largest", "condition_to_largest"]]
    for index, value in enumerate(singular_values, start=1):
        rows.append([
            index,
            value,
            value / largest if largest else np.nan,
            largest / value if value else np.inf,
        ])
    _write_csv(path, rows)


def _write_weak_modes(path: Path, analysis: IdentifiabilityAnalysis) -> None:
    rows = [[
        "mode_from_weakest",
        "singular_value_index",
        "singular_value",
        "relative_to_largest",
        "parameter_rank_in_mode",
        "parameter",
        "coefficient",
        "abs_coefficient",
    ]]
    singular_values = analysis.singular_values
    vt = analysis.right_singular_vectors
    if singular_values.size:
        largest = singular_values[0]
        for mode_from_weakest in range(1, analysis.weak_mode_count + 1):
            sv_index = len(singular_values) - mode_from_weakest
            vector = vt[sv_index]
            order = np.argsort(np.abs(vector))[::-1]
            for rank, parameter_index in enumerate(order[: min(8, len(analysis.names))], start=1):
                rows.append([
                    mode_from_weakest,
                    sv_index + 1,
                    singular_values[sv_index],
                    singular_values[sv_index] / largest if largest else np.nan,
                    rank,
                    analysis.names[parameter_index],
                    vector[parameter_index],
                    abs(vector[parameter_index]),
                ])
    _write_csv(path, rows)


def _write_strong_correlation_pairs(
    path: Path,
    analysis: IdentifiabilityAnalysis,
    *,
    threshold: float,
) -> None:
    rows = [["parameter_1", "parameter_2", "correlation", "abs_correlation"]]
    correlation = analysis.correlation
    if correlation is not None:
        pairs = []
        for i in range(len(analysis.names)):
            for j in range(i + 1, len(analysis.names)):
                value = float(correlation[i, j])
                if np.isfinite(value) and abs(value) >= threshold:
                    pairs.append((abs(value), analysis.names[i], analysis.names[j], value))
        for _abs_value, name_i, name_j, value in sorted(pairs, reverse=True):
            rows.append([name_i, name_j, value, abs(value)])
    _write_csv(path, rows)


def _write_summary(
    path: Path,
    built: BuiltProject,
    result: Any,
    analysis: IdentifiabilityAnalysis,
    plot_notes: list[str],
) -> None:
    names = list(analysis.names)
    low_order = np.argsort(analysis.relative_sensitivity)
    weak_order = np.argsort(analysis.weak_mode_participation)[::-1]
    uncertainty_order = np.argsort(np.nan_to_num(analysis.scaled_stderr, nan=-np.inf))[::-1]
    singular_values = analysis.singular_values
    objective = getattr(result, "final_cost", None)
    lines = [
        f"# SWANX Identifiability Analysis: {built.spec.name}",
        "",
        f"Residual count: {analysis.residuals.size}",
        f"Parameter count: {len(analysis.parameters)}",
        f"Least-squares objective: {objective if objective is not None else 'not available'}",
        f"Optimizer status: `{getattr(result, 'message', 'not available')}`",
        "",
        "## Scaled Jacobian Conditioning",
        "",
        f"Largest singular value: {_format_float(singular_values[0]) if singular_values.size else 'nan'}",
        f"Smallest singular value: {_format_float(singular_values[-1]) if singular_values.size else 'nan'}",
        f"Condition number: {_format_float(analysis.condition_number)}",
        "",
        "The Jacobian columns are scaled by each parameter range before this analysis.",
        "",
        "## Lowest Scaled Sensitivity",
        "",
        "| Parameter | Relative sensitivity | Scaled stderr | Max abs corr. | Weak-mode part. | Suggestion |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for index in low_order:
        lines.append(
            f"| {names[index]} | {analysis.relative_sensitivity[index]:.4g} | "
            f"{analysis.scaled_stderr[index]:.4g} | "
            f"{analysis.max_abs_correlation[index]:.4g} | "
            f"{analysis.weak_mode_participation[index]:.4g} | "
            f"{analysis.suggestions[index]} |"
        )

    lines.extend(["", "## Highest Weak-Mode Participation", ""])
    lines.extend(["| Parameter | Weak-mode participation | Relative sensitivity | Suggestion |", "|---|---:|---:|---|"])
    for index in weak_order:
        lines.append(
            f"| {names[index]} | {analysis.weak_mode_participation[index]:.4g} | "
            f"{analysis.relative_sensitivity[index]:.4g} | {analysis.suggestions[index]} |"
        )

    lines.extend(["", "## Largest Scaled Uncertainties", ""])
    lines.extend(["| Parameter | Scaled stderr | Relative sensitivity | Suggestion |", "|---|---:|---:|---|"])
    for index in uncertainty_order:
        lines.append(
            f"| {names[index]} | {analysis.scaled_stderr[index]:.4g} | "
            f"{analysis.relative_sensitivity[index]:.4g} | {analysis.suggestions[index]} |"
        )

    lines.extend(["", "## Near-Bound Parameters", ""])
    near_bound = [name for name, bound in zip(names, analysis.active_bounds) if bound]
    if near_bound:
        for name in near_bound:
            index = names.index(name)
            lines.append(
                f"- {name}: {analysis.active_bounds[index]}, scaled position "
                f"{analysis.parameters[index].scaled_position:.4g}, suggestion: "
                f"{analysis.suggestions[index]}"
            )
    else:
        lines.append("- None using the configured tolerance.")

    lines.extend(["", "## Strongest Correlation Pairs", ""])
    lines.extend(["| Parameter 1 | Parameter 2 | Correlation |", "|---|---|---:|"])
    pair_lines = _strongest_pair_lines(analysis, limit=10)
    lines.extend(pair_lines or ["| none available |  |  |"])

    lines.extend(["", "## Weakest SVD Modes", ""])
    if singular_values.size:
        for mode_from_weakest in range(1, analysis.weak_mode_count + 1):
            sv_index = len(singular_values) - mode_from_weakest
            vector = analysis.right_singular_vectors[sv_index]
            order = np.argsort(np.abs(vector))[::-1][:5]
            terms = ", ".join(f"{vector[index]:+.2f} {names[index]}" for index in order)
            relative = singular_values[sv_index] / singular_values[0] if singular_values[0] else np.nan
            lines.append(
                f"- Weak mode {mode_from_weakest}: singular value "
                f"{singular_values[sv_index]:.6g} ({relative:.3g} of largest): {terms}"
            )

    lines.extend([
        "",
        "## Dataset Sensitivity Caveat",
        "",
        "Dataset sensitivity uses the final weighted residual Jacobian, so it diagnoses "
        "the configured objective scaling and weights. A dominant dataset is a signal "
        "to audit weighting, not automatic proof that the physical data scaling is wrong.",
        "",
        "## Output Files",
        "",
        "- `parameter_identifiability.csv`: per-parameter sensitivity, uncertainty, bound, and suggestion table.",
        "- `dataset_sensitivity.csv`: per-dataset contribution to each parameter sensitivity.",
        "- `singular_values.csv` and `weak_modes.csv`: scaled-Jacobian SVD diagnostics.",
        "- `strong_correlation_pairs.csv`: parameter pairs above the configured correlation threshold.",
    ])
    for note in plot_notes:
        if " skipped " not in f" {note} ":
            lines.append(f"- `{Path(note).name}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _strongest_pair_lines(analysis: IdentifiabilityAnalysis, limit: int) -> list[str]:
    if analysis.correlation is None:
        return []
    pairs = []
    for i in range(len(analysis.names)):
        for j in range(i + 1, len(analysis.names)):
            value = float(analysis.correlation[i, j])
            if np.isfinite(value):
                pairs.append((abs(value), analysis.names[i], analysis.names[j], value))
    return [
        f"| {name_i} | {name_j} | {value:.4g} |"
        for _abs_value, name_i, name_j, value in sorted(pairs, reverse=True)[:limit]
    ]


def _write_plots(directory: Path, analysis: IdentifiabilityAnalysis) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return ["identifiability_analysis plots skipped because matplotlib is not installed"]
    notes = []
    _plot_scaled_sensitivity(directory / "scaled_sensitivity.png", plt, analysis)
    notes.append("identifiability_analysis/scaled_sensitivity.png")
    _plot_singular_values(directory / "singular_values.png", plt, analysis)
    notes.append("identifiability_analysis/singular_values.png")
    if analysis.correlation is not None:
        _plot_correlation_heatmap(directory / "correlation_heatmap.png", plt, analysis)
        notes.append("identifiability_analysis/correlation_heatmap.png")
    if analysis.singular_values.size and analysis.weak_mode_count:
        _plot_weak_modes(directory / "weak_modes.png", plt, analysis)
        notes.append("identifiability_analysis/weak_modes.png")
    if _plot_dataset_sensitivity(directory / "dataset_sensitivity_heatmap.png", plt, analysis):
        notes.append("identifiability_analysis/dataset_sensitivity_heatmap.png")
    plt.close("all")
    return notes


def _plot_scaled_sensitivity(path: Path, plt, analysis: IdentifiabilityAnalysis) -> None:
    order = np.argsort(analysis.relative_sensitivity)
    fig, ax = plt.subplots(figsize=(8.2, max(3.6, 0.35 * len(analysis.names) + 1.8)))
    y = np.arange(len(analysis.names))
    ax.barh(y, analysis.relative_sensitivity[order], color="#4477AA")
    ax.set_yticks(y, [analysis.names[index] for index in order], fontsize=8)
    ax.set_xlabel("Relative scaled sensitivity")
    ax.set_title("Parameter Sensitivity")
    ax.set_xlim(left=0.0)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_singular_values(path: Path, plt, analysis: IdentifiabilityAnalysis) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    indices = np.arange(1, analysis.singular_values.size + 1)
    plotted = np.where(analysis.singular_values > 0, analysis.singular_values, np.nan)
    ax.semilogy(indices, plotted, marker="o", linewidth=1.5, markersize=4)
    ax.set_xlabel("SVD component")
    ax.set_ylabel("Singular value")
    ax.set_title("Scaled Jacobian Singular Values")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_correlation_heatmap(path: Path, plt, analysis: IdentifiabilityAnalysis) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    image = ax.imshow(analysis.correlation, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(np.arange(len(analysis.names)), analysis.names, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(analysis.names)), analysis.names, fontsize=8)
    ax.set_title("Parameter Correlation")
    fig.colorbar(image, ax=ax, shrink=0.78, label="correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_weak_modes(path: Path, plt, analysis: IdentifiabilityAnalysis) -> None:
    count = min(analysis.weak_mode_count, analysis.right_singular_vectors.shape[0], 4)
    fig, axes = plt.subplots(count, 1, figsize=(8.8, max(2.2 * count, 3.0)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis_index, ax in enumerate(axes, start=1):
        sv_index = len(analysis.singular_values) - axis_index
        vector = analysis.right_singular_vectors[sv_index]
        order = np.argsort(np.abs(vector))[::-1][:8]
        labels = [analysis.names[index] for index in order][::-1]
        values = vector[order][::-1]
        colors = ["#CC6677" if value < 0 else "#228833" for value in values]
        ax.barh(np.arange(len(labels)), values, color=colors)
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"Weak mode {axis_index}: singular value {analysis.singular_values[sv_index]:.3g}")
        ax.grid(axis="x", alpha=0.2)
    axes[-1].set_xlabel("SVD coefficient in scaled parameter space")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_dataset_sensitivity(path: Path, plt, analysis: IdentifiabilityAnalysis) -> bool:
    rows = analysis.dataset_sensitivity_rows
    data_rows = [row for row in rows[1:] if row and row[0] != "note"]
    if not data_rows:
        return False
    datasets = list(dict.fromkeys(str(row[0]) for row in data_rows))
    matrix = np.zeros((len(datasets), len(analysis.names)), dtype=float)
    parameter_index = {name: index for index, name in enumerate(analysis.names)}
    dataset_index = {name: index for index, name in enumerate(datasets)}
    for dataset, _count, parameter, _norm, fraction in data_rows:
        matrix[dataset_index[str(dataset)], parameter_index[str(parameter)]] = float(fraction)
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(analysis.names)), analysis.names, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(datasets)), datasets, fontsize=8)
    ax.set_title("Fraction of Parameter Sensitivity by Dataset")
    fig.colorbar(image, ax=ax, shrink=0.78, label="fraction of column norm")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def _format_float(value: float) -> str:
    return "nan" if np.isnan(value) else f"{value:.6g}"


__all__ = ["write_identifiability_outputs"]
