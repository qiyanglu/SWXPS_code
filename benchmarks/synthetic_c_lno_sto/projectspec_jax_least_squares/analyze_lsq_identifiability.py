"""Post-fit sensitivity diagnostics for the synthetic ProjectSpec LS benchmark.

This script reads an existing ProjectSpec JAX least-squares output folder and
analyzes the final residual vector and Jacobian. It does not rerun the fit.

Default input:
    runs/<latest>/optimizer/least_squares/

Outputs:
    <run>/identifiability_analysis/*.csv
    <run>/identifiability_analysis/*.png
    <run>/identifiability_analysis/summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = SCRIPT_DIR / "runs"


@dataclass(frozen=True)
class ParameterInfo:
    name: str
    initial: float
    lower: float
    upper: float
    best: float

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def scaled_position(self) -> float:
        return (self.best - self.lower) / self.width


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve() if args.run_dir else latest_run_dir(DEFAULT_RUNS_DIR)
    fit_dir = run_dir / "fit"
    least_squares_dir = run_dir / "optimizer" / "least_squares"
    output_dir = (args.output_dir or (run_dir / "identifiability_analysis")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    parameters = read_parameters(fit_dir / "best_parameters.csv")
    residuals = read_vector(least_squares_dir / "residual_vector.csv", "residual")
    jacobian = read_matrix_triplets(least_squares_dir / "jacobian.csv")
    validate_shapes(jacobian, residuals, parameters)

    names = [parameter.name for parameter in parameters]
    widths = np.asarray([parameter.width for parameter in parameters], dtype=float)
    scaled_position = np.asarray([parameter.scaled_position for parameter in parameters])
    scaled_jacobian = jacobian * widths[np.newaxis, :]

    sensitivity = np.linalg.norm(scaled_jacobian, axis=0)
    sensitivity_rms = sensitivity / np.sqrt(max(residuals.size, 1))
    max_sensitivity = float(np.max(sensitivity)) if sensitivity.size else 1.0
    relative_sensitivity = sensitivity / max(max_sensitivity, np.finfo(float).tiny)
    scaled_gradient = scaled_jacobian.T @ residuals

    singular_values, vt = svd_or_empty(scaled_jacobian)
    condition = condition_number(singular_values)
    weak_mode_count = min(args.weak_modes, vt.shape[0])
    weak_participation = weak_mode_participation(vt, weak_mode_count, len(parameters))

    covariance = read_optional_matrix(least_squares_dir / "covariance.csv")
    correlation = read_optional_matrix(least_squares_dir / "correlation.csv")
    stderr = covariance_stderr(covariance, len(parameters))
    scaled_stderr = stderr / widths if stderr is not None else np.full(len(parameters), np.nan)
    max_abs_correlation = max_abs_correlations(correlation, len(parameters))

    active_bounds = classify_bounds(scaled_position, args.active_bound_tol)
    suggestions = suggest_actions(
        relative_sensitivity=relative_sensitivity,
        scaled_stderr=scaled_stderr,
        max_abs_correlation=max_abs_correlation,
        weak_participation=weak_participation,
        active_bounds=active_bounds,
        low_sensitivity_threshold=args.low_sensitivity_threshold,
        high_uncertainty_threshold=args.high_uncertainty_threshold,
        high_correlation_threshold=args.high_correlation_threshold,
        high_weak_participation_threshold=args.high_weak_participation_threshold,
    )

    write_parameter_summary(
        output_dir / "parameter_identifiability.csv",
        parameters,
        scaled_position,
        sensitivity,
        sensitivity_rms,
        relative_sensitivity,
        scaled_gradient,
        stderr,
        scaled_stderr,
        max_abs_correlation,
        weak_participation,
        active_bounds,
        suggestions,
    )
    write_singular_values(output_dir / "singular_values.csv", singular_values)
    write_weak_modes(output_dir / "weak_modes.csv", singular_values, vt, names, weak_mode_count)
    write_correlation_pairs(
        output_dir / "strong_correlation_pairs.csv",
        correlation,
        names,
        threshold=args.high_correlation_threshold,
    )
    write_dataset_sensitivity(
        output_dir / "dataset_sensitivity.csv",
        fit_dir / "residuals.csv",
        scaled_jacobian,
        sensitivity,
        names,
    )
    plot_notes = write_plots(
        output_dir,
        fit_dir / "residuals.csv",
        names,
        scaled_jacobian,
        sensitivity,
        relative_sensitivity,
        singular_values,
        correlation,
        vt,
        weak_mode_count,
    )
    write_summary(
        output_dir / "summary.md",
        run_dir,
        read_status(least_squares_dir / "status.json"),
        residuals,
        parameters,
        singular_values,
        condition,
        relative_sensitivity,
        scaled_stderr,
        max_abs_correlation,
        weak_participation,
        active_bounds,
        suggestions,
        correlation,
        vt,
        weak_mode_count,
        plot_notes,
    )

    print(f"Input run: {run_dir}")
    print(f"Output:    {output_dir}")
    print(f"Residuals / parameters: {residuals.size} / {len(parameters)}")
    print(f"Scaled Jacobian condition number: {condition:.6g}")
    print("Lowest relative scaled sensitivities:")
    for index in np.argsort(relative_sensitivity)[: min(args.print_count, len(parameters))]:
        print(
            f"  {names[index]}: {relative_sensitivity[index]:.6g} "
            f"({suggestions[index]})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="ProjectSpec run directory. Defaults to newest run under ./runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run-dir>/identifiability_analysis.",
    )
    parser.add_argument("--weak-modes", type=int, default=5)
    parser.add_argument("--print-count", type=int, default=5)
    parser.add_argument("--active-bound-tol", type=float, default=0.02)
    parser.add_argument("--low-sensitivity-threshold", type=float, default=0.05)
    parser.add_argument("--high-uncertainty-threshold", type=float, default=0.50)
    parser.add_argument("--high-correlation-threshold", type=float, default=0.90)
    parser.add_argument("--high-weak-participation-threshold", type=float, default=0.50)
    return parser.parse_args()


def latest_run_dir(runs_dir: Path) -> Path:
    runs = [
        path
        for path in runs_dir.iterdir()
        if (path / "optimizer" / "least_squares" / "jacobian.csv").is_file()
    ]
    if not runs:
        raise FileNotFoundError(f"No least-squares run directories found under {runs_dir}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def read_parameters(path: Path) -> list[ParameterInfo]:
    rows = read_dict_rows(path)
    return [
        ParameterInfo(
            name=row["name"],
            initial=float(row["initial"]),
            lower=float(row["lower"]),
            upper=float(row["upper"]),
            best=float(row["best_value"]),
        )
        for row in rows
    ]


def read_dict_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_vector(path: Path, column: str) -> np.ndarray:
    rows = read_dict_rows(path)
    return np.asarray([float(row[column]) for row in rows], dtype=float)


def read_matrix_triplets(path: Path) -> np.ndarray:
    rows = read_dict_rows(path)
    if not rows:
        return np.empty((0, 0), dtype=float)
    row_count = max(int(row["row"]) for row in rows) + 1
    column_count = max(int(row["column"]) for row in rows) + 1
    matrix = np.zeros((row_count, column_count), dtype=float)
    for row in rows:
        matrix[int(row["row"]), int(row["column"])] = float(row["value"])
    return matrix


def read_optional_matrix(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    matrix = read_matrix_triplets(path)
    return matrix if matrix.size else None


def read_status(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_shapes(
    jacobian: np.ndarray,
    residuals: np.ndarray,
    parameters: list[ParameterInfo],
) -> None:
    expected = (residuals.size, len(parameters))
    if jacobian.shape != expected:
        raise ValueError(f"Jacobian shape {jacobian.shape} does not match {expected}")
    if any(parameter.width <= 0.0 for parameter in parameters):
        raise ValueError("All fitted parameter ranges must have positive width")


def svd_or_empty(scaled_jacobian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if scaled_jacobian.size == 0:
        return np.array([], dtype=float), np.empty((0, 0), dtype=float)
    _u, singular_values, vt = np.linalg.svd(scaled_jacobian, full_matrices=False)
    return singular_values, vt


def condition_number(singular_values: np.ndarray) -> float:
    if singular_values.size == 0:
        return float("nan")
    if singular_values[-1] <= 0.0:
        return float("inf")
    return float(singular_values[0] / singular_values[-1])


def weak_mode_participation(vt: np.ndarray, weak_mode_count: int, parameter_count: int) -> np.ndarray:
    if weak_mode_count <= 0:
        return np.zeros(parameter_count, dtype=float)
    weak_vt = vt[-weak_mode_count:, :]
    return np.sqrt(np.sum(weak_vt**2, axis=0))


def covariance_stderr(covariance: np.ndarray | None, parameter_count: int) -> np.ndarray | None:
    if covariance is None:
        return None
    if covariance.shape != (parameter_count, parameter_count):
        raise ValueError("Covariance shape does not match parameter count")
    return np.sqrt(np.maximum(np.diag(covariance), 0.0))


def max_abs_correlations(correlation: np.ndarray | None, parameter_count: int) -> np.ndarray:
    if correlation is None:
        return np.full(parameter_count, np.nan)
    result = np.full(parameter_count, np.nan)
    for index in range(parameter_count):
        values = np.delete(np.abs(correlation[index]), index)
        result[index] = float(np.nanmax(values)) if values.size else np.nan
    return result


def classify_bounds(scaled_position: np.ndarray, active_tol: float) -> list[str]:
    labels = []
    for value in scaled_position:
        if value <= active_tol:
            labels.append("near_lower")
        elif value >= 1.0 - active_tol:
            labels.append("near_upper")
        else:
            labels.append("")
    return labels


def suggest_actions(
    *,
    relative_sensitivity: np.ndarray,
    scaled_stderr: np.ndarray,
    max_abs_correlation: np.ndarray,
    weak_participation: np.ndarray,
    active_bounds: list[str],
    low_sensitivity_threshold: float,
    high_uncertainty_threshold: float,
    high_correlation_threshold: float,
    high_weak_participation_threshold: float,
) -> list[str]:
    suggestions = []
    for index in range(relative_sensitivity.size):
        flags = []
        if relative_sensitivity[index] < low_sensitivity_threshold:
            flags.append("low_sensitivity")
        if np.isfinite(scaled_stderr[index]) and scaled_stderr[index] > high_uncertainty_threshold:
            flags.append("high_uncertainty")
        if np.isfinite(max_abs_correlation[index]) and max_abs_correlation[index] > high_correlation_threshold:
            flags.append("high_correlation")
        if weak_participation[index] > high_weak_participation_threshold:
            flags.append("weak_svd_mode")
        if active_bounds[index]:
            flags.append(active_bounds[index])

        if "low_sensitivity" in flags and "weak_svd_mode" in flags:
            action = "fix_or_profile_candidate"
        elif "high_correlation" in flags and "weak_svd_mode" in flags:
            action = "tie_or_reparameterize_candidate"
        elif active_bounds[index]:
            action = "review_bound_or_model"
        elif "high_uncertainty" in flags or "weak_svd_mode" in flags:
            action = "needs_profile_check"
        else:
            action = "keep_for_now"
        suggestions.append(action + ("; " + "; ".join(flags) if flags else ""))
    return suggestions


def write_parameter_summary(
    path: Path,
    parameters: list[ParameterInfo],
    scaled_position: np.ndarray,
    sensitivity: np.ndarray,
    sensitivity_rms: np.ndarray,
    relative_sensitivity: np.ndarray,
    scaled_gradient: np.ndarray,
    stderr: np.ndarray | None,
    scaled_stderr: np.ndarray,
    max_abs_correlation: np.ndarray,
    weak_participation: np.ndarray,
    active_bounds: list[str],
    suggestions: list[str],
) -> None:
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
    for index, parameter in enumerate(parameters):
        rows.append([
            parameter.name,
            parameter.initial,
            parameter.best,
            parameter.lower,
            parameter.upper,
            scaled_position[index],
            active_bounds[index],
            sensitivity[index],
            sensitivity_rms[index],
            relative_sensitivity[index],
            scaled_gradient[index],
            abs(scaled_gradient[index]),
            "" if stderr is None else stderr[index],
            scaled_stderr[index],
            max_abs_correlation[index],
            weak_participation[index],
            suggestions[index],
        ])
    write_rows(path, rows)


def write_singular_values(path: Path, singular_values: np.ndarray) -> None:
    largest = singular_values[0] if singular_values.size else np.nan
    rows = [["index", "singular_value", "relative_to_largest", "condition_to_largest"]]
    for index, value in enumerate(singular_values, start=1):
        rows.append([
            index,
            value,
            value / largest if largest else np.nan,
            largest / value if value else np.inf,
        ])
    write_rows(path, rows)


def write_weak_modes(
    path: Path,
    singular_values: np.ndarray,
    vt: np.ndarray,
    names: list[str],
    weak_mode_count: int,
) -> None:
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
    if not singular_values.size:
        write_rows(path, rows)
        return
    largest = singular_values[0]
    for mode_from_weakest in range(1, weak_mode_count + 1):
        sv_index = len(singular_values) - mode_from_weakest
        vector = vt[sv_index]
        order = np.argsort(np.abs(vector))[::-1]
        for rank, parameter_index in enumerate(order[: min(8, len(names))], start=1):
            rows.append([
                mode_from_weakest,
                sv_index + 1,
                singular_values[sv_index],
                singular_values[sv_index] / largest,
                rank,
                names[parameter_index],
                vector[parameter_index],
                abs(vector[parameter_index]),
            ])
    write_rows(path, rows)


def write_correlation_pairs(
    path: Path,
    correlation: np.ndarray | None,
    names: list[str],
    threshold: float,
) -> None:
    rows = [["parameter_1", "parameter_2", "correlation", "abs_correlation"]]
    if correlation is not None:
        pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                value = float(correlation[i, j])
                if abs(value) >= threshold:
                    pairs.append((abs(value), names[i], names[j], value))
        for _abs_value, name_i, name_j, value in sorted(pairs, reverse=True):
            rows.append([name_i, name_j, value, abs(value)])
    write_rows(path, rows)


def write_dataset_sensitivity(
    path: Path,
    residual_rows_path: Path,
    scaled_jacobian: np.ndarray,
    total_sensitivity: np.ndarray,
    names: list[str],
) -> None:
    residual_rows = read_dict_rows(residual_rows_path)
    if len(residual_rows) != scaled_jacobian.shape[0]:
        write_rows(path, [["note"], ["fit/residuals.csv length does not match residual vector length"]])
        return
    datasets = [row["dataset"] for row in residual_rows]
    rows = [[
        "dataset",
        "residual_count",
        "parameter",
        "scaled_sensitivity_norm",
        "fraction_of_parameter_sensitivity",
    ]]
    for dataset in dict.fromkeys(datasets):
        mask = np.asarray([value == dataset for value in datasets], dtype=bool)
        block = scaled_jacobian[mask, :]
        block_sensitivity = np.linalg.norm(block, axis=0)
        fraction = np.divide(
            block_sensitivity,
            total_sensitivity,
            out=np.zeros_like(block_sensitivity),
            where=total_sensitivity > 0,
        )
        for index, name in enumerate(names):
            rows.append([dataset, int(mask.sum()), name, block_sensitivity[index], fraction[index]])
    write_rows(path, rows)


def write_summary(
    path: Path,
    run_dir: Path,
    status: dict[str, object],
    residuals: np.ndarray,
    parameters: list[ParameterInfo],
    singular_values: np.ndarray,
    condition: float,
    relative_sensitivity: np.ndarray,
    scaled_stderr: np.ndarray,
    max_abs_correlation: np.ndarray,
    weak_participation: np.ndarray,
    active_bounds: list[str],
    suggestions: list[str],
    correlation: np.ndarray | None,
    vt: np.ndarray,
    weak_mode_count: int,
    plot_notes: list[str],
) -> None:
    names = [parameter.name for parameter in parameters]
    low_order = np.argsort(relative_sensitivity)
    weak_order = np.argsort(weak_participation)[::-1]
    uncertainty_order = np.argsort(np.nan_to_num(scaled_stderr, nan=-np.inf))[::-1]

    lines = [
        "# Synthetic C/LNO/STO LS Identifiability Analysis",
        "",
        f"Input run: `{relative_display_path(run_dir)}`",
        f"Residual count: {residuals.size}",
        f"Parameter count: {len(parameters)}",
        f"Least-squares objective: {status.get('objective', 'unknown')}",
        f"Optimizer status: {status.get('message', 'unknown')}",
        "",
        "## Scaled Jacobian Conditioning",
        "",
        f"Largest singular value: {format_float(singular_values[0])}",
        f"Smallest singular value: {format_float(singular_values[-1])}",
        f"Condition number: {format_float(condition)}",
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
            f"| {names[index]} | {relative_sensitivity[index]:.4g} | "
            f"{scaled_stderr[index]:.4g} | {max_abs_correlation[index]:.4g} | "
            f"{weak_participation[index]:.4g} | {suggestions[index]} |"
        )

    lines.extend(["", "## Highest Weak-Mode Participation", ""])
    lines.extend(["| Parameter | Weak-mode participation | Relative sensitivity | Suggestion |", "|---|---:|---:|---|"])
    for index in weak_order:
        lines.append(
            f"| {names[index]} | {weak_participation[index]:.4g} | "
            f"{relative_sensitivity[index]:.4g} | {suggestions[index]} |"
        )

    lines.extend(["", "## Largest Scaled Uncertainties", ""])
    lines.extend(["| Parameter | Scaled stderr | Relative sensitivity | Suggestion |", "|---|---:|---:|---|"])
    for index in uncertainty_order:
        lines.append(
            f"| {names[index]} | {scaled_stderr[index]:.4g} | "
            f"{relative_sensitivity[index]:.4g} | {suggestions[index]} |"
        )

    near_bound = [names[index] for index, value in enumerate(active_bounds) if value]
    lines.extend(["", "## Near-Bound Parameters", ""])
    if near_bound:
        for name in near_bound:
            index = names.index(name)
            lines.append(
                f"- {name}: {active_bounds[index]}, scaled position "
                f"{parameters[index].scaled_position:.4g}, suggestion: {suggestions[index]}"
            )
    else:
        lines.append("- None using the configured tolerance.")

    lines.extend(["", "## Strongest Correlation Pairs", ""])
    lines.extend(["| Parameter 1 | Parameter 2 | Correlation |", "|---|---|---:|"])
    pair_lines = strongest_pairs(correlation, names, limit=10)
    lines.extend(pair_lines or ["| none available |  |  |"])

    lines.extend(["", "## Weakest SVD Modes", ""])
    if singular_values.size:
        for mode_from_weakest in range(1, weak_mode_count + 1):
            sv_index = len(singular_values) - mode_from_weakest
            vector = vt[sv_index]
            order = np.argsort(np.abs(vector))[::-1][:5]
            terms = ", ".join(f"{vector[index]:+.2f} {names[index]}" for index in order)
            lines.append(
                f"- Weak mode {mode_from_weakest}: singular value "
                f"{singular_values[sv_index]:.6g} "
                f"({singular_values[sv_index] / singular_values[0]:.3g} of largest): {terms}"
            )

    lines.extend(["", "## Output Files", ""])
    lines.extend([
        "- `parameter_identifiability.csv`: per-parameter sensitivity, uncertainty, bound, and suggestion table.",
        "- `dataset_sensitivity.csv`: per-dataset contribution to each parameter sensitivity.",
        "- `singular_values.csv` and `weak_modes.csv`: scaled-Jacobian SVD diagnostics.",
        "- `strong_correlation_pairs.csv`: parameter pairs above the configured correlation threshold.",
    ])
    for note in plot_notes:
        lines.append(f"- `{note}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def relative_display_path(path: Path) -> str:
    resolved = path.resolve()
    for base in (Path.cwd().resolve(), SCRIPT_DIR):
        try:
            return str(resolved.relative_to(base))
        except ValueError:
            continue
    return str(path)


def format_float(value: float) -> str:
    return "nan" if np.isnan(value) else f"{value:.6g}"


def strongest_pairs(correlation: np.ndarray | None, names: list[str], limit: int) -> list[str]:
    if correlation is None:
        return []
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            value = float(correlation[i, j])
            pairs.append((abs(value), names[i], names[j], value))
    return [
        f"| {name_i} | {name_j} | {value:.4g} |"
        for _abs_value, name_i, name_j, value in sorted(pairs, reverse=True)[:limit]
    ]


def write_plots(
    output_dir: Path,
    residual_rows_path: Path,
    names: list[str],
    scaled_jacobian: np.ndarray,
    sensitivity: np.ndarray,
    relative_sensitivity: np.ndarray,
    singular_values: np.ndarray,
    correlation: np.ndarray | None,
    vt: np.ndarray,
    weak_mode_count: int,
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return ["plots skipped because matplotlib is not installed"]

    notes = []
    plot_scaled_sensitivity(output_dir / "scaled_sensitivity.png", plt, names, relative_sensitivity)
    notes.append("scaled_sensitivity.png")
    plot_singular_values(output_dir / "singular_values.png", plt, singular_values)
    notes.append("singular_values.png")
    if correlation is not None:
        plot_correlation_heatmap(output_dir / "correlation_heatmap.png", plt, names, correlation)
        notes.append("correlation_heatmap.png")
    if singular_values.size and weak_mode_count:
        plot_weak_modes(output_dir / "weak_modes.png", plt, names, singular_values, vt, weak_mode_count)
        notes.append("weak_modes.png")
    if plot_dataset_sensitivity(
        output_dir / "dataset_sensitivity_heatmap.png",
        plt,
        residual_rows_path,
        names,
        scaled_jacobian,
        sensitivity,
    ):
        notes.append("dataset_sensitivity_heatmap.png")
    plt.close("all")
    return notes


def plot_scaled_sensitivity(path: Path, plt, names: list[str], relative_sensitivity: np.ndarray) -> None:
    order = np.argsort(relative_sensitivity)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    y = np.arange(len(names))
    ax.barh(y, relative_sensitivity[order], color="#4477AA")
    ax.set_yticks(y, [names[index] for index in order], fontsize=8)
    ax.set_xlabel("Relative scaled sensitivity")
    ax.set_title("Synthetic C/LNO/STO Parameter Sensitivity")
    ax.set_xlim(left=0.0)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_singular_values(path: Path, plt, singular_values: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    indices = np.arange(1, singular_values.size + 1)
    ax.semilogy(indices, singular_values, marker="o", linewidth=1.5, markersize=4)
    ax.set_xlabel("SVD component")
    ax.set_ylabel("Singular value")
    ax.set_title("Scaled Jacobian Singular Values")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_correlation_heatmap(path: Path, plt, names: list[str], correlation: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    image = ax.imshow(correlation, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(np.arange(len(names)), names, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(names)), names, fontsize=8)
    ax.set_title("Parameter Correlation")
    fig.colorbar(image, ax=ax, shrink=0.78, label="correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_weak_modes(
    path: Path,
    plt,
    names: list[str],
    singular_values: np.ndarray,
    vt: np.ndarray,
    weak_mode_count: int,
) -> None:
    count = min(weak_mode_count, vt.shape[0], 4)
    fig, axes = plt.subplots(count, 1, figsize=(8.8, max(2.2 * count, 3.0)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis_index, ax in enumerate(axes, start=1):
        sv_index = len(singular_values) - axis_index
        vector = vt[sv_index]
        order = np.argsort(np.abs(vector))[::-1][:8]
        labels = [names[index] for index in order][::-1]
        values = vector[order][::-1]
        colors = ["#CC6677" if value < 0 else "#228833" for value in values]
        ax.barh(np.arange(len(labels)), values, color=colors)
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"Weak mode {axis_index}: singular value {singular_values[sv_index]:.3g}")
        ax.grid(axis="x", alpha=0.2)
    axes[-1].set_xlabel("SVD coefficient in scaled parameter space")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_dataset_sensitivity(
    path: Path,
    plt,
    residual_rows_path: Path,
    names: list[str],
    scaled_jacobian: np.ndarray,
    total_sensitivity: np.ndarray,
) -> bool:
    residual_rows = read_dict_rows(residual_rows_path)
    if len(residual_rows) != scaled_jacobian.shape[0]:
        return False
    datasets = [row["dataset"] for row in residual_rows]
    dataset_names = list(dict.fromkeys(datasets))
    matrix = np.zeros((len(dataset_names), len(names)), dtype=float)
    for row_index, dataset in enumerate(dataset_names):
        mask = np.asarray([value == dataset for value in datasets], dtype=bool)
        block_sensitivity = np.linalg.norm(scaled_jacobian[mask, :], axis=0)
        matrix[row_index] = np.divide(
            block_sensitivity,
            total_sensitivity,
            out=np.zeros_like(block_sensitivity),
            where=total_sensitivity > 0,
        )
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(names)), names, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(dataset_names)), dataset_names, fontsize=8)
    ax.set_title("Fraction of Parameter Sensitivity by Dataset")
    fig.colorbar(image, ax=ax, shrink=0.78, label="fraction of column norm")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def write_rows(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
