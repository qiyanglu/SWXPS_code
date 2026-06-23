"""Fit the synthetic C/LNO/STO case with fixed-grid JAX TRF least squares."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
for path in (SRC_DIR, CASE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fit_reflectivity_rc_bo as case  # noqa: E402
from swanx import (  # noqa: E402
    JaxLeastSquaresOptimizerSettings,
    JaxLeastSquaresResidualSettings,
    LayerSlicingPolicy,
    build_jax_residual_function,
    fixed_layer_grid,
    fixed_layer_grid_plan,
    optimize_with_jax_least_squares,
)
from swanx.diagnostics import (  # noqa: E402
    diagnostics_from_least_squares_result,
    plot_correlation_matrix,
    plot_parameter_estimates,
)
from swanx.reflectivity_jax import (  # noqa: E402
    transfer_matrix_field_intensity_jax,
    transfer_matrix_reflectivity_jax,
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "runs" / "synthetic_c_lno_sto" / "unified_jax_least_squares"
)

BAD_START = {
    "carbon_thickness": 15.5,
    "carbon_roughness_fraction": 0.90,
    "lno_thickness": 21.8,
    "sto_thickness": 18.2,
    "superlattice_roughness": 4.8,
    "substrate_roughness": 4.8,
    "angle_offset": 0.22,
}


class UnifiedSyntheticJaxModel:
    """Fully JAX-traceable fixed-grid model for the synthetic benchmark."""

    def __init__(self, angles, offpeak_mask, core_levels, plan, nominal_stack):
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        self.jax = jax
        self.jnp = jnp
        self.angles = jnp.asarray(angles, dtype=jnp.float64)
        self.offpeak_mask = jnp.asarray(offpeak_mask, dtype=bool)
        self.parameter_index = {
            parameter.name: index for index, parameter in enumerate(case.PARAMETERS)
        }
        capacity_grid = fixed_layer_grid(nominal_stack.optical_layers, plan)
        nominal_index = np.asarray(capacity_grid.nominal_layer_index, dtype=np.int32)
        count_by_layer = np.asarray(plan.slice_counts, dtype=float)
        self.cell_nominal_index = jnp.asarray(nominal_index, dtype=jnp.int32)
        self.cell_slice_count = jnp.asarray(
            count_by_layer[nominal_index - 1], dtype=jnp.float64
        )
        self.effective_layer_index = jnp.asarray(
            capacity_grid.effective_layer_index, dtype=jnp.int32
        )
        optical_layers = nominal_stack.optical_layers
        self.nominal_delta = jnp.asarray(
            [layer.delta for layer in optical_layers], dtype=jnp.float64
        )
        self.nominal_beta = jnp.asarray(
            [layer.beta for layer in optical_layers], dtype=jnp.float64
        )
        self.core_inputs = tuple(
            self._core_inputs(core, nominal_stack.materials) for core in core_levels
        )

    def _core_inputs(self, core, materials):
        concentration = np.asarray(
            [core.concentration_by_material.get(material, 0.0) for material in materials],
            dtype=float,
        )
        imfp = np.asarray(
            [core.imfp_by_material[material] for material in materials],
            dtype=float,
        )
        return (
            self.jnp.asarray(concentration, dtype=self.jnp.float64),
            self.jnp.asarray(1.0 / imfp, dtype=self.jnp.float64),
            float(core.emission_angle_deg),
        )

    def simulate_curves(self, physical_vector):
        """Return reflectivity and four normalized RC arrays."""

        jnp = self.jnp
        vector = jnp.asarray(physical_vector, dtype=jnp.float64)
        carbon = vector[self.parameter_index["carbon_thickness"]]
        fraction = vector[self.parameter_index["carbon_roughness_fraction"]]
        lno = vector[self.parameter_index["lno_thickness"]]
        sto = vector[self.parameter_index["sto_thickness"]]
        super_roughness = vector[self.parameter_index["superlattice_roughness"]]
        substrate_roughness = vector[
            self.parameter_index["substrate_roughness"]
        ]
        angle_offset = vector[self.parameter_index["angle_offset"]]

        period = jnp.stack((lno, sto))
        finite_thicknesses = jnp.concatenate(
            (jnp.reshape(carbon, (1,)), jnp.tile(period, case.SUPERLATTICE_REPEATS))
        )
        nominal_thicknesses = jnp.concatenate(
            (jnp.zeros((1,)), finite_thicknesses, jnp.zeros((1,)))
        )
        carbon_roughness = 1.0 + fraction * (jnp.minimum(5.0, carbon) - 1.0)
        finite_roughness = jnp.concatenate(
            (
                jnp.reshape(carbon_roughness, (1,)),
                jnp.full((2 * case.SUPERLATTICE_REPEATS,), super_roughness),
            )
        )
        nominal_roughness = jnp.concatenate(
            (
                jnp.zeros((1,)),
                finite_roughness,
                jnp.reshape(substrate_roughness, (1,)),
            )
        )

        widths = (
            finite_thicknesses[self.cell_nominal_index - 1]
            / self.cell_slice_count
        )
        centers = jnp.cumsum(widths) - 0.5 * widths
        boundaries = jnp.concatenate(
            (jnp.zeros((1,)), jnp.cumsum(finite_thicknesses))
        )
        cell_delta = self._graded_optical_property(
            centers, boundaries, nominal_roughness, self.nominal_delta
        )
        cell_beta = self._graded_optical_property(
            centers, boundaries, nominal_roughness, self.nominal_beta
        )
        effective_thicknesses = jnp.concatenate(
            (jnp.zeros((1,)), widths, jnp.zeros((1,)))
        )
        effective_delta = jnp.concatenate(
            (self.nominal_delta[:1], cell_delta, self.nominal_delta[-1:])
        )
        effective_beta = jnp.concatenate(
            (self.nominal_beta[:1], cell_beta, self.nominal_beta[-1:])
        )
        calculation_angles = self.angles + angle_offset
        reflectivity = transfer_matrix_reflectivity_jax(
            calculation_angles,
            case.PHOTON_ENERGY_EV,
            effective_thicknesses,
            effective_delta,
            effective_beta,
        )
        field_intensity = transfer_matrix_field_intensity_jax(
            calculation_angles,
            case.PHOTON_ENERGY_EV,
            effective_thicknesses,
            effective_delta,
            effective_beta,
            centers,
            self.effective_layer_index,
        )
        curves = tuple(
            self._normalized_curve(
                field_intensity,
                widths,
                centers,
                boundaries,
                nominal_roughness,
                concentration,
                attenuation_coefficient,
                emission_angle,
            )
            for concentration, attenuation_coefficient, emission_angle in self.core_inputs
        )
        return reflectivity, curves

    def _graded_optical_property(
        self, centers, boundaries, roughnesses, nominal_values
    ):
        """Match the validated nearest-interface optical grading rule."""

        jnp = self.jnp
        distances = centers[:, None] - boundaries[None, :]
        nearest = jnp.argmin(jnp.abs(distances), axis=1)
        distance = distances[jnp.arange(centers.size), nearest]
        sigma = roughnesses[nearest + 1]
        safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)
        fraction = 0.5 * (
            1.0 + self.jax.lax.erf(distance / (jnp.sqrt(2.0) * safe_sigma))
        )
        mixed = (
            (1.0 - fraction) * nominal_values[nearest]
            + fraction * nominal_values[nearest + 1]
        )
        base = nominal_values[self.cell_nominal_index]
        active = (sigma > 0.0) & (jnp.abs(distance) <= 4.0 * sigma)
        return jnp.where(active, mixed, base)

    def _graded_xps_property(
        self, centers, boundaries, roughnesses, nominal_values
    ):
        """Match the maintained sequential XPS property-grading rule."""

        jnp = self.jnp
        values = nominal_values[self.cell_nominal_index]
        for interface_index in range(len(nominal_values) - 1):
            sigma = roughnesses[interface_index + 1]
            safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)
            distance = centers - boundaries[interface_index]
            fraction = 0.5 * (
                1.0
                + self.jax.lax.erf(distance / (jnp.sqrt(2.0) * safe_sigma))
            )
            mixed = (
                (1.0 - fraction) * nominal_values[interface_index]
                + fraction * nominal_values[interface_index + 1]
            )
            adjacent = (self.cell_nominal_index == interface_index) | (
                self.cell_nominal_index == interface_index + 1
            )
            active = (
                (sigma > 0.0)
                & (jnp.abs(distance) <= 4.0 * sigma)
                & adjacent
            )
            values = jnp.where(active, mixed, values)
        return values

    def _normalized_curve(
        self,
        field_intensity,
        widths,
        centers,
        boundaries,
        roughnesses,
        nominal_concentration,
        nominal_attenuation_coefficient,
        emission_angle,
    ):
        del centers
        jnp = self.jnp
        concentration = self._graded_xps_property(
            jnp.cumsum(widths) - 0.5 * widths,
            boundaries,
            roughnesses,
            nominal_concentration,
        )
        attenuation_coefficient = self._graded_xps_property(
            jnp.cumsum(widths) - 0.5 * widths,
            boundaries,
            roughnesses,
            nominal_attenuation_coefficient,
        )
        cos_alpha = jnp.cos(jnp.deg2rad(emission_angle))
        cell_optical_depth = widths * attenuation_coefficient / cos_alpha
        optical_depth = jnp.cumsum(cell_optical_depth) - 0.5 * cell_optical_depth
        weights = concentration * jnp.exp(-optical_depth) * widths
        raw = jnp.sum(field_intensity * weights[:, None], axis=0)
        normalization = jnp.sum(jnp.where(self.offpeak_mask, raw, 0.0)) / jnp.sum(
            self.offpeak_mask
        )
        return raw / normalization


def capacity_values() -> dict[str, float]:
    values = dict(case.TRUE_VALUES)
    for parameter in case.PARAMETERS:
        if parameter.name in {"carbon_thickness", "lno_thickness", "sto_thickness"}:
            values[parameter.name] = parameter.upper
    return values


def bad_initial_vector() -> np.ndarray:
    return np.asarray([BAD_START[item.name] for item in case.PARAMETERS], dtype=float)


def values_from_vector(vector: np.ndarray) -> dict[str, float]:
    return {
        parameter.name: float(value)
        for parameter, value in zip(case.PARAMETERS, vector)
    }


def run_fit(args: argparse.Namespace) -> dict[str, object]:
    """Build, compile, run, validate, and save one synthetic TRF fit."""

    data = case.load_data(case.DATA_FILE, stride=args.stride)
    reflectivity_weight, _ = case.resolve_reflectivity_weight(data, "auto")
    base_problem = case.make_problem(data, reflectivity_weight)
    capacity_stack = case.build_stack(capacity_values())
    policy = LayerSlicingPolicy(
        min_slices=args.min_slices,
        max_slice_thickness=args.max_slice_thickness,
    )
    plan = fixed_layer_grid_plan(capacity_stack.optical_layers, policy)
    problem = replace(base_problem, slicing=plan)
    angles = data["angle_deg"]
    peak_angle = angles[np.argmax(data["reflectivity"])]
    offpeak_mask = np.abs(angles - peak_angle) > 1.25
    model = UnifiedSyntheticJaxModel(
        angles,
        offpeak_mask,
        problem.core_levels,
        plan,
        capacity_stack,
    )
    residual_function = build_jax_residual_function(
        model.simulate_curves,
        reflectivity=problem.reflectivity,
        rocking_curves=problem.rocking_curves,
        settings=JaxLeastSquaresResidualSettings(
            reflectivity_log=True,
            rocking_curve_normalization="mean_absolute",
        ),
    )
    initial = bad_initial_vector()
    compile_start = perf_counter()
    initial_residuals = residual_function(initial)
    initial_jacobian = residual_function.jacobian(initial)
    compile_seconds = perf_counter() - compile_start
    initial_cost = 0.5 * float(initial_residuals @ initial_residuals)
    counts_after_warmup = (
        residual_function.compilation_counter.residual_compilations,
        residual_function.compilation_counter.jacobian_compilations,
    )

    result = optimize_with_jax_least_squares(
        case.PARAMETERS,
        residual_function,
        initial=initial,
        settings=JaxLeastSquaresOptimizerSettings(
            max_nfev=args.max_nfev,
            ftol=args.ftol,
            xtol=args.xtol,
            gtol=args.gtol,
            estimate_covariance=True,
        ),
    )
    counts_after_fit = (
        residual_function.compilation_counter.residual_compilations,
        residual_function.compilation_counter.jacobian_compilations,
    )
    if counts_after_warmup != (1, 1) or counts_after_fit != (1, 1):
        raise RuntimeError(
            f"unexpected compilation counts: warmup={counts_after_warmup}, "
            f"fit={counts_after_fit}"
        )

    best_numpy = problem.evaluate(result.best_parameters)
    best_simulation = problem.simulate(result.best_parameters)
    true_vector = np.asarray(
        [case.TRUE_VALUES[item.name] for item in case.PARAMETERS], dtype=float
    )
    true_residuals = residual_function(true_vector)
    true_cost = 0.5 * float(true_residuals @ true_residuals)
    model_reflectivity, model_curves = model.simulate_curves(
        np.asarray([result.best_parameters[item.name] for item in case.PARAMETERS])
    )
    parity = _parity_metrics(
        best_simulation,
        np.asarray(model_reflectivity),
        tuple(np.asarray(curve) for curve in model_curves),
    )
    output = {
        "problem": problem,
        "plan": plan,
        "result": result,
        "initial": initial,
        "initial_cost": initial_cost,
        "initial_jacobian_shape": initial_jacobian.shape,
        "true_cost": true_cost,
        "compile_seconds": compile_seconds,
        "counts_after_warmup": counts_after_warmup,
        "counts_after_fit": counts_after_fit,
        "best_numpy": best_numpy,
        "best_simulation": best_simulation,
        "parity": parity,
        "reflectivity_weight": reflectivity_weight,
    }
    save_outputs(args.output_dir, output)
    return output


def _parity_metrics(simulation, model_reflectivity, model_curves):
    metrics = {
        "reflectivity_max_absolute": float(
            np.max(np.abs(simulation.reflectivity.reflectivity - model_reflectivity))
        )
    }
    by_name = {
        item.name: item.curve.intensity for item in simulation.rocking_curves.core_levels
    }
    for name, curve in zip(case.RC_COLUMN_BY_NAME, model_curves):
        metrics[f"{name}_max_absolute"] = float(np.max(np.abs(by_name[name] - curve)))
    return metrics


def save_outputs(output_dir: Path, output: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = output["result"]
    summary = summary_values(output)
    (output_dir / "summary.txt").write_text(
        "".join(f"{key}: {value}\n" for key, value in summary.items()),
        encoding="utf-8",
    )
    _write_rows(output_dir / "summary.csv", [summary])
    _write_rows(
        output_dir / "history.csv",
        [
            {
                "iteration": item.iteration,
                "cost": item.cost,
                "gradient_norm": item.gradient_norm,
                **item.parameters,
            }
            for item in result.history
        ],
    )
    _save_curves(output_dir / "best_fit_curves.csv", output)
    _plot_fit(output_dir / "best_fit.png", output)
    _plot_convergence(output_dir / "convergence.png", output)
    _plot_parameter_diagnostics(output_dir, result)


def _plot_parameter_diagnostics(output_dir: Path, result) -> None:
    """Save public-API uncertainty and correlation diagnostics for the fit."""

    import matplotlib.pyplot as plt

    diagnostics = diagnostics_from_least_squares_result(result, case.PARAMETERS)
    uncertainty_figure, _ = plot_parameter_estimates(diagnostics)
    uncertainty_figure.savefig(
        output_dir / "parameter_uncertainty.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(uncertainty_figure)

    correlation_figure, _ = plot_correlation_matrix(diagnostics)
    correlation_figure.savefig(
        output_dir / "parameter_correlation.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(correlation_figure)


def summary_values(output: dict[str, object]) -> dict[str, object]:
    result = output["result"]
    values: dict[str, object] = {
        "initial_cost": output["initial_cost"],
        "true_parameter_cost": output["true_cost"],
        "final_cost": result.final_cost,
        "cost_reduction_factor": output["initial_cost"] / result.final_cost,
        "success": result.success,
        "status": result.status,
        "message": result.message,
        "nfev": result.nfev,
        "njev": result.njev,
        "optimality": result.optimality,
        "compile_seconds": output["compile_seconds"],
        "optimization_seconds": result.total_seconds,
        "residual_compilations": output["counts_after_fit"][0],
        "jacobian_compilations": output["counts_after_fit"][1],
        "total_compilations": sum(output["counts_after_fit"]),
        "grid_cells": sum(output["plan"].slice_counts),
        "reflectivity_weight": output["reflectivity_weight"],
        "numpy_objective": output["best_numpy"].objective,
        **output["parity"],
    }
    singular_values = np.linalg.svd(result.final_jacobian, compute_uv=False)
    values["jacobian_rank"] = int(np.linalg.matrix_rank(result.final_jacobian))
    values["jacobian_largest_singular_value"] = float(singular_values[0])
    values["jacobian_smallest_singular_value"] = float(singular_values[-1])
    values["jacobian_condition_number"] = float(
        singular_values[0] / singular_values[-1]
    )
    initial_values = values_from_vector(output["initial"])
    for parameter_index, parameter in enumerate(case.PARAMETERS):
        name = parameter.name
        best = result.best_parameters[name]
        true = case.TRUE_VALUES[name]
        values[f"{name}_initial"] = initial_values[name]
        values[f"{name}_best"] = best
        values[f"{name}_true"] = true
        values[f"{name}_error"] = best - true
        values[f"{name}_jacobian_column_norm"] = float(
            np.linalg.norm(result.final_jacobian[:, parameter_index])
        )
        if result.covariance is not None:
            variance = result.covariance[
                parameter_index, parameter_index
            ]
            values[f"{name}_estimated_uncertainty"] = float(
                np.sqrt(max(float(variance), 0.0))
            )
    return values


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _save_curves(path: Path, output: dict[str, object]) -> None:
    problem = output["problem"]
    simulation = output["best_simulation"]
    columns = [problem.reflectivity.angles, problem.reflectivity.reflectivity]
    names = ["angle_deg", "reflectivity_target"]
    columns.append(simulation.reflectivity.reflectivity)
    names.append("reflectivity_fit")
    simulated = {
        core.name: core.curve.intensity for core in simulation.rocking_curves.core_levels
    }
    for data in problem.rocking_curves:
        columns.extend((data.intensity, simulated[data.name]))
        label = case.RC_COLUMN_BY_NAME[data.name]
        names.extend((f"{label}_target", f"{label}_fit"))
    np.savetxt(
        path,
        np.column_stack(columns),
        delimiter=",",
        header=",".join(names),
        comments="",
    )


def _plot_fit(path: Path, output: dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    problem = output["problem"]
    simulation = output["best_simulation"]
    fig, axes = plt.subplots(5, 1, figsize=(8.0, 10.0), sharex=True)
    axes[0].semilogy(
        problem.reflectivity.angles,
        problem.reflectivity.reflectivity,
        "o",
        markersize=2.5,
        alpha=0.55,
        label="target",
    )
    axes[0].semilogy(
        simulation.reflectivity.angle,
        simulation.reflectivity.reflectivity,
        linewidth=1.4,
        label="fit",
    )
    axes[0].set_ylabel("Reflectivity")
    axes[0].legend()
    by_name = {
        core.name: core.curve.intensity for core in simulation.rocking_curves.core_levels
    }
    for axis, data in zip(axes[1:], problem.rocking_curves):
        axis.plot(data.angles, data.intensity, "o", markersize=2.5, alpha=0.55)
        axis.plot(data.angles, by_name[data.name], linewidth=1.4)
        axis.set_ylabel(data.name)
    for axis in axes:
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Grazing incidence angle (deg)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_convergence(path: Path, output: dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    history = output["result"].history
    fig, axis = plt.subplots(figsize=(7.0, 4.2))
    axis.semilogy(
        [item.iteration for item in history],
        [item.cost for item in history],
        marker="o",
        markersize=3,
    )
    axis.set_xlabel("Jacobian evaluation")
    axis.set_ylabel("Least-squares cost")
    axis.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def print_results(output: dict[str, object]) -> None:
    summary = summary_values(output)
    print("Synthetic unified-grid JAX TRF result")
    for key in (
        "initial_cost",
        "true_parameter_cost",
        "final_cost",
        "cost_reduction_factor",
        "success",
        "message",
        "nfev",
        "njev",
        "compile_seconds",
        "optimization_seconds",
        "residual_compilations",
        "jacobian_compilations",
        "grid_cells",
        "numpy_objective",
        "reflectivity_max_absolute",
    ):
        print(f"  {key}: {summary[key]}")
    print("Parameters (initial -> best; true):")
    for parameter in case.PARAMETERS:
        name = parameter.name
        print(
            f"  {name}: {summary[name + '_initial']:.6g} -> "
            f"{summary[name + '_best']:.6g}; true {summary[name + '_true']:.6g}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--min-slices", type=int, default=10)
    parser.add_argument("--max-slice-thickness", type=float, default=2.0)
    parser.add_argument("--ftol", type=float, default=1.0e-10)
    parser.add_argument("--xtol", type=float, default=1.0e-10)
    parser.add_argument("--gtol", type=float, default=1.0e-8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_fit(args)
    print_results(output)
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
