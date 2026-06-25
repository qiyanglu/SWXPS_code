"""Fit Sample 13 on one fixed unified grid with JAX Jacobians and TRF."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


FIT_DIR = Path(__file__).resolve().parent
CASE_DIR = FIT_DIR.parent
REPO_ROOT = CASE_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
LEGACY_TRF_DIR = CASE_DIR / "jax_least_squares_all_rcs"
GRADIENT_DIR = CASE_DIR / "jax_gradient_fit_without_la4d"
for path in (SRC_DIR, LEGACY_TRF_DIR, GRADIENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from swanx.diagnostics import plot_stack_schematic

from swanx.fitting import (

    JaxLeastSquaresOptimizerSettings,

    JaxLeastSquaresResidualSettings,

    build_jax_residual_function,

    optimize_with_jax_least_squares,

)

from swanx.stack import (

    LayerSlicingPolicy,

    fixed_layer_grid,

    fixed_layer_grid_plan,

)
from swanx.reflectivity_jax import (  # noqa: E402
    transfer_matrix_field_intensity_jax,
    transfer_matrix_reflectivity_jax,
)
from swanx.result_exports import (  # noqa: E402
    save_fit_curve_data_csv,
    save_optimized_stack_csv,
)

import fit_sample13_reflectivity_all_rcs_least_squares as legacy_trf  # noqa: E402


PARAMETERS = legacy_trf.PARAMETERS
PARAMETER_INDEX = {item.name: index for index, item in enumerate(PARAMETERS)}
START_SUMMARY = (
    CASE_DIR / "best_results_so_far" / "sample13_trf_all_rcs_summary.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "runs" / "sample_13" / "jax_least_squares_fixed_grid" / "single_run"
)
SUPERLATTICE_REPEATS = legacy_trf.fit13.sample13.SUPERLATTICE_REPEATS
BOTTOM_LNO_THICKNESS = legacy_trf.fit13.BOTTOM_LNO_THICKNESS


class Sample13FixedGridJaxModel:
    """Fully traceable Sample 13 optics and XPS model on a fixed layer grid."""

    def __init__(
        self,
        reflectivity_angles,
        rocking_curve_angles,
        core_levels,
        plan,
        nominal_stack,
    ):
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        self.jax = jax
        self.jnp = jnp
        self.reflectivity_angles = jnp.asarray(
            reflectivity_angles, dtype=jnp.float64
        )
        self.rocking_curve_angles = jnp.asarray(
            rocking_curve_angles, dtype=jnp.float64
        )
        self.repeat_indices = jnp.arange(SUPERLATTICE_REPEATS, dtype=jnp.float64)

        grid = fixed_layer_grid(nominal_stack.optical_layers, plan)
        nominal_index = np.asarray(grid.nominal_layer_index, dtype=np.int32)
        counts = np.asarray(plan.slice_counts, dtype=float)
        self.cell_nominal_index = jnp.asarray(nominal_index, dtype=jnp.int32)
        self.cell_slice_count = jnp.asarray(
            counts[nominal_index - 1], dtype=jnp.float64
        )
        self.effective_layer_index = jnp.asarray(
            grid.effective_layer_index, dtype=jnp.int32
        )
        self.nominal_delta = jnp.asarray(
            [layer.delta for layer in nominal_stack.optical_layers],
            dtype=jnp.float64,
        )
        self.nominal_beta = jnp.asarray(
            [layer.beta for layer in nominal_stack.optical_layers],
            dtype=jnp.float64,
        )
        self.core_inputs = tuple(
            self._core_inputs(core, nominal_stack.materials) for core in core_levels
        )

    @property
    def cell_count(self) -> int:
        return int(self.cell_nominal_index.size)

    def _core_inputs(self, core, materials):
        concentration = np.asarray(
            [core.concentration_by_material.get(material, 0.0) for material in materials],
            dtype=float,
        )
        if core.emitting_layer_indices is not None:
            emitting = set(core.emitting_layer_indices)
            concentration = np.asarray(
                [value if index in emitting else 0.0 for index, value in enumerate(concentration)],
                dtype=float,
            )
        imfp = np.asarray(
            [core.imfp_by_material[material] for material in materials], dtype=float
        )
        return (
            self.jnp.asarray(concentration, dtype=self.jnp.float64),
            self.jnp.asarray(1.0 / imfp, dtype=self.jnp.float64),
            float(core.emission_angle_deg),
        )

    def nominal_geometry(self, physical_vector):
        """Return fixed-length nominal thickness and upper-interface roughness arrays."""

        jnp = self.jnp
        vector = jnp.asarray(physical_vector, dtype=jnp.float64)

        def value(name):
            return vector[PARAMETER_INDEX[name]]

        carbon = value("carbon_thickness")
        layer1 = value("top_lno_layer1_thickness")
        layer2 = value("top_lno_total_thickness") - layer1
        transition_fraction = 0.5 * (
            1.0
            + self.jax.lax.erf(
                (self.repeat_indices - value("thickness_transition_repeat"))
                / (jnp.sqrt(2.0) * value("thickness_transition_width"))
            )
        )
        sto = value("sto_thickness_start") + (
            transition_fraction * value("sto_thickness_delta")
        )
        lno = value("lno_thickness_start") + (
            transition_fraction * value("lno_thickness_delta")
        )
        superlattice_thicknesses = jnp.stack((sto, lno), axis=1).reshape(-1)
        finite_thicknesses = jnp.concatenate(
            (
                jnp.stack((carbon, layer1, layer2)),
                jnp.asarray([BOTTOM_LNO_THICKNESS], dtype=jnp.float64),
                superlattice_thicknesses,
            )
        )
        thicknesses = jnp.concatenate(
            (jnp.zeros((1,)), finite_thicknesses, jnp.zeros((1,)))
        )

        carbon_roughness = 1.0 + value("carbon_roughness_fraction") * (
            jnp.minimum(8.0, carbon) - 1.0
        )
        repeat_fraction = self.repeat_indices / (SUPERLATTICE_REPEATS - 1)
        sto_roughness = value("sto_roughness_first") + repeat_fraction * (
            value("sto_roughness_last") - value("sto_roughness_first")
        )
        lno_roughness = value("lno_roughness_first") + repeat_fraction * (
            value("lno_roughness_last") - value("lno_roughness_first")
        )
        superlattice_roughnesses = jnp.stack(
            (sto_roughness, lno_roughness), axis=1
        ).reshape(-1)
        roughnesses = jnp.concatenate(
            (
                jnp.zeros((1,)),
                jnp.stack(
                    (
                        carbon_roughness,
                        jnp.asarray(0.0, dtype=jnp.float64),
                        value("lno2_roughness_fraction") * layer1,
                        jnp.asarray(0.0, dtype=jnp.float64),
                    )
                ),
                superlattice_roughnesses,
                jnp.reshape(value("substrate_roughness"), (1,)),
            )
        )
        return thicknesses, roughnesses

    def simulate_curves(self, physical_vector):
        """Return reflectivity and three normalized rocking curves."""

        jnp = self.jnp
        vector = jnp.asarray(physical_vector, dtype=jnp.float64)
        thicknesses, roughnesses = self.nominal_geometry(vector)
        finite_thicknesses = thicknesses[1:-1]
        widths = (
            finite_thicknesses[self.cell_nominal_index - 1]
            / self.cell_slice_count
        )
        centers = jnp.cumsum(widths) - 0.5 * widths
        boundaries = jnp.concatenate(
            (jnp.zeros((1,)), jnp.cumsum(finite_thicknesses))
        )
        cell_delta = self._graded_optical_property(
            centers, boundaries, roughnesses, self.nominal_delta
        )
        cell_beta = self._graded_optical_property(
            centers, boundaries, roughnesses, self.nominal_beta
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

        reflectivity_angles = (
            self.reflectivity_angles + vector[PARAMETER_INDEX["reflectivity_angle_offset"]]
        )
        rocking_curve_angles = (
            self.rocking_curve_angles + vector[PARAMETER_INDEX["rc_angle_offset"]]
        )
        reflectivity = transfer_matrix_reflectivity_jax(
            reflectivity_angles,
            legacy_trf.fit13.sample13.PHOTON_ENERGY_EV,
            effective_thicknesses,
            effective_delta,
            effective_beta,
        )
        field_intensity = transfer_matrix_field_intensity_jax(
            rocking_curve_angles,
            legacy_trf.fit13.sample13.PHOTON_ENERGY_EV,
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
                roughnesses,
                concentration,
                attenuation_coefficient,
                emission_angle,
            )
            for concentration, attenuation_coefficient, emission_angle in self.core_inputs
        )
        return reflectivity, curves

    def _graded_optical_property(self, centers, boundaries, roughnesses, nominal_values):
        """Apply the validated nearest-interface optical grading rule."""

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

    def _graded_xps_property(self, centers, boundaries, roughnesses, nominal_values):
        """Apply the maintained sequential adjacent-interface XPS grading rule."""

        jnp = self.jnp
        initial = nominal_values[self.cell_nominal_index]

        def grade_interface(interface_index, values):
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
            return jnp.where(active, mixed, values)

        return self.jax.lax.fori_loop(
            0, nominal_values.size - 1, grade_interface, initial
        )

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
        jnp = self.jnp
        concentration = self._graded_xps_property(
            centers, boundaries, roughnesses, nominal_concentration
        )
        attenuation_coefficient = self._graded_xps_property(
            centers,
            boundaries,
            roughnesses,
            nominal_attenuation_coefficient,
        )
        cos_alpha = jnp.cos(jnp.deg2rad(emission_angle))
        cell_optical_depth = widths * attenuation_coefficient / cos_alpha
        optical_depth = jnp.cumsum(cell_optical_depth) - 0.5 * cell_optical_depth
        weights = concentration * jnp.exp(-optical_depth) * widths
        raw = jnp.sum(field_intensity * weights[:, None], axis=0)
        return raw / jnp.mean(raw)


def capacity_layers(nominal_stack):
    """Return topology-matched layers with independent fitted thickness capacities."""

    maximum_superlattice_thickness = np.nextafter(
        max(
            next(item.upper for item in PARAMETERS if item.name == "sto_thickness_start")
            + next(item.upper for item in PARAMETERS if item.name == "sto_thickness_delta"),
            next(item.upper for item in PARAMETERS if item.name == "lno_thickness_start")
            + next(item.upper for item in PARAMETERS if item.name == "lno_thickness_delta"),
        ),
        np.inf,
    )
    finite_capacities = (
        next(item.upper for item in PARAMETERS if item.name == "carbon_thickness"),
        next(item.upper for item in PARAMETERS if item.name == "top_lno_layer1_thickness"),
        next(item.upper for item in PARAMETERS if item.name == "top_lno_total_thickness")
        - next(item.lower for item in PARAMETERS if item.name == "top_lno_layer1_thickness"),
        BOTTOM_LNO_THICKNESS,
        *([maximum_superlattice_thickness] * (2 * SUPERLATTICE_REPEATS)),
    )
    layers = nominal_stack.optical_layers
    if len(finite_capacities) != len(layers) - 2:
        raise ValueError("Sample 13 capacity topology does not match the nominal stack")
    return (
        layers[0],
        *(replace(layer, thickness=capacity) for layer, capacity in zip(layers[1:-1], finite_capacities)),
        layers[-1],
    )


def make_fixed_grid_problem(problem, plan):
    """Attach the same fixed grid plan to reflectivity and RC problems."""

    return legacy_trf.fit13.JointCap3Problem(
        parameters=problem.parameters,
        reflectivity_problem=replace(
            problem.reflectivity_problem,
            roughness_step=1.0,
            slicing=plan,
        ),
        rc_problem=replace(
            problem.rc_problem,
            field_step=1.0,
            roughness_step=1.0,
            slicing=plan,
        ),
    )


def values_from_vector(vector) -> dict[str, float]:
    return {
        parameter.name: float(value)
        for parameter, value in zip(PARAMETERS, np.asarray(vector, dtype=float))
    }


def resolve_reflectivity_weight(data, plan, start_values, requested):
    if requested != "auto":
        return float(requested), {}
    diagnostic = make_fixed_grid_problem(
        legacy_trf.make_problem(data, reflectivity_weight=1.0), plan
    ).evaluate(start_values)
    raw = {item.name: item.raw for item in diagnostic.contributions}
    weighted = {item.name: item.weighted for item in diagnostic.contributions}
    rc_weighted = sum(weighted[name] for name in ("C 1s", "Ni 3p", "La 4d"))
    weight = 1.0 if raw["reflectivity"] <= 0 else rc_weighted / raw["reflectivity"]
    return weight, {"reflectivity_raw": raw["reflectivity"], "rc_weighted": rc_weighted}


def parity_metrics(simulation, model_reflectivity, model_curves):
    metrics = {
        "reflectivity_max_absolute": float(
            np.max(np.abs(simulation.reflectivity.reflectivity - model_reflectivity))
        )
    }
    simulated = {
        item.name: item.curve.intensity
        for item in simulation.rocking_curves.core_levels
    }
    for name, curve in zip(("C 1s", "Ni 3p", "La 4d"), model_curves):
        metrics[f"{name}_max_absolute"] = float(
            np.max(np.abs(simulated[name] - curve))
        )
    return metrics


def contribution_rows(problem, reflectivity, curves):
    target = problem.reflectivity
    block = np.log10(np.maximum(reflectivity, target.log_floor)) - np.log10(
        np.maximum(target.reflectivity, target.log_floor)
    )
    rows = [
        {
            "name": "reflectivity",
            "raw": float(np.mean(block**2)),
            "weight": target.weight,
            "weighted": float(target.weight * np.mean(block**2)),
        }
    ]
    for data, curve in zip(problem.rocking_curves, curves):
        raw = float(np.mean((np.asarray(curve) - data.intensity) ** 2))
        rows.append(
            {"name": data.name, "raw": raw, "weight": data.weight, "weighted": data.weight * raw}
        )
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_outputs(
    output_dir,
    problem,
    simulation,
    result,
    plan,
    initial_objective,
    initial_unified,
    initial_legacy,
    best_unified,
    best_legacy,
    initial_contributions,
    best_contributions,
    reflectivity_weight,
    weight_diagnostics,
    compile_seconds,
    total_wall_seconds,
    compilation_counts,
    initial_parity,
    best_parity,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_trf.save_history(output_dir / "history.csv", result)
    legacy_trf.plot_history(output_dir / "convergence.png", result)
    write_rows(output_dir / "initial_jax_contributions.csv", initial_contributions)
    write_rows(output_dir / "jax_contributions.csv", best_contributions)
    legacy_trf.save_contributions(
        output_dir / "numpy_unified_contributions.csv", best_unified
    )
    legacy_trf.save_contributions(
        output_dir / "legacy_validation_contributions.csv", best_legacy
    )
    legacy_trf.save_covariance(output_dir / "covariance.csv", result)
    legacy_trf.save_uncertainties(output_dir / "parameter_uncertainties.csv", result)
    legacy_trf.save_parameter_positions(output_dir / "parameter_positions.csv", result)
    legacy_trf.plot_parameter_positions(output_dir / "parameter_positions.png", result)
    legacy_trf.plot_best_fit(
        output_dir / "best_fit.png",
        problem.reflectivity,
        problem.rocking_curves,
        simulation,
    )
    plot_stack_schematic(
        output_dir / "stack_schematic.png",
        simulation.stack,
        title="Sample #13 fixed-grid JAX least-squares fit",
        top_layers=5,
        bottom_layers=3,
    )
    save_fit_curve_data_csv(
        output_dir / "best_fit_experiment_and_simulation.csv",
        problem.reflectivity,
        problem.rocking_curves,
        simulation,
    )
    save_optimized_stack_csv(output_dir / "optimized_stack_layers.csv", simulation.stack)
    write_rows(
        output_dir / "grid_plan.csv",
        [
            {
                "finite_layer_index": index,
                "material": material,
                "capacity_thickness": capacity,
                "best_thickness": layer.thickness,
                "slice_count": count,
                "best_cell_width": layer.thickness / count,
            }
            for index, (material, layer, capacity, count) in enumerate(
                zip(
                    simulation.stack.materials[1:-1],
                    simulation.stack.optical_layers[1:-1],
                    plan.capacity_thicknesses,
                    plan.slice_counts,
                ),
                start=1,
            )
        ],
    )

    singular_values = np.linalg.svd(result.final_jacobian, compute_uv=False)
    jacobian_rank = int(np.linalg.matrix_rank(result.final_jacobian))
    jacobian_condition = float(
        np.inf
        if singular_values[-1] == 0.0
        else singular_values[0] / singular_values[-1]
    )
    write_rows(
        output_dir / "jacobian_singular_values.csv",
        [
            {"index": index, "singular_value": value}
            for index, value in enumerate(singular_values, start=1)
        ],
    )
    near_bound_count = sum(
        row["near_lower_bound_1pct"] or row["near_upper_bound_1pct"]
        for row in legacy_trf.parameter_position_rows(result)
    )

    row = {
        "initial_jax_objective": initial_objective,
        "initial_numpy_unified_objective": initial_unified.objective,
        "initial_legacy_objective": initial_legacy.objective,
        "optimizer_objective": 2.0 * result.final_cost,
        "best_numpy_unified_objective": best_unified.objective,
        "best_legacy_objective": best_legacy.objective,
        "success": result.success,
        "status": result.status,
        "message": result.message,
        "nfev": result.nfev,
        "njev": result.njev,
        "optimality": result.optimality,
        "compile_seconds": compile_seconds,
        "optimization_seconds": result.total_seconds,
        "total_wall_seconds": total_wall_seconds,
        "residual_compilations": compilation_counts[0],
        "jacobian_compilations": compilation_counts[1],
        "grid_cells": sum(plan.slice_counts),
        "jacobian_rank": jacobian_rank,
        "jacobian_condition_number": jacobian_condition,
        "parameters_within_1pct_of_bound": near_bound_count,
        "reflectivity_weight": reflectivity_weight,
        "carbon_roughness": legacy_trf.fit13.carbon_roughness_from_values(result.best_parameters),
        "lno1_roughness": legacy_trf.fit13.lno1_roughness_from_values(result.best_parameters),
        "lno2_roughness": legacy_trf.fit13.lno2_roughness_from_values(result.best_parameters),
        "top_lno_layer2_thickness": (
            result.best_parameters["top_lno_total_thickness"]
            - result.best_parameters["top_lno_layer1_thickness"]
        ),
        **{f"initial_parity_{key}": value for key, value in initial_parity.items()},
        **{f"best_parity_{key}": value for key, value in best_parity.items()},
        **result.best_parameters,
    }
    for name, value in weight_diagnostics.items():
        row[f"weight_diagnostic_{name}"] = value
    write_rows(output_dir / "summary.csv", [row])
    (output_dir / "summary.txt").write_text(
        "".join(f"{key}: {value}\n" for key, value in row.items()),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-nfev", type=int, default=60)
    parser.add_argument("--start-summary", type=Path, default=START_SUMMARY)
    parser.add_argument("--reflectivity-weight", default="auto")
    parser.add_argument("--min-slices", type=int, default=10)
    parser.add_argument("--max-slice-thickness", type=float, default=2.0)
    parser.add_argument("--ftol", type=float, default=1.0e-10)
    parser.add_argument("--xtol", type=float, default=1.0e-10)
    parser.add_argument("--gtol", type=float, default=1.0e-8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--setup-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    start_wall = perf_counter()
    legacy_trf.fit13.patch_archived_context_paths()
    data = legacy_trf.fit13.sample13.load_and_prepare_data(10.0, 2)
    data = legacy_trf.fit13.sample13.apply_reflectivity_window(
        data,
        legacy_trf.fit13.sample13.RC_START_DEG,
        legacy_trf.fit13.sample13.RC_STOP_DEG,
    )
    start_values = legacy_trf.load_start_values(args.start_summary)
    initial = legacy_trf.interior_vector(start_values)
    initial_values = values_from_vector(initial)
    nominal_stack = legacy_trf.fit13.build_cap3_stack(initial_values)
    policy = LayerSlicingPolicy(
        min_slices=args.min_slices,
        max_slice_thickness=args.max_slice_thickness,
    )
    plan = fixed_layer_grid_plan(capacity_layers(nominal_stack), policy)
    reflectivity_weight, weight_diagnostics = resolve_reflectivity_weight(
        data, plan, initial_values, args.reflectivity_weight
    )
    legacy_problem = legacy_trf.make_problem(data, reflectivity_weight)
    unified_problem = make_fixed_grid_problem(legacy_problem, plan)
    model = Sample13FixedGridJaxModel(
        unified_problem.reflectivity.angles,
        unified_problem.rocking_curves[0].angles,
        unified_problem.rc_problem.core_levels,
        plan,
        nominal_stack,
    )
    residual_function = build_jax_residual_function(
        model.simulate_curves,
        reflectivity=unified_problem.reflectivity,
        rocking_curves=unified_problem.rocking_curves,
        settings=JaxLeastSquaresResidualSettings(
            reflectivity_log=True,
            rocking_curve_normalization="none",
        ),
    )

    compile_start = perf_counter()
    initial_residuals = residual_function(initial)
    initial_jacobian = residual_function.jacobian(initial)
    compile_seconds = perf_counter() - compile_start
    compilation_counts = (
        residual_function.compilation_counter.residual_compilations,
        residual_function.compilation_counter.jacobian_compilations,
    )
    if compilation_counts != (1, 1):
        raise RuntimeError(f"unexpected warmup compilation counts: {compilation_counts}")
    initial_objective = float(initial_residuals @ initial_residuals)
    initial_model_reflectivity, initial_model_curves = model.simulate_curves(initial)
    initial_model_reflectivity = np.asarray(initial_model_reflectivity)
    initial_model_curves = tuple(np.asarray(curve) for curve in initial_model_curves)
    initial_simulation = unified_problem.simulate(initial_values)
    initial_unified = unified_problem.evaluate(initial_values)
    initial_legacy = legacy_problem.evaluate(initial_values)
    initial_parity = parity_metrics(
        initial_simulation, initial_model_reflectivity, initial_model_curves
    )

    print("Sample #13 fixed-grid JAX TRF setup")
    print(f"  output directory: {args.output_dir}")
    print(f"  residuals x parameters: {initial_residuals.size} x {initial.size}")
    print(f"  Jacobian shape: {initial_jacobian.shape}")
    print(f"  nominal layers: {len(nominal_stack.optical_layers)}")
    print(f"  fixed grid cells: {model.cell_count}")
    print(f"  compile seconds: {compile_seconds:.3f}")
    print(f"  compilations (residual, Jacobian): {compilation_counts}")
    print(f"  initial JAX objective: {initial_objective:.12g}")
    print(f"  initial NumPy unified objective: {initial_unified.objective:.12g}")
    print(f"  initial legacy objective: {initial_legacy.objective:.12g}")
    print(f"  initial parity: {initial_parity}")
    if args.setup_only:
        return

    result = optimize_with_jax_least_squares(
        PARAMETERS,
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
    if counts_after_fit != (1, 1):
        raise RuntimeError(f"fixed-grid model recompiled during fit: {counts_after_fit}")

    best_vector = np.asarray(
        [result.best_parameters[item.name] for item in PARAMETERS], dtype=float
    )
    best_model_reflectivity, best_model_curves = model.simulate_curves(best_vector)
    best_model_reflectivity = np.asarray(best_model_reflectivity)
    best_model_curves = tuple(np.asarray(curve) for curve in best_model_curves)
    best_simulation = unified_problem.simulate(result.best_parameters)
    best_unified = unified_problem.evaluate(result.best_parameters)
    best_legacy = legacy_problem.evaluate(result.best_parameters)
    best_parity = parity_metrics(
        best_simulation, best_model_reflectivity, best_model_curves
    )
    initial_contributions = contribution_rows(
        unified_problem, initial_model_reflectivity, initial_model_curves
    )
    best_contributions = contribution_rows(
        unified_problem, best_model_reflectivity, best_model_curves
    )
    total_wall_seconds = perf_counter() - start_wall
    save_outputs(
        args.output_dir,
        unified_problem,
        best_simulation,
        result,
        plan,
        initial_objective,
        initial_unified,
        initial_legacy,
        best_unified,
        best_legacy,
        initial_contributions,
        best_contributions,
        reflectivity_weight,
        weight_diagnostics,
        compile_seconds,
        total_wall_seconds,
        counts_after_fit,
        initial_parity,
        best_parity,
    )

    print("Sample #13 fixed-grid JAX TRF result")
    print(f"  success: {result.success}")
    print(f"  status: {result.status} ({result.message})")
    print(f"  nfev/njev: {result.nfev}/{result.njev}")
    print(f"  optimality: {result.optimality:.6g}")
    print(f"  optimization seconds: {result.total_seconds:.3f}")
    print(f"  total wall seconds: {total_wall_seconds:.3f}")
    print(f"  initial objective: {initial_objective:.12g}")
    print(f"  final JAX objective: {2.0 * result.final_cost:.12g}")
    print(f"  final NumPy unified objective: {best_unified.objective:.12g}")
    print(f"  final legacy objective: {best_legacy.objective:.12g}")
    print(f"  compilations (residual, Jacobian): {counts_after_fit}")
    print(f"  best parity: {best_parity}")
    print(f"  saved artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
