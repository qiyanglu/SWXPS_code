"""Internal fixed-grid JAX residual builder for ProjectSpec fits."""

from __future__ import annotations

import ast
from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from swanx.fitting import JaxLeastSquaresResidualSettings, build_jax_residual_function
from swanx.polarization import polarization_weights
from swanx.reflectivity_jax import (
    transfer_matrix_field_intensity_jax,
    transfer_matrix_reflectivity_jax,
)
from swanx.stack.slicing import FixedLayerGridPlan

from .expressions import ExpressionError
from .spec import ProjectValidationError


def build_projectspec_jax_residual_function(built):
    """Build the default fixed-shape JAX residual from ProjectSpec inputs."""

    problem = built.fitting_problem
    if problem is None:
        raise ProjectValidationError("run.mode='jax_least_squares' requires at least one dataset")
    if not isinstance(problem.slicing, FixedLayerGridPlan):
        raise ProjectValidationError(
            "run.optimizer.residual='auto_fixed_grid' requires "
            "settings.slicing.mode: 'fixed_grid' so the JAX array shapes are fixed"
        )
    if problem.reflectivity is None and not problem.rocking_curves:
        raise ProjectValidationError("run.mode='jax_least_squares' requires at least one dataset")

    reflectivity = None
    if problem.reflectivity is not None:
        log_floor = float(built.spec.datasets.get("reflectivity", {}).get("log_floor", problem.reflectivity.log_floor))
        reflectivity = replace(problem.reflectivity, log_floor=log_floor)
        angles = np.asarray(reflectivity.angles, dtype=float)
    else:
        angles = np.asarray(problem.rocking_curves[0].angles, dtype=float)

    offpeak_mask = (
        np.ones(angles.shape, dtype=bool)
        if problem.offpeak_mask is None
        else np.asarray(problem.offpeak_mask, dtype=bool)
    )
    model = _ProjectSpecFixedGridJaxModel(
        built=built,
        plan=problem.slicing,
        angles=angles,
        offpeak_mask=offpeak_mask,
        normalization_mode=problem.rocking_curve_normalization,
        normalization_edge_fraction=problem.normalization_edge_fraction,
        normalization_polynomial_order=problem.normalization_polynomial_order,
    )
    return build_jax_residual_function(
        model.simulate_curves,
        reflectivity=reflectivity,
        rocking_curves=problem.rocking_curves,
        settings=JaxLeastSquaresResidualSettings(
            reflectivity_log=True,
            rocking_curve_normalization="mean_absolute",
        ),
    )


class _ProjectSpecFixedGridJaxModel:
    def __init__(
        self,
        *,
        built,
        plan: FixedLayerGridPlan,
        angles,
        offpeak_mask,
        normalization_mode: str,
        normalization_edge_fraction: float,
        normalization_polynomial_order: int,
    ):
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        self.jax = jax
        self.jnp = jnp
        self.spec = built.spec
        self.problem = built.fitting_problem
        self.plan = plan
        self.angles = jnp.asarray(angles, dtype=jnp.float64)
        self.offpeak_mask = jnp.asarray(offpeak_mask, dtype=bool)
        self.normalization_mode = str(normalization_mode)
        self.parameter_names = tuple(parameter.name for parameter in self.problem.parameters)
        self.parameter_index = {name: index for index, name in enumerate(self.parameter_names)}
        self.default_values = dict(self.spec.default_parameter_values())
        self.angle_offset_index = self._optional_parameter_index(self.problem.angle_offset_parameter)
        self.reflectivity_angle_offset_index = self._optional_parameter_index(
            self.problem.reflectivity_angle_offset_parameter
        )
        self.rocking_curve_angle_offset_index = self._optional_parameter_index(
            self.problem.rocking_curve_angle_offset_parameter
        )

        stack = built.stack
        finite_count = len(stack.layers) - 2
        if finite_count <= 0:
            raise ProjectValidationError("auto fixed-grid JAX residual requires at least one finite layer")
        if len(plan.slice_counts) != finite_count:
            raise ProjectValidationError(
                "fixed-grid slicing plan does not match the ProjectSpec stack topology"
            )

        count_by_layer = np.asarray(plan.slice_counts, dtype=float)
        nominal_index = np.concatenate(
            [
                np.full(int(count), finite_layer + 1, dtype=np.int32)
                for finite_layer, count in enumerate(plan.slice_counts)
            ]
        )
        if nominal_index.size == 0:
            raise ProjectValidationError("auto fixed-grid JAX residual produced an empty depth grid")
        self.cell_nominal_index = jnp.asarray(nominal_index, dtype=jnp.int32)
        self.cell_slice_count = jnp.asarray(count_by_layer[nominal_index - 1], dtype=jnp.float64)
        self.effective_layer_index = jnp.arange(1, nominal_index.size + 1, dtype=jnp.int32)

        self.nominal_delta = jnp.asarray([layer.delta for layer in stack.layers], dtype=jnp.float64)
        self.nominal_beta = jnp.asarray([layer.beta for layer in stack.layers], dtype=jnp.float64)
        self.core_inputs = tuple(self._core_inputs(core, stack.materials) for core in self.problem.core_levels)
        self.s_weight, self.p_weight = polarization_weights(self.problem.polarization)
        self.energy_ev = float(self.spec.photon_energy_ev)
        if self.problem.rocking_curves and self.normalization_mode == "edge_polynomial":
            (
                self.edge_indices,
                self.edge_design_pinv,
                self.full_design,
            ) = self._edge_polynomial_matrices(
                np.asarray(angles, dtype=float),
                normalization_edge_fraction,
                normalization_polynomial_order,
            )
        else:
            self.edge_indices = jnp.asarray((), dtype=jnp.int32)
            self.edge_design_pinv = jnp.zeros((0, 0), dtype=jnp.float64)
            self.full_design = jnp.zeros((0, 0), dtype=jnp.float64)

    def _optional_parameter_index(self, name: str | None) -> int | None:
        if name is None:
            return None
        return self.parameter_index.get(name)

    def _core_inputs(self, core, materials):
        concentration = np.asarray(
            [core.concentration_by_material.get(material, 0.0) for material in materials],
            dtype=float,
        )
        if core.emitting_layer_indices is not None:
            selected = np.zeros(concentration.shape, dtype=bool)
            selected[np.asarray(core.emitting_layer_indices, dtype=int)] = True
            concentration = np.where(selected, concentration, 0.0)
        imfp = np.asarray([core.imfp_by_material[material] for material in materials], dtype=float)
        return (
            self.jnp.asarray(concentration, dtype=self.jnp.float64),
            self.jnp.asarray(1.0 / imfp, dtype=self.jnp.float64),
            float(core.emission_angle_deg),
        )

    def simulate_curves(self, physical_vector):
        """Return reflectivity and normalized rocking curves for one trial vector."""

        jnp = self.jnp
        vector = jnp.asarray(physical_vector, dtype=jnp.float64)
        finite_thicknesses, nominal_roughness = self._layer_arrays(vector)
        widths = finite_thicknesses[self.cell_nominal_index - 1] / self.cell_slice_count
        centers = jnp.cumsum(widths) - 0.5 * widths
        boundaries = jnp.concatenate((jnp.zeros((1,), dtype=jnp.float64), jnp.cumsum(finite_thicknesses)))
        cell_delta = self._graded_optical_property(centers, boundaries, nominal_roughness, self.nominal_delta)
        cell_beta = self._graded_optical_property(centers, boundaries, nominal_roughness, self.nominal_beta)
        effective_thicknesses = jnp.concatenate((jnp.zeros((1,), dtype=jnp.float64), widths, jnp.zeros((1,), dtype=jnp.float64)))
        effective_delta = jnp.concatenate((self.nominal_delta[:1], cell_delta, self.nominal_delta[-1:]))
        effective_beta = jnp.concatenate((self.nominal_beta[:1], cell_beta, self.nominal_beta[-1:]))

        reflectivity_angles = self.angles + self._angle_offset(vector, reflectivity=True)
        rocking_angles = self.angles + self._angle_offset(vector, reflectivity=False)
        reflectivity = self._reflectivity(
            reflectivity_angles,
            effective_thicknesses,
            effective_delta,
            effective_beta,
        )
        field_intensity = self._field_intensity(
            rocking_angles,
            effective_thicknesses,
            effective_delta,
            effective_beta,
            centers,
        )
        curves = tuple(
            self._normalized_curve(
                field_intensity,
                widths,
                boundaries,
                nominal_roughness,
                concentration,
                attenuation_coefficient,
                emission_angle,
            )
            for concentration, attenuation_coefficient, emission_angle in self.core_inputs
        )
        return reflectivity, curves

    def _layer_arrays(self, vector):
        jnp = self.jnp
        thicknesses = []
        roughnesses = []
        for layer in self.spec.stack:
            variables = self._variables(vector, layer)
            thicknesses.append(
                _evaluate_jax_number(
                    layer.thickness_expr,
                    variables,
                    jnp=self.jnp,
                    jax=self.jax,
                    label=f"stack layer {layer.id!r} thickness_A",
                )
            )
            roughnesses.append(
                _evaluate_jax_number(
                    layer.roughness_expr,
                    variables,
                    jnp=self.jnp,
                    jax=self.jax,
                    label=f"stack layer {layer.id!r} roughness_A",
                )
            )
        return jnp.stack(thicknesses[1:-1]), jnp.stack(roughnesses)

    def _variables(self, vector, layer) -> dict[str, Any]:
        variables: dict[str, Any] = {
            name: self.jnp.asarray(value, dtype=self.jnp.float64)
            for name, value in self.default_values.items()
        }
        for name, index in self.parameter_index.items():
            variables[name] = vector[index]
        variables["repeat_index"] = self.jnp.asarray(
            0.0 if layer.repeat_index is None else float(layer.repeat_index),
            dtype=self.jnp.float64,
        )
        variables["repeat_index0"] = self.jnp.asarray(
            0.0 if layer.repeat_index is None else float(layer.repeat_index - 1),
            dtype=self.jnp.float64,
        )
        variables["layer_index"] = self.jnp.asarray(float(layer.layer_index), dtype=self.jnp.float64)
        return variables

    def _angle_offset(self, vector, *, reflectivity: bool):
        specific = self.reflectivity_angle_offset_index if reflectivity else self.rocking_curve_angle_offset_index
        if specific is not None:
            return vector[specific]
        if self.angle_offset_index is not None:
            return vector[self.angle_offset_index]
        return self.jnp.asarray(0.0, dtype=self.jnp.float64)

    def _reflectivity(self, angles, thicknesses, deltas, betas):
        value = self.jnp.zeros(angles.shape, dtype=self.jnp.float64)
        if self.s_weight:
            value = value + self.s_weight * transfer_matrix_reflectivity_jax(
                angles,
                self.energy_ev,
                thicknesses,
                deltas,
                betas,
                0,
            )
        if self.p_weight:
            value = value + self.p_weight * transfer_matrix_reflectivity_jax(
                angles,
                self.energy_ev,
                thicknesses,
                deltas,
                betas,
                1,
            )
        return value

    def _field_intensity(self, angles, thicknesses, deltas, betas, centers):
        value = self.jnp.zeros((centers.size, angles.size), dtype=self.jnp.float64)
        if self.s_weight:
            value = value + self.s_weight * transfer_matrix_field_intensity_jax(
                angles,
                self.energy_ev,
                thicknesses,
                deltas,
                betas,
                centers,
                self.effective_layer_index,
                0,
            )
        if self.p_weight:
            value = value + self.p_weight * transfer_matrix_field_intensity_jax(
                angles,
                self.energy_ev,
                thicknesses,
                deltas,
                betas,
                centers,
                self.effective_layer_index,
                1,
            )
        return value

    def _graded_optical_property(self, centers, boundaries, roughnesses, nominal_values):
        """Match the maintained nearest-interface optical grading rule."""

        jnp = self.jnp
        distances = centers[:, None] - boundaries[None, :]
        nearest = jnp.argmin(jnp.abs(distances), axis=1)
        distance = distances[jnp.arange(centers.size), nearest]
        sigma = roughnesses[nearest + 1]
        safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)
        fraction = 0.5 * (1.0 + self.jax.lax.erf(distance / (jnp.sqrt(2.0) * safe_sigma)))
        mixed = (1.0 - fraction) * nominal_values[nearest] + fraction * nominal_values[nearest + 1]
        base = nominal_values[self.cell_nominal_index]
        active = (sigma > 0.0) & (jnp.abs(distance) <= 4.0 * sigma)
        return jnp.where(active, mixed, base)

    def _graded_xps_property(self, centers, boundaries, roughnesses, nominal_values):
        """Match the maintained sequential XPS property-grading rule."""

        jnp = self.jnp
        values = nominal_values[self.cell_nominal_index]
        for interface_index in range(len(nominal_values) - 1):
            sigma = roughnesses[interface_index + 1]
            safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)
            distance = centers - boundaries[interface_index]
            fraction = 0.5 * (1.0 + self.jax.lax.erf(distance / (jnp.sqrt(2.0) * safe_sigma)))
            mixed = (1.0 - fraction) * nominal_values[interface_index] + fraction * nominal_values[interface_index + 1]
            adjacent = (self.cell_nominal_index == interface_index) | (
                self.cell_nominal_index == interface_index + 1
            )
            active = (sigma > 0.0) & (jnp.abs(distance) <= 4.0 * sigma) & adjacent
            values = jnp.where(active, mixed, values)
        return values

    def _normalized_curve(
        self,
        field_intensity,
        widths,
        boundaries,
        roughnesses,
        nominal_concentration,
        nominal_attenuation_coefficient,
        emission_angle,
    ):
        jnp = self.jnp
        centers = jnp.cumsum(widths) - 0.5 * widths
        concentration = self._graded_xps_property(
            centers,
            boundaries,
            roughnesses,
            nominal_concentration,
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
        if self.normalization_mode == "mean":
            normalization = jnp.sum(jnp.where(self.offpeak_mask, raw, 0.0)) / jnp.sum(self.offpeak_mask)
            return raw / normalization
        if self.normalization_mode == "edge_polynomial":
            edge_values = raw[self.edge_indices]
            coefficients = self.edge_design_pinv @ edge_values
            background = self.full_design @ coefficients
            return raw / background
        raise ProjectValidationError("settings.normalization must be 'mean' or 'edge_polynomial'")

    def _edge_polynomial_matrices(self, angles, edge_fraction, polynomial_order):
        order = int(polynomial_order)
        if order < 0:
            raise ProjectValidationError("settings.normalization_polynomial_order must be non-negative")
        fraction = float(edge_fraction)
        if fraction > 1.0:
            fraction /= 100.0
        if not 0.0 < fraction <= 0.5:
            raise ProjectValidationError(
                "settings.normalization_edge_fraction must select between 0 and 50 percent per edge"
            )
        edge_count = max(1, int(np.ceil(fraction * angles.size)))
        if 2 * edge_count > angles.size:
            raise ProjectValidationError("settings.normalization_edge_fraction selects more than the full curve")
        edge_mask = np.zeros(angles.size, dtype=bool)
        edge_mask[:edge_count] = True
        edge_mask[-edge_count:] = True
        if np.count_nonzero(edge_mask) <= order:
            raise ProjectValidationError("not enough edge points for settings.normalization_polynomial_order")
        edge_angles = angles[edge_mask]
        edge_design = np.vander(edge_angles, N=order + 1, increasing=False)
        full_design = np.vander(angles, N=order + 1, increasing=False)
        return (
            self.jnp.asarray(np.flatnonzero(edge_mask), dtype=self.jnp.int32),
            self.jnp.asarray(np.linalg.pinv(edge_design), dtype=self.jnp.float64),
            self.jnp.asarray(full_design, dtype=self.jnp.float64),
        )


def _evaluate_jax_number(value: Any, variables: Mapping[str, Any], *, jnp, jax, label: str):
    if isinstance(value, (int, float)):
        return jnp.asarray(float(value), dtype=jnp.float64)
    if not isinstance(value, str):
        raise ProjectValidationError(f"{label} must be a number or expression string")
    text = _normalize_reference(value)
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as error:
        raise ProjectValidationError(f"invalid expression for {label}: {value!r}") from error
    for node in ast.walk(tree):
        _validate_ast_node(node, label=label)
    try:
        return _eval_jax_node(tree.body, variables, jnp=jnp, jax=jax, label=label)
    except ExpressionError as error:
        raise ProjectValidationError(str(error)) from error


def _normalize_reference(value: str) -> str:
    text = value.strip()
    if text.startswith("$"):
        name = text[1:]
        if not name.isidentifier():
            raise ProjectValidationError(f"invalid parameter reference {value!r}")
        return name
    return text


def _validate_ast_node(node: ast.AST, *, label: str) -> None:
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.UAdd,
        ast.USub,
    )
    if not isinstance(node, allowed):
        _raise_expression_error(label)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _JAX_FUNCTION_NAMES:
            raise ProjectValidationError(f"unknown function in {label}")
        if node.keywords:
            raise ProjectValidationError(f"{label} function calls do not accept keyword arguments")


def _eval_jax_node(node: ast.AST, variables: Mapping[str, Any], *, jnp, jax, label: str):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return jnp.asarray(float(node.value), dtype=jnp.float64)
        raise ExpressionError(f"{label} contains a non-numeric literal")
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise ExpressionError(f"unknown parameter or variable {node.id!r} in {label}")
        return variables[node.id]
    if isinstance(node, ast.BinOp):
        left = _eval_jax_node(node.left, variables, jnp=jnp, jax=jax, label=label)
        right = _eval_jax_node(node.right, variables, jnp=jnp, jax=jax, label=label)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    if isinstance(node, ast.UnaryOp):
        value = _eval_jax_node(node.operand, variables, jnp=jnp, jax=jax, label=label)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        arguments = [
            _eval_jax_node(argument, variables, jnp=jnp, jax=jax, label=label)
            for argument in node.args
        ]
        try:
            return _JAX_FUNCTION_NAMES[node.func.id](*arguments, jnp=jnp, jax=jax)
        except TypeError as error:
            raise ExpressionError(f"wrong argument count for {node.func.id!r} in {label}") from error
    _raise_expression_error(label)


def _jax_min(*values, jnp, jax):
    del jax
    if len(values) < 1:
        raise TypeError("min requires at least one argument")
    result = values[0]
    for value in values[1:]:
        result = jnp.minimum(result, value)
    return result


def _jax_max(*values, jnp, jax):
    del jax
    if len(values) < 1:
        raise TypeError("max requires at least one argument")
    result = values[0]
    for value in values[1:]:
        result = jnp.maximum(result, value)
    return result


def _jax_sqrt(value, *, jnp, jax):
    del jax
    return jnp.sqrt(value)


def _jax_erf(value, *, jnp, jax):
    del jnp
    return jax.lax.erf(value)


def _jax_linear_map(x, x0, x1, y0, y1, *, jnp, jax):
    del jnp, jax
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def _jax_transition_erf(x, start, end, center, width, *, jnp, jax):
    fraction = 0.5 * (1.0 + jax.lax.erf((x - center) / (jnp.sqrt(2.0) * width)))
    return (1.0 - fraction) * start + fraction * end


_JAX_FUNCTION_NAMES = {
    "min": _jax_min,
    "max": _jax_max,
    "sqrt": _jax_sqrt,
    "erf": _jax_erf,
    "linear_map": _jax_linear_map,
    "transition_erf": _jax_transition_erf,
}


def _raise_expression_error(label: str) -> None:
    functions = ", ".join(sorted(_JAX_FUNCTION_NAMES))
    raise ProjectValidationError(
        f"{label} may contain only numbers, parameter names, repeat_index, "
        "repeat_index0, layer_index, +, -, *, /, parentheses, and safe "
        f"functions: {functions}"
    )
