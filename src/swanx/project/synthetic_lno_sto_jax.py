"""JAX residual helper for the packaged C/LaNiO3/SrTiO3 starter project."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from swanx.fitting import JaxLeastSquaresResidualSettings, build_jax_residual_function
from swanx.reflectivity_jax import (
    transfer_matrix_field_intensity_jax,
    transfer_matrix_reflectivity_jax,
)
from swanx.stack.slicing import (
    FixedLayerGridPlan,
    LayerSlicingPolicy,
    fixed_layer_grid,
    fixed_layer_grid_plan,
)


PARAMETER_NAMES = (
    "carbon_thickness",
    "carbon_roughness_fraction",
    "lno_thickness",
    "sto_thickness",
    "superlattice_roughness",
    "substrate_roughness",
    "angle_offset",
)


def build_residual_function(problem):
    """Return the fixed-shape JAX residual for the starter fitting project."""

    if problem.reflectivity is None:
        raise ValueError("the starter JAX fit requires a reflectivity dataset")
    if len(problem.rocking_curves) != 4:
        raise ValueError("the starter JAX fit requires La 4d, O 1s, Ti 2p, and C 1s datasets")
    parameter_names = tuple(parameter.name for parameter in problem.parameters)
    missing = [name for name in PARAMETER_NAMES if name not in parameter_names]
    if missing:
        raise ValueError("starter JAX fit is missing parameters: " + ", ".join(missing))

    capacity_stack = problem.stack_builder(_capacity_values(problem))
    plan = (
        problem.slicing
        if isinstance(problem.slicing, FixedLayerGridPlan)
        else fixed_layer_grid_plan(
            capacity_stack.optical_layers,
            LayerSlicingPolicy(min_slices=10, max_slice_thickness=2.0),
        )
    )
    angles = np.asarray(problem.reflectivity.angles, dtype=float)
    offpeak_mask = (
        np.ones(angles.shape, dtype=bool)
        if problem.offpeak_mask is None
        else np.asarray(problem.offpeak_mask, dtype=bool)
    )
    model = _StarterSyntheticJaxModel(
        angles=angles,
        offpeak_mask=offpeak_mask,
        core_levels=problem.core_levels,
        plan=plan,
        capacity_stack=capacity_stack,
        parameter_names=parameter_names,
    )
    return build_jax_residual_function(
        model.simulate_curves,
        reflectivity=replace(problem.reflectivity, log_floor=1.0e-12),
        rocking_curves=problem.rocking_curves,
        settings=JaxLeastSquaresResidualSettings(
            reflectivity_log=True,
            rocking_curve_normalization="mean_absolute",
        ),
    )


def _capacity_values(problem) -> dict[str, float]:
    values = {}
    for parameter in problem.parameters:
        if parameter.name in {"carbon_thickness", "lno_thickness", "sto_thickness"}:
            values[parameter.name] = float(parameter.upper)
        elif parameter.name == "angle_offset":
            values[parameter.name] = 0.0
        else:
            values[parameter.name] = float(parameter.initial if parameter.initial is not None else parameter.upper)
    return values


class _StarterSyntheticJaxModel:
    """Fixed-grid model for C/[LaNiO3/SrTiO3]xN/SrTiO3 starter fits."""

    def __init__(self, *, angles, offpeak_mask, core_levels, plan, capacity_stack, parameter_names):
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        self.jax = jax
        self.jnp = jnp
        self.angles = jnp.asarray(angles, dtype=jnp.float64)
        self.offpeak_mask = jnp.asarray(offpeak_mask, dtype=bool)
        self.parameter_index = {name: index for index, name in enumerate(parameter_names)}
        self.repeats = (len(capacity_stack.optical_layers) - 3) // 2
        if self.repeats <= 0 or len(capacity_stack.optical_layers) != 2 * self.repeats + 3:
            raise ValueError("starter JAX fit expects vacuum/C/[LNO/STO]xN/STO stack topology")

        capacity_grid = fixed_layer_grid(capacity_stack.optical_layers, plan)
        nominal_index = np.asarray(capacity_grid.nominal_layer_index, dtype=np.int32)
        count_by_layer = np.asarray(plan.slice_counts, dtype=float)
        self.cell_nominal_index = jnp.asarray(nominal_index, dtype=jnp.int32)
        self.cell_slice_count = jnp.asarray(count_by_layer[nominal_index - 1], dtype=jnp.float64)
        self.effective_layer_index = jnp.asarray(capacity_grid.effective_layer_index, dtype=jnp.int32)
        optical_layers = capacity_stack.optical_layers
        self.nominal_delta = jnp.asarray([layer.delta for layer in optical_layers], dtype=jnp.float64)
        self.nominal_beta = jnp.asarray([layer.beta for layer in optical_layers], dtype=jnp.float64)
        self.core_inputs = tuple(self._core_inputs(core, capacity_stack.materials) for core in core_levels)

    def _core_inputs(self, core, materials):
        concentration = np.asarray(
            [core.concentration_by_material.get(material, 0.0) for material in materials],
            dtype=float,
        )
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
        carbon = vector[self.parameter_index["carbon_thickness"]]
        carbon_roughness_fraction = vector[self.parameter_index["carbon_roughness_fraction"]]
        lno = vector[self.parameter_index["lno_thickness"]]
        sto = vector[self.parameter_index["sto_thickness"]]
        super_roughness = vector[self.parameter_index["superlattice_roughness"]]
        substrate_roughness = vector[self.parameter_index["substrate_roughness"]]
        angle_offset = vector[self.parameter_index["angle_offset"]]

        period = jnp.stack((lno, sto))
        finite_thicknesses = jnp.concatenate(
            (jnp.reshape(carbon, (1,)), jnp.tile(period, self.repeats))
        )
        carbon_roughness = 1.0 + carbon_roughness_fraction * (jnp.minimum(5.0, carbon) - 1.0)
        finite_roughness = jnp.concatenate(
            (
                jnp.reshape(carbon_roughness, (1,)),
                jnp.full((2 * self.repeats,), super_roughness),
            )
        )
        nominal_roughness = jnp.concatenate(
            (jnp.zeros((1,)), finite_roughness, jnp.reshape(substrate_roughness, (1,)))
        )

        widths = finite_thicknesses[self.cell_nominal_index - 1] / self.cell_slice_count
        centers = jnp.cumsum(widths) - 0.5 * widths
        boundaries = jnp.concatenate((jnp.zeros((1,)), jnp.cumsum(finite_thicknesses)))
        cell_delta = self._graded_optical_property(centers, boundaries, nominal_roughness, self.nominal_delta)
        cell_beta = self._graded_optical_property(centers, boundaries, nominal_roughness, self.nominal_beta)
        effective_thicknesses = jnp.concatenate((jnp.zeros((1,)), widths, jnp.zeros((1,))))
        effective_delta = jnp.concatenate((self.nominal_delta[:1], cell_delta, self.nominal_delta[-1:]))
        effective_beta = jnp.concatenate((self.nominal_beta[:1], cell_beta, self.nominal_beta[-1:]))

        calculation_angles = self.angles + angle_offset
        reflectivity = transfer_matrix_reflectivity_jax(
            calculation_angles,
            1000.0,
            effective_thicknesses,
            effective_delta,
            effective_beta,
        )
        field_intensity = transfer_matrix_field_intensity_jax(
            calculation_angles,
            1000.0,
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
                boundaries,
                nominal_roughness,
                concentration,
                attenuation_coefficient,
                emission_angle,
            )
            for concentration, attenuation_coefficient, emission_angle in self.core_inputs
        )
        return reflectivity, curves

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
        normalization = jnp.sum(jnp.where(self.offpeak_mask, raw, 0.0)) / jnp.sum(self.offpeak_mask)
        return raw / normalization
