"""Optimizer-independent fitting helpers for reflectivity and SW-XPS data."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal

import numpy as np

from .stack.slicing import FixedLayerGridPlan, LayerSlicingPolicy
from .simulation import (
    CoreLevelRequest,
    ReflectivityRequest,
    ReflectivityResult,
    RockingCurveRequest,
    RockingCurveResult,
    SimulationStack,
    StackLayer,
    simulate_reflectivity,
    simulate_rocking_curves,
)


@dataclass(frozen=True)
class FitParameter:
    """A named bounded scalar parameter."""

    name: str
    lower: float
    upper: float
    unit: str = ""
    initial: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("parameter name must be non-empty")
        if not np.isfinite(self.lower) or not np.isfinite(self.upper):
            raise ValueError("parameter bounds must be finite")
        if self.lower >= self.upper:
            raise ValueError("parameter lower bound must be smaller than upper bound")
        if self.initial is not None and not self.lower <= self.initial <= self.upper:
            raise ValueError("parameter initial value must be inside bounds")


@dataclass(frozen=True)
class LayerUpdate:
    """Bind one fit parameter to one attribute on one or more stack layers."""

    parameter: str
    layer_indices: tuple[int, ...]
    attribute: Literal["thickness", "roughness", "delta", "beta"]


@dataclass(frozen=True)
class ReflectivityData:
    """Experimental reflectivity curve."""

    name: str
    angles: np.ndarray
    reflectivity: np.ndarray
    sigma: np.ndarray | None = None
    weight: float = 1.0
    log_floor: float = 1.0e-12

    def __post_init__(self) -> None:
        _validate_data_arrays(self.angles, self.reflectivity, self.sigma, "reflectivity")
        if self.weight < 0:
            raise ValueError("dataset weight must be non-negative")
        if self.log_floor <= 0:
            raise ValueError("log_floor must be positive")


@dataclass(frozen=True)
class RockingCurveData:
    """Experimental normalized SW-XPS rocking curve."""

    name: str
    angles: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        _validate_data_arrays(self.angles, self.intensity, self.sigma, "rocking curve")
        if self.weight < 0:
            raise ValueError("dataset weight must be non-negative")


@dataclass(frozen=True)
class FitSimulation:
    """Simulated curves for one fitted parameter set."""

    parameters: dict[str, float]
    stack: SimulationStack
    reflectivity: ReflectivityResult | None
    rocking_curves: RockingCurveResult | None


@dataclass(frozen=True)
class FitContribution:
    """One weighted dataset contribution to the joint objective."""

    name: str
    raw: float
    weight: float

    @property
    def weighted(self) -> float:
        """Return the weighted contribution."""

        return self.weight * self.raw


@dataclass(frozen=True)
class FitEvaluation:
    """Result of one objective-function evaluation."""

    parameters: dict[str, float]
    objective: float
    contributions: tuple[FitContribution, ...]
    timings: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FitHistory:
    """A simple immutable record of fitting evaluations."""

    evaluations: tuple[FitEvaluation, ...] = ()

    def append(self, evaluation: FitEvaluation) -> FitHistory:
        """Return a new history with one additional evaluation."""

        return FitHistory(self.evaluations + (evaluation,))

    @property
    def best(self) -> FitEvaluation | None:
        """Return the lowest-objective evaluation, if any."""

        if not self.evaluations:
            return None
        return min(self.evaluations, key=lambda evaluation: evaluation.objective)


@dataclass
class JointObjective:
    """Scalar objective callable with optimizer-independent evaluation history."""

    parameters: tuple[FitParameter, ...]
    evaluate_parameters: Callable[[dict[str, float]], FitEvaluation]
    invalid_value: float = 1.0e100
    history: FitHistory = field(default_factory=FitHistory)

    def evaluate(self, vector: Sequence[float]) -> FitEvaluation:
        """Evaluate a parameter vector and update the internal history."""

        values = parameter_dict(self.parameters, vector)
        try:
            evaluation = self.evaluate_parameters(values)
        except ValueError:
            contribution = FitContribution("invalid", raw=self.invalid_value, weight=1.0)
            evaluation = FitEvaluation(
                parameters=values,
                objective=self.invalid_value,
                contributions=(contribution,),
            )
        self.history = self.history.append(evaluation)
        return evaluation

    def __call__(self, vector: Sequence[float]) -> float:
        """Return only the scalar objective for optimizer APIs."""

        return self.evaluate(vector).objective


@dataclass(frozen=True)
class FittingProblem:
    """User-facing definition of one reflectivity/SW-XPS fitting problem."""

    parameters: tuple[FitParameter, ...]
    stack_builder: Callable[[dict[str, float]], SimulationStack]
    photon_energy_ev: float
    reflectivity: ReflectivityData | None = None
    rocking_curves: tuple[RockingCurveData, ...] = ()
    core_levels: tuple[CoreLevelRequest, ...] = ()
    angle_offset_parameter: str | None = "angle_offset"
    field_step: float = 1.0
    roughness_step: float | Sequence[float] = 1.0
    roughness_profile: Literal["erf", "linear"] = "erf"
    slicing: LayerSlicingPolicy | FixedLayerGridPlan | None = None
    offpeak_mask: np.ndarray | None = None
    rocking_curve_normalization: Literal["mean", "edge_polynomial"] = "mean"
    normalization_edge_fraction: float = 0.10
    normalization_polynomial_order: int = 2
    validate_roughness: bool = True
    simulation_backend: Literal["numpy", "jax"] = "numpy"
    fixed_values: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_parameter_names(self.parameters)
        if self.reflectivity is None and not self.rocking_curves:
            raise ValueError("fitting problem requires at least one dataset")
        if self.rocking_curves and not self.core_levels:
            raise ValueError("core_levels are required when fitting rocking curves")
        if self.rocking_curves:
            _validate_rocking_curve_targets(self.rocking_curves, self.core_levels)
        if self.offpeak_mask is not None and self.rocking_curves:
            mask = np.asarray(self.offpeak_mask, dtype=bool)
            if mask.shape != self.rocking_curves[0].angles.shape:
                raise ValueError("offpeak_mask must match rocking-curve angles")
        if self.simulation_backend not in {"numpy", "jax"}:
            raise ValueError("simulation_backend must be 'numpy' or 'jax'")
        if self.rocking_curve_normalization not in {"mean", "edge_polynomial"}:
            raise ValueError(
                "rocking_curve_normalization must be 'mean' or 'edge_polynomial'"
            )

    def objective(self) -> JointObjective:
        """Return a scalar objective ready for an optimizer backend."""

        return JointObjective(self.parameters, self.evaluate)

    def evaluate(self, values: dict[str, float]) -> FitEvaluation:
        """Evaluate one named parameter set."""

        total_start = perf_counter()
        all_values = self._merge_values(values)
        stack_start = perf_counter()
        stack = self.stack_builder(all_values)
        if self.validate_roughness:
            validate_finite_layer_roughness(stack)
        stack_seconds = perf_counter() - stack_start
        angle_offset = _angle_offset(all_values, self.angle_offset_parameter)

        contributions: list[FitContribution] = []
        reflectivity_seconds = 0.0
        rocking_curve_seconds = 0.0
        scoring_seconds = 0.0
        if self.reflectivity is not None:
            reflectivity_start = perf_counter()
            reflectivity_result = self._simulate_reflectivity(
                ReflectivityRequest(
                    angles=self.reflectivity.angles,
                    energy_ev=self.photon_energy_ev,
                    stack=stack,
                    angle_offset=angle_offset,
                    roughness_step=self.roughness_step,
                    roughness_profile=self.roughness_profile,
                    slicing=self.slicing,
                )
            )
            reflectivity_seconds = perf_counter() - reflectivity_start
            scoring_start = perf_counter()
            contributions.append(
                FitContribution(
                    name=self.reflectivity.name,
                    raw=reflectivity_log_mse(
                        self.reflectivity,
                        reflectivity_result.reflectivity,
                    ),
                    weight=self.reflectivity.weight,
                )
            )
            scoring_seconds += perf_counter() - scoring_start
        if self.rocking_curves:
            rocking_curve_start = perf_counter()
            rocking_curve_result = self._simulate_rocking_curves(
                RockingCurveRequest(
                    angles=self.rocking_curves[0].angles,
                    photon_energy_ev=self.photon_energy_ev,
                    stack=stack,
                    core_levels=self.core_levels,
                    angle_offset=angle_offset,
                    field_step=self.field_step,
                    roughness_step=self.roughness_step,
                    roughness_profile=self.roughness_profile,
                    slicing=self.slicing,
                    offpeak_mask=self.offpeak_mask,
                    normalization_mode=self.rocking_curve_normalization,
                    normalization_edge_fraction=self.normalization_edge_fraction,
                    normalization_polynomial_order=self.normalization_polynomial_order,
                )
            )
            rocking_curve_seconds = perf_counter() - rocking_curve_start
            scoring_start = perf_counter()
            result_by_name = {
                core.name: core.curve.intensity
                for core in rocking_curve_result.core_levels
            }
            for data in self.rocking_curves:
                if data.name not in result_by_name:
                    raise ValueError(f"missing simulated core level {data.name!r}")
                contributions.append(
                    FitContribution(
                        name=data.name,
                        raw=rocking_curve_mse(data, result_by_name[data.name]),
                        weight=data.weight,
                    )
                )
            scoring_seconds += perf_counter() - scoring_start
        return evaluation_from_contributions(
            all_values,
            contributions,
            timings={
                "stack_seconds": stack_seconds,
                "reflectivity_simulation_seconds": reflectivity_seconds,
                "rocking_curve_simulation_seconds": rocking_curve_seconds,
                "scoring_seconds": scoring_seconds,
                "objective_total_seconds": perf_counter() - total_start,
            },
        )

    def simulate(self, values: dict[str, float]) -> FitSimulation:
        """Simulate all datasets for one named parameter set."""

        all_values = self._merge_values(values)
        stack = self.stack_builder(all_values)
        if self.validate_roughness:
            validate_finite_layer_roughness(stack)
        angle_offset = _angle_offset(all_values, self.angle_offset_parameter)
        reflectivity_result = None
        if self.reflectivity is not None:
            reflectivity_result = self._simulate_reflectivity(
                ReflectivityRequest(
                    angles=self.reflectivity.angles,
                    energy_ev=self.photon_energy_ev,
                    stack=stack,
                    angle_offset=angle_offset,
                    roughness_step=self.roughness_step,
                    roughness_profile=self.roughness_profile,
                    slicing=self.slicing,
                )
            )
        rocking_curve_result = None
        if self.rocking_curves:
            rocking_curve_result = self._simulate_rocking_curves(
                RockingCurveRequest(
                    angles=self.rocking_curves[0].angles,
                    photon_energy_ev=self.photon_energy_ev,
                    stack=stack,
                    core_levels=self.core_levels,
                    angle_offset=angle_offset,
                    field_step=self.field_step,
                    roughness_step=self.roughness_step,
                    roughness_profile=self.roughness_profile,
                    slicing=self.slicing,
                    offpeak_mask=self.offpeak_mask,
                    normalization_mode=self.rocking_curve_normalization,
                    normalization_edge_fraction=self.normalization_edge_fraction,
                    normalization_polynomial_order=self.normalization_polynomial_order,
                )
            )
        return FitSimulation(
            parameters=dict(all_values),
            stack=stack,
            reflectivity=reflectivity_result,
            rocking_curves=rocking_curve_result,
        )

    def _merge_values(self, values: dict[str, float]) -> dict[str, float]:
        merged = {name: float(value) for name, value in self.fixed_values.items()}
        merged.update({name: float(value) for name, value in values.items()})
        return merged

    def _simulate_reflectivity(self, request: ReflectivityRequest) -> ReflectivityResult:
        if self.simulation_backend == "numpy":
            return simulate_reflectivity(request)
        from .simulation_jax import simulate_reflectivity_jax

        return simulate_reflectivity_jax(request)

    def _simulate_rocking_curves(self, request: RockingCurveRequest) -> RockingCurveResult:
        if self.simulation_backend == "numpy":
            return simulate_rocking_curves(request)
        from .simulation_jax import simulate_rocking_curves_jax

        return simulate_rocking_curves_jax(request)


def parameter_dict(
    parameters: Sequence[FitParameter],
    vector: Sequence[float],
) -> dict[str, float]:
    """Map a vector to a named parameter dictionary."""

    if len(parameters) != len(vector):
        raise ValueError("parameter vector length does not match parameters")
    names = [parameter.name for parameter in parameters]
    if len(set(names)) != len(names):
        raise ValueError("parameter names must be unique")
    values = {parameter.name: float(value) for parameter, value in zip(parameters, vector)}
    for parameter in parameters:
        value = values[parameter.name]
        if not parameter.lower <= value <= parameter.upper:
            raise ValueError(f"parameter {parameter.name!r} is outside bounds")
    return values


def initial_vector(parameters: Sequence[FitParameter]) -> list[float]:
    """Return initial values when available, otherwise parameter midpoints."""

    return [
        float(parameter.initial)
        if parameter.initial is not None
        else 0.5 * (parameter.lower + parameter.upper)
        for parameter in parameters
    ]


def stack_with_updates(
    stack: SimulationStack,
    values: dict[str, float],
    updates: Sequence[LayerUpdate],
    validate_roughness: bool = True,
) -> SimulationStack:
    """Return a new stack with selected layer attributes updated."""

    layers = list(stack.layers)
    for update in updates:
        if update.parameter not in values:
            raise ValueError(f"missing parameter {update.parameter!r}")
        value = values[update.parameter]
        for index in update.layer_indices:
            if index < 0 or index >= len(layers):
                raise ValueError("layer update index is outside stack")
            layers[index] = _updated_stack_layer(layers[index], update.attribute, value)
    updated = SimulationStack(tuple(layers))
    if validate_roughness:
        validate_finite_layer_roughness(updated)
    return updated


def validate_finite_layer_roughness(stack: SimulationStack) -> None:
    """Reject finite layers whose roughness is larger than their thickness."""

    for index, layer in enumerate(stack.layers):
        if layer.thickness <= 0:
            continue
        if layer.roughness > layer.thickness:
            raise ValueError(
                "finite layer roughness cannot exceed thickness: "
                f"layer {index} ({layer.material}) has roughness "
                f"{layer.roughness} Angstrom and thickness {layer.thickness} Angstrom"
            )


def reflectivity_log_mse(
    data: ReflectivityData,
    simulated_reflectivity: np.ndarray,
) -> float:
    """Return mean-squared log10 residuals for reflectivity."""

    measured = np.asarray(data.reflectivity, dtype=float)
    simulated = np.asarray(simulated_reflectivity, dtype=float)
    _require_same_shape(measured, simulated, "reflectivity")
    if data.log_floor <= 0:
        raise ValueError("log_floor must be positive")
    residual = np.log10(measured + data.log_floor) - np.log10(simulated + data.log_floor)
    return _weighted_mean_square(residual, data.sigma)


def rocking_curve_mse(
    data: RockingCurveData,
    simulated_intensity: np.ndarray,
) -> float:
    """Return mean-squared residuals for a normalized rocking curve."""

    measured = np.asarray(data.intensity, dtype=float)
    simulated = np.asarray(simulated_intensity, dtype=float)
    _require_same_shape(measured, simulated, "rocking curve")
    residual = measured - simulated
    return _weighted_mean_square(residual, data.sigma)


def reflectivity_contribution(
    data: ReflectivityData,
    request: ReflectivityRequest,
) -> FitContribution:
    """Simulate and score one reflectivity dataset."""

    result = simulate_reflectivity(request)
    return FitContribution(
        name=data.name,
        raw=reflectivity_log_mse(data, result.reflectivity),
        weight=data.weight,
    )


def rocking_curve_contributions(
    data: Sequence[RockingCurveData],
    request: RockingCurveRequest,
) -> tuple[FitContribution, ...]:
    """Simulate and score one multi-core-level SW-XPS request."""

    result = simulate_rocking_curves(request)
    result_by_name = {core.name: core.curve.intensity for core in result.core_levels}
    contributions = []
    for curve_data in data:
        if curve_data.name not in result_by_name:
            raise ValueError(f"missing simulated core level {curve_data.name!r}")
        contributions.append(
            FitContribution(
                name=curve_data.name,
                raw=rocking_curve_mse(curve_data, result_by_name[curve_data.name]),
                weight=curve_data.weight,
            )
        )
    return tuple(contributions)


def evaluation_from_contributions(
    parameters: dict[str, float],
    contributions: Sequence[FitContribution],
    timings: dict[str, float] | None = None,
) -> FitEvaluation:
    """Build a total objective from per-dataset contributions."""

    total = float(sum(contribution.weighted for contribution in contributions))
    return FitEvaluation(
        parameters=dict(parameters),
        objective=total,
        contributions=tuple(contributions),
        timings={} if timings is None else dict(timings),
    )


def _updated_stack_layer(
    layer: StackLayer,
    attribute: Literal["thickness", "roughness", "delta", "beta"],
    value: float,
) -> StackLayer:
    return StackLayer(
        material=layer.material,
        thickness=value if attribute == "thickness" else layer.thickness,
        delta=value if attribute == "delta" else layer.delta,
        beta=value if attribute == "beta" else layer.beta,
        roughness=value if attribute == "roughness" else layer.roughness,
    )


def _weighted_mean_square(residual: np.ndarray, sigma: np.ndarray | None) -> float:
    if sigma is None:
        return float(np.mean(residual**2))
    weights = np.asarray(sigma, dtype=float)
    _require_same_shape(residual, weights, "sigma")
    if np.any(weights <= 0):
        raise ValueError("sigma values must be positive")
    return float(np.mean((residual / weights) ** 2))


def _require_same_shape(left: np.ndarray, right: np.ndarray, name: str) -> None:
    if left.shape != right.shape:
        raise ValueError(f"{name} arrays must have the same shape")


def _validate_data_arrays(
    angles: np.ndarray,
    values: np.ndarray,
    sigma: np.ndarray | None,
    name: str,
) -> None:
    angle_array = np.asarray(angles, dtype=float)
    value_array = np.asarray(values, dtype=float)
    _require_same_shape(angle_array, value_array, name)
    if sigma is not None:
        sigma_array = np.asarray(sigma, dtype=float)
        _require_same_shape(value_array, sigma_array, "sigma")
        if np.any(sigma_array <= 0):
            raise ValueError("sigma values must be positive")


def _angle_offset(values: dict[str, float], parameter: str | None) -> float:
    if parameter is None:
        return 0.0
    return float(values.get(parameter, 0.0))


def _validate_parameter_names(parameters: Sequence[FitParameter]) -> None:
    names = [parameter.name for parameter in parameters]
    if len(set(names)) != len(names):
        raise ValueError("parameter names must be unique")


def _validate_rocking_curve_targets(
    rocking_curves: Sequence[RockingCurveData],
    core_levels: Sequence[CoreLevelRequest],
) -> None:
    reference_angles = np.asarray(rocking_curves[0].angles, dtype=float)
    for data in rocking_curves[1:]:
        if not np.allclose(np.asarray(data.angles, dtype=float), reference_angles):
            raise ValueError("all rocking curves in one fitting problem must share angles")
    core_names = {core.name for core in core_levels}
    for data in rocking_curves:
        if data.name not in core_names:
            raise ValueError(f"rocking-curve data {data.name!r} has no matching core level")
