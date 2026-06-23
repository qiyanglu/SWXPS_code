"""Standalone nonlinear least-squares optimization with JAX Jacobians."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal

import numpy as np

from ._fitting import (
    FitParameter,
    ReflectivityData,
    RockingCurveData,
    initial_vector,
)
from .jax_gradient import (
    parameter_dict_from_vector,
    physical_to_scaled,
    scaled_to_physical,
)


JaxCurveSimulator = Callable[[object], tuple[object | None, tuple[object, ...]]]


@dataclass(frozen=True)
class JaxLeastSquaresResidualSettings:
    """Weighting and normalization choices for concatenated curve residuals."""

    reflectivity_log: bool = True
    reflectivity_epsilon: float | None = None
    rocking_curve_normalization: Literal["mean_absolute", "none"] = "mean_absolute"
    rocking_curve_scales: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.reflectivity_epsilon is not None and self.reflectivity_epsilon <= 0:
            raise ValueError("reflectivity_epsilon must be positive when provided")
        if self.rocking_curve_normalization not in {"mean_absolute", "none"}:
            raise ValueError(
                "rocking_curve_normalization must be ''mean_absolute'' or ''none''"
            )
        if self.rocking_curve_scales is not None:
            scales = np.asarray(self.rocking_curve_scales, dtype=float)
            if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
                raise ValueError("rocking_curve_scales must be finite and positive")


@dataclass
class JaxCompilationCounter:
    """Count fixed-shape residual and Jacobian traces performed by JAX."""

    residual_compilations: int = 0
    jacobian_compilations: int = 0

    @property
    def total_compilations(self) -> int:
        """Return the total number of residual and Jacobian compilations."""

        return self.residual_compilations + self.jacobian_compilations


@dataclass(frozen=True)
class JaxResidualFunction:
    """JIT-compiled physical-space residual and Jacobian functions."""

    residuals_jax: Callable[[object], object]
    jacobian_jax: Callable[[object], object]
    residual_count: int
    compilation_counter: JaxCompilationCounter = field(
        default_factory=JaxCompilationCounter
    )

    def __call__(self, physical_vector: Sequence[float]) -> np.ndarray:
        """Return the residual vector as a NumPy array for SciPy."""

        return np.asarray(
            self.residuals_jax(np.asarray(physical_vector, dtype=float)),
            dtype=float,
        )

    def jacobian(self, physical_vector: Sequence[float]) -> np.ndarray:
        """Return the physical-space Jacobian as a NumPy array for SciPy."""

        return np.asarray(
            self.jacobian_jax(np.asarray(physical_vector, dtype=float)),
            dtype=float,
        )


@dataclass(frozen=True)
class JaxLeastSquaresOptimizerSettings:
    """Settings for bounded trust-region reflective least squares."""

    max_nfev: int | None = 100
    ftol: float | None = 1.0e-8
    xtol: float | None = 1.0e-8
    gtol: float | None = 1.0e-8
    record_history: bool = True
    estimate_covariance: bool = True

    def __post_init__(self) -> None:
        if self.max_nfev is not None and self.max_nfev <= 0:
            raise ValueError("max_nfev must be positive")
        for name, value in (
            ("ftol", self.ftol),
            ("xtol", self.xtol),
            ("gtol", self.gtol),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.ftol is None and self.xtol is None and self.gtol is None:
            raise ValueError("at least one convergence tolerance must be enabled")


@dataclass(frozen=True)
class JaxLeastSquaresHistoryRecord:
    """One Jacobian evaluation recorded during TRF optimization."""

    iteration: int
    cost: float
    parameters: dict[str, float]
    gradient_norm: float


@dataclass(frozen=True)
class JaxLeastSquaresOptimizationResult:
    """Package-native result for local JAX nonlinear least squares."""

    best_parameters: dict[str, float]
    final_cost: float
    final_residuals: np.ndarray
    final_jacobian: np.ndarray
    status: int
    message: str
    success: bool
    nfev: int
    njev: int | None
    optimality: float
    history: tuple[JaxLeastSquaresHistoryRecord, ...]
    covariance: np.ndarray | None
    raw_result: object
    total_seconds: float


def build_jax_residual_function(
    simulate_curves: JaxCurveSimulator,
    reflectivity: ReflectivityData | None = None,
    rocking_curves: Sequence[RockingCurveData] = (),
    settings: JaxLeastSquaresResidualSettings | None = None,
) -> JaxResidualFunction:
    """Build JIT-compiled residuals and a forward-mode JAX Jacobian.

    simulate_curves(params) must be JAX-traceable and return a pair containing
    reflectivity (or None) and a tuple of rocking curves. The tuple length and
    all array shapes must remain fixed while JIT-compiled.
    """

    if reflectivity is None and not rocking_curves:
        raise ValueError("at least one reflectivity or rocking-curve dataset is required")
    settings = JaxLeastSquaresResidualSettings() if settings is None else settings
    rocking_curve_tuple = tuple(rocking_curves)
    if (
        settings.rocking_curve_scales is not None
        and len(settings.rocking_curve_scales) != len(rocking_curve_tuple)
    ):
        raise ValueError("rocking_curve_scales must match the rocking-curve count")

    jax, jnp = _load_jax()
    reflectivity_target = (
        None
        if reflectivity is None
        else jnp.asarray(reflectivity.reflectivity, dtype=jnp.float64)
    )
    reflectivity_sigma = (
        None
        if reflectivity is None or reflectivity.sigma is None
        else jnp.asarray(reflectivity.sigma, dtype=jnp.float64)
    )
    rocking_targets = tuple(
        jnp.asarray(data.intensity, dtype=jnp.float64)
        for data in rocking_curve_tuple
    )
    rocking_sigmas = tuple(
        None if data.sigma is None else jnp.asarray(data.sigma, dtype=jnp.float64)
        for data in rocking_curve_tuple
    )
    rocking_scales = _rocking_curve_scales(rocking_curve_tuple, settings)

    def residuals(physical_vector):
        simulated_reflectivity, simulated_rocking_curves = simulate_curves(
            physical_vector
        )
        blocks = ()
        if reflectivity is not None:
            simulated = jnp.asarray(simulated_reflectivity, dtype=jnp.float64)
            if settings.reflectivity_log:
                epsilon = (
                    reflectivity.log_floor
                    if settings.reflectivity_epsilon is None
                    else settings.reflectivity_epsilon
                )
                block = jnp.log10(jnp.maximum(simulated, epsilon)) - jnp.log10(
                    jnp.maximum(reflectivity_target, epsilon)
                )
            else:
                block = simulated - reflectivity_target
            if reflectivity_sigma is not None:
                block = block / reflectivity_sigma
            block = jnp.sqrt(reflectivity.weight / block.size) * block
            blocks += (block,)

        for index, (data, target, sigma, scale) in enumerate(
            zip(
                rocking_curve_tuple,
                rocking_targets,
                rocking_sigmas,
                rocking_scales,
            )
        ):
            simulated = jnp.asarray(
                simulated_rocking_curves[index],
                dtype=jnp.float64,
            )
            block = (simulated - target) / scale
            if sigma is not None:
                block = block / sigma
            block = jnp.sqrt(data.weight / block.size) * block
            blocks += (block,)
        return jnp.concatenate(blocks)

    compilation_counter = JaxCompilationCounter()

    def counted_residuals(physical_vector):
        compilation_counter.residual_compilations += 1
        return residuals(physical_vector)

    jacobian_core = jax.jacfwd(residuals)

    def counted_jacobian(physical_vector):
        compilation_counter.jacobian_compilations += 1
        return jacobian_core(physical_vector)

    residuals_jax = jax.jit(counted_residuals)
    jacobian_jax = jax.jit(counted_jacobian)
    residual_count = (
        (0 if reflectivity is None else reflectivity.reflectivity.size)
        + sum(data.intensity.size for data in rocking_curve_tuple)
    )
    return JaxResidualFunction(
        residuals_jax=residuals_jax,
        jacobian_jax=jacobian_jax,
        residual_count=residual_count,
        compilation_counter=compilation_counter,
    )


def optimize_with_jax_least_squares(
    parameters: Sequence[FitParameter],
    residual_function: JaxResidualFunction,
    initial: Sequence[float] | None = None,
    settings: JaxLeastSquaresOptimizerSettings | None = None,
) -> JaxLeastSquaresOptimizationResult:
    """Minimize JAX residuals with bounded SciPy TRF in scaled coordinates."""

    parameter_tuple = tuple(parameters)
    if not parameter_tuple:
        raise ValueError("at least one parameter is required")
    settings = JaxLeastSquaresOptimizerSettings() if settings is None else settings
    least_squares = _load_scipy_least_squares()

    lower = np.asarray([parameter.lower for parameter in parameter_tuple], dtype=float)
    upper = np.asarray([parameter.upper for parameter in parameter_tuple], dtype=float)
    physical_initial = (
        np.asarray(initial_vector(parameter_tuple), dtype=float)
        if initial is None
        else np.asarray(initial, dtype=float)
    )
    _validate_initial(physical_initial, lower, upper)
    scaled_initial = physical_to_scaled(physical_initial, lower, upper)
    parameter_ranges = upper - lower

    history: list[JaxLeastSquaresHistoryRecord] = []
    latest_scaled: np.ndarray | None = None
    latest_residuals: np.ndarray | None = None

    def residuals_scaled(scaled_vector: np.ndarray) -> np.ndarray:
        nonlocal latest_scaled, latest_residuals
        physical_vector = scaled_to_physical(scaled_vector, lower, upper)
        residuals = residual_function(physical_vector)
        if residuals.shape != (residual_function.residual_count,):
            raise ValueError("residual vector shape changed during optimization")
        if not np.all(np.isfinite(residuals)):
            raise ValueError("residual vector contains non-finite values")
        latest_scaled = np.asarray(scaled_vector, dtype=float).copy()
        latest_residuals = residuals
        return residuals

    def jacobian_scaled(scaled_vector: np.ndarray) -> np.ndarray:
        physical_vector = scaled_to_physical(scaled_vector, lower, upper)
        physical_jacobian = residual_function.jacobian(physical_vector)
        expected_shape = (residual_function.residual_count, len(parameter_tuple))
        if physical_jacobian.shape != expected_shape:
            raise ValueError(
                f"Jacobian shape must be {expected_shape}, got {physical_jacobian.shape}"
            )
        scaled_jacobian = physical_jacobian * parameter_ranges[np.newaxis, :]
        if settings.record_history:
            if latest_scaled is not None and np.array_equal(scaled_vector, latest_scaled):
                residuals = np.asarray(latest_residuals, dtype=float)
            else:
                residuals = residual_function(physical_vector)
            history.append(
                JaxLeastSquaresHistoryRecord(
                    iteration=len(history) + 1,
                    cost=0.5 * float(np.dot(residuals, residuals)),
                    parameters=parameter_dict_from_vector(
                        parameter_tuple,
                        physical_vector,
                    ),
                    gradient_norm=float(
                        np.linalg.norm(scaled_jacobian.T @ residuals)
                    ),
                )
            )
        return scaled_jacobian

    start = perf_counter()
    raw_result = least_squares(
        residuals_scaled,
        scaled_initial,
        jac=jacobian_scaled,
        bounds=(
            np.zeros(len(parameter_tuple), dtype=float),
            np.ones(len(parameter_tuple), dtype=float),
        ),
        method="trf",
        max_nfev=settings.max_nfev,
        ftol=settings.ftol,
        xtol=settings.xtol,
        gtol=settings.gtol,
    )
    total_seconds = perf_counter() - start

    best_vector = scaled_to_physical(
        np.asarray(raw_result.x, dtype=float),
        lower,
        upper,
    )
    final_residuals = residual_function(best_vector)
    final_jacobian = residual_function.jacobian(best_vector)
    covariance = (
        _estimate_covariance(final_jacobian, final_residuals)
        if settings.estimate_covariance
        else None
    )
    raw_njev = getattr(raw_result, "njev", None)
    return JaxLeastSquaresOptimizationResult(
        best_parameters=parameter_dict_from_vector(parameter_tuple, best_vector),
        final_cost=0.5 * float(np.dot(final_residuals, final_residuals)),
        final_residuals=final_residuals,
        final_jacobian=final_jacobian,
        status=int(raw_result.status),
        message=str(raw_result.message),
        success=bool(raw_result.success),
        nfev=int(raw_result.nfev),
        njev=None if raw_njev is None else int(raw_njev),
        optimality=float(raw_result.optimality),
        history=tuple(history),
        covariance=covariance,
        raw_result=raw_result,
        total_seconds=total_seconds,
    )


def _rocking_curve_scales(
    rocking_curves: tuple[RockingCurveData, ...],
    settings: JaxLeastSquaresResidualSettings,
) -> tuple[float, ...]:
    if settings.rocking_curve_scales is not None:
        return tuple(float(scale) for scale in settings.rocking_curve_scales)
    if settings.rocking_curve_normalization == "none":
        return (1.0,) * len(rocking_curves)
    scales = []
    for data in rocking_curves:
        scale = float(np.mean(np.abs(np.asarray(data.intensity, dtype=float))))
        scales.append(scale if scale > 0 else 1.0)
    return tuple(scales)


def _validate_initial(
    initial: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    if initial.shape != lower.shape:
        raise ValueError("initial parameter vector length does not match parameters")
    if not np.all(np.isfinite(initial)):
        raise ValueError("initial parameter values must be finite")
    if np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("initial parameter values must be inside bounds")


def _estimate_covariance(
    jacobian: np.ndarray,
    residuals: np.ndarray,
    rcond: float = 1.0e-12,
) -> np.ndarray | None:
    residual_count, parameter_count = jacobian.shape
    degrees_of_freedom = residual_count - parameter_count
    if degrees_of_freedom <= 0:
        return None
    try:
        normal_inverse = np.linalg.pinv(
            jacobian.T @ jacobian,
            rcond=rcond,
        )
    except np.linalg.LinAlgError:
        return None
    residual_variance = float(np.dot(residuals, residuals)) / degrees_of_freedom
    covariance = residual_variance * normal_inverse
    return 0.5 * (covariance + covariance.T)


def _load_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:
        raise ImportError(
            "jax is required for JAX least-squares residuals; install the "
            "least-squares extra or install jax directly"
        ) from error
    jax.config.update("jax_enable_x64", True)
    return jax, jnp


def _load_scipy_least_squares():
    try:
        from scipy.optimize import least_squares
    except ImportError as error:
        raise ImportError(
            "scipy is required for JAX least-squares optimization; install the "
            "least-squares extra or install scipy directly"
        ) from error
    return least_squares
