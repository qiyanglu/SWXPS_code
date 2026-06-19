"""Standalone gradient-based optimizer for JAX differentiable losses."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .fitting import FitParameter, initial_vector


ValueAndGradient = Callable[[np.ndarray], tuple[float, np.ndarray]]


@dataclass(frozen=True)
class JaxGradientOptimizerSettings:
    """Settings for the first local JAX-gradient optimizer."""

    maxiter: int = 100
    ftol: float | None = None
    gtol: float | None = None
    record_history: bool = True

    def __post_init__(self) -> None:
        if self.maxiter <= 0:
            raise ValueError("maxiter must be positive")
        if self.ftol is not None and self.ftol <= 0:
            raise ValueError("ftol must be positive")
        if self.gtol is not None and self.gtol <= 0:
            raise ValueError("gtol must be positive")


@dataclass(frozen=True)
class JaxGradientHistoryRecord:
    """One recorded L-BFGS-B iteration."""

    iteration: int
    loss: float
    parameters: dict[str, float]
    gradient_norm: float


@dataclass(frozen=True)
class JaxGradientOptimizationResult:
    """Package-native result for local JAX-gradient optimization."""

    best_parameters: dict[str, float]
    best_loss: float
    status: int
    message: str
    success: bool
    nit: int
    nfev: int
    history: tuple[JaxGradientHistoryRecord, ...]
    raw_result: object
    total_seconds: float


def optimize_with_jax_gradient(
    parameters: Sequence[FitParameter],
    value_and_grad: ValueAndGradient,
    initial: Sequence[float] | None = None,
    settings: JaxGradientOptimizerSettings | None = None,
) -> JaxGradientOptimizationResult:
    """Minimize a physical-space JAX value-and-gradient callback with L-BFGS-B.

    The SciPy optimizer runs in scaled coordinates with every bounded parameter
    mapped to ``[0, 1]``. The provided ``value_and_grad`` callback receives and
    returns physical-space vectors.
    """

    parameter_tuple = tuple(parameters)
    if not parameter_tuple:
        raise ValueError("at least one parameter is required")
    settings = JaxGradientOptimizerSettings() if settings is None else settings
    minimize = _load_scipy_minimize()

    lower, upper = _parameter_bounds(parameter_tuple)
    physical_initial = (
        np.asarray(initial_vector(parameter_tuple), dtype=float)
        if initial is None
        else np.asarray(initial, dtype=float)
    )
    _validate_initial(physical_initial, lower, upper)
    scaled_initial = physical_to_scaled(physical_initial, lower, upper)

    history: list[JaxGradientHistoryRecord] = []
    latest: dict[str, np.ndarray | float] = {}

    def objective(scaled_vector: np.ndarray) -> tuple[float, np.ndarray]:
        physical_vector = scaled_to_physical(scaled_vector, lower, upper)
        loss, physical_gradient = value_and_grad(np.asarray(physical_vector, dtype=float))
        loss_float = float(loss)
        gradient = np.asarray(physical_gradient, dtype=float)
        if gradient.shape != physical_vector.shape:
            raise ValueError("gradient shape must match parameter vector shape")
        scaled_gradient = gradient * (upper - lower)
        latest["loss"] = loss_float
        latest["physical_vector"] = physical_vector
        latest["physical_gradient"] = gradient
        return loss_float, scaled_gradient

    def callback(scaled_vector: np.ndarray) -> None:
        if not settings.record_history:
            return
        physical_vector = scaled_to_physical(scaled_vector, lower, upper)
        if "loss" not in latest or not np.allclose(
            physical_vector,
            latest.get("physical_vector"),
            rtol=0.0,
            atol=1e-12,
        ):
            loss, physical_gradient = value_and_grad(np.asarray(physical_vector, dtype=float))
            loss_float = float(loss)
            gradient = np.asarray(physical_gradient, dtype=float)
        else:
            loss_float = float(latest["loss"])
            gradient = np.asarray(latest["physical_gradient"], dtype=float)
        history.append(
            JaxGradientHistoryRecord(
                iteration=len(history) + 1,
                loss=loss_float,
                parameters=parameter_dict_from_vector(parameter_tuple, physical_vector),
                gradient_norm=float(np.linalg.norm(gradient)),
            )
        )

    options: dict[str, float | int] = {"maxiter": settings.maxiter}
    if settings.ftol is not None:
        options["ftol"] = settings.ftol
    if settings.gtol is not None:
        options["gtol"] = settings.gtol

    start = perf_counter()
    raw_result = minimize(
        objective,
        scaled_initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, 1.0)] * len(parameter_tuple),
        callback=callback,
        options=options,
    )
    total_seconds = perf_counter() - start

    best_vector = scaled_to_physical(np.asarray(raw_result.x, dtype=float), lower, upper)
    return JaxGradientOptimizationResult(
        best_parameters=parameter_dict_from_vector(parameter_tuple, best_vector),
        best_loss=float(raw_result.fun),
        status=int(raw_result.status),
        message=str(raw_result.message),
        success=bool(raw_result.success),
        nit=int(raw_result.nit),
        nfev=int(raw_result.nfev),
        history=tuple(history),
        raw_result=raw_result,
        total_seconds=total_seconds,
    )


def physical_to_scaled(
    physical_vector: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> np.ndarray:
    """Map physical parameter values to scaled ``[0, 1]`` coordinates."""

    physical = np.asarray(physical_vector, dtype=float)
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    return (physical - lower_array) / (upper_array - lower_array)


def scaled_to_physical(
    scaled_vector: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> np.ndarray:
    """Map scaled ``[0, 1]`` coordinates to physical parameter values."""

    scaled = np.asarray(scaled_vector, dtype=float)
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    return lower_array + scaled * (upper_array - lower_array)


def parameter_dict_from_vector(
    parameters: Sequence[FitParameter],
    vector: Sequence[float],
) -> dict[str, float]:
    """Return a named parameter dictionary from a physical vector."""

    values = np.asarray(vector, dtype=float)
    if len(parameters) != len(values):
        raise ValueError("parameter vector length does not match parameters")
    return {
        parameter.name: float(value)
        for parameter, value in zip(parameters, values)
    }


def _parameter_bounds(parameters: Sequence[FitParameter]) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray([parameter.lower for parameter in parameters], dtype=float)
    upper = np.asarray([parameter.upper for parameter in parameters], dtype=float)
    if np.any(lower >= upper):
        raise ValueError("parameter lower bounds must be smaller than upper bounds")
    return lower, upper


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


def _load_scipy_minimize():
    try:
        from scipy.optimize import minimize
    except ImportError as error:
        raise ImportError(
            "scipy is required for JAX gradient optimization; install the "
            "gradient extra or install scipy directly"
        ) from error
    return minimize
