"""High-level simulation, fitting, diagnostics, and reporting entry points."""

from importlib import import_module

from .simulate import (
    CoreLevelRequest,
    CoreLevelResult,
    ReflectivityRequest,
    ReflectivityResult,
    RockingCurveRequest,
    RockingCurveResult,
    simulate_reflectivity,
    simulate_rocking_curve,
    simulate_rocking_curves,
)

_LAZY_EXPORTS = {
    "compute_parameter_diagnostics": (
        "swanx.diagnostics",
        "compute_parameter_diagnostics",
    ),
    "diagnostics_from_least_squares_result": (
        "swanx.diagnostics",
        "diagnostics_from_least_squares_result",
    ),
    "plot_best_fit": ("swanx.diagnostics", "plot_best_fit"),
    "plot_correlation_matrix": (
        "swanx.diagnostics",
        "plot_correlation_matrix",
    ),
    "plot_parameter_estimates": (
        "swanx.diagnostics",
        "plot_parameter_estimates",
    ),
    "FittingProblem": ("swanx.fitting", "FittingProblem"),
    "build_jax_residual_function": (
        "swanx.fitting",
        "build_jax_residual_function",
    ),
    "optimize_with_jax_gradient": (
        "swanx.fitting",
        "optimize_with_jax_gradient",
    ),
    "optimize_with_jax_least_squares": (
        "swanx.fitting",
        "optimize_with_jax_least_squares",
    ),
    "run_bayesian_fit": ("swanx.fitting", "run_bayesian_fit"),
    "run_bayesian_optimization": (
        "swanx.fitting",
        "run_bayesian_optimization",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "CoreLevelRequest",
    "CoreLevelResult",
    "FittingProblem",
    "ReflectivityRequest",
    "ReflectivityResult",
    "RockingCurveRequest",
    "RockingCurveResult",
    "build_jax_residual_function",
    "compute_parameter_diagnostics",
    "diagnostics_from_least_squares_result",
    "optimize_with_jax_gradient",
    "optimize_with_jax_least_squares",
    "plot_best_fit",
    "plot_correlation_matrix",
    "plot_parameter_estimates",
    "run_bayesian_fit",
    "run_bayesian_optimization",
    "simulate_reflectivity",
    "simulate_rocking_curve",
    "simulate_rocking_curves",
]
