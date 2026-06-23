"""High-level simulation, fitting, diagnostics, and reporting entry points."""

from ..diagnostics import (
    compute_parameter_diagnostics,
    diagnostics_from_least_squares_result,
    plot_best_fit,
    plot_correlation_matrix,
    plot_parameter_estimates,
)
from ..fitting import (
    FittingProblem,
    build_jax_residual_function,
    optimize_with_jax_gradient,
    optimize_with_jax_least_squares,
    run_bayesian_fit,
    run_bayesian_optimization,
)
from ..simulation import (
    CoreLevelRequest,
    ReflectivityRequest,
    RockingCurveRequest,
    simulate_reflectivity,
    simulate_rocking_curve,
    simulate_rocking_curves,
)

__all__ = [name for name in globals() if not name.startswith("_")]
