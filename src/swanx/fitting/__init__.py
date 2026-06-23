"""Fitting parameters, objectives, and maintained optimizer backends."""

from .._fitting import (
    FitContribution,
    FitEvaluation,
    FitHistory,
    FitParameter,
    FitSimulation,
    FittingProblem,
    JointObjective,
    LayerUpdate,
    ReflectivityData,
    RockingCurveData,
    evaluation_from_contributions,
    initial_vector,
    parameter_dict,
    reflectivity_contribution,
    reflectivity_log_mse,
    rocking_curve_contributions,
    rocking_curve_mse,
    stack_with_updates,
    validate_finite_layer_roughness,
)
from ..bo import (
    BayesianOptimizationResult,
    BayesianOptimizationSettings,
    FitStage,
    OptimizationTiming,
    StageFitResult,
    StageRunResult,
    StagedFitResult,
    run_bayesian_fit,
    run_bayesian_optimization,
    run_staged_multistart_bayesian_fit,
)
from ..jax_gradient import (
    JaxGradientHistoryRecord,
    JaxGradientOptimizationResult,
    JaxGradientOptimizerSettings,
    optimize_with_jax_gradient,
    physical_to_scaled,
    scaled_to_physical,
)
from ..jax_least_squares import (
    JaxCompilationCounter,
    JaxLeastSquaresHistoryRecord,
    JaxLeastSquaresOptimizationResult,
    JaxLeastSquaresOptimizerSettings,
    JaxLeastSquaresResidualSettings,
    JaxResidualFunction,
    build_jax_residual_function,
    optimize_with_jax_least_squares,
)

__all__ = [name for name in globals() if not name.startswith("_")]
