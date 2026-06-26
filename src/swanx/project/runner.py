"""Run and validate YAML-backed SWANX projects."""

from __future__ import annotations

from datetime import datetime
import importlib
from pathlib import Path
from typing import Any

from swanx.bo import BayesianOptimizationSettings, run_bayesian_fit
from swanx.fitting import initial_vector
from swanx.workflows.simulate import (
    ReflectivityRequest,
    RockingCurveRequest,
    simulate_reflectivity,
    simulate_rocking_curves,
)

from .builder import BuiltProject, angles_from_settings, build_project, project_polarization
from .reports import (
    prepare_output_dir,
    write_experimental_files,
    write_fit_files,
    write_fit_summary,
    write_input_files,
    write_method_outputs,
    write_plots,
    write_resolved_files,
    write_simulation_files,
)
from .spec import ProjectSpec, ProjectValidationError, load_project_spec


def validate_project(path: str | Path) -> ProjectSpec:
    """Parse and validate a YAML ProjectSpec."""

    spec = load_project_spec(path)
    build_project(spec)
    return spec


def run_project(path: str | Path) -> Path:
    """Run a YAML ProjectSpec and write a report folder."""

    spec = load_project_spec(path)
    built = build_project(spec)
    output = prepare_output_dir(spec)
    timestamp = datetime.now().isoformat(timespec="seconds")
    write_input_files(output, built, timestamp)
    write_resolved_files(output, built)

    result = _run_backend(built)
    best_values = _best_values(built, result)
    final_built = build_project(spec, best_values)
    simulation = _simulate(final_built)
    evaluation = (
        None
        if final_built.fitting_problem is None
        else final_built.fitting_problem.evaluate(best_values)
    )

    write_simulation_files(output, simulation)
    write_experimental_files(output, final_built)
    write_fit_files(output, final_built, simulation, evaluation, result)
    write_fit_summary(output, final_built, timestamp=timestamp, result=result, evaluation=evaluation)
    write_method_outputs(output, spec.fit_method, result)
    write_plots(output, final_built, simulation)
    return output


def _run_backend(built: BuiltProject) -> Any:
    method = built.spec.fit_method
    if method == "simulate_only":
        return None
    if built.fitting_problem is None:
        raise ProjectValidationError(f"{method} requires at least one dataset")
    if method == "bayesian_optimization":
        settings = built.spec.settings.get("optimizer", {})
        return run_bayesian_fit(
            built.fitting_problem,
            BayesianOptimizationSettings(
                n_calls=int(settings.get("n_calls", 40)),
                n_initial_points=int(settings.get("n_initial_points", 10)),
                acquisition_function=str(settings.get("acquisition_function", "EI")),
                random_state=settings.get("random_state"),
                show_progress=bool(settings.get("show_progress", False)),
            ),
        )
    if method == "jax_least_squares":
        settings = built.spec.settings.get("optimizer", {})
        factory_path = settings.get("residual_function_factory")
        if not factory_path:
            raise ProjectValidationError(
                "settings.fit_method='jax_least_squares' requires "
                "settings.optimizer.residual_function_factory pointing to a fixed-shape "
                "factory callback; ProjectSpec v1 does not synthesize one automatically."
            )
        from swanx.fitting import JaxLeastSquaresOptimizerSettings, optimize_with_jax_least_squares

        residual_function = _load_callable(factory_path)(built.fitting_problem)
        return optimize_with_jax_least_squares(
            built.fitting_problem.parameters,
            residual_function,
            settings=JaxLeastSquaresOptimizerSettings(
                max_nfev=settings.get("max_nfev", 100),
                ftol=settings.get("ftol", 1.0e-8),
                xtol=settings.get("xtol", 1.0e-8),
                gtol=settings.get("gtol", 1.0e-8),
                record_history=bool(settings.get("record_history", True)),
                estimate_covariance=bool(settings.get("estimate_covariance", True)),
            ),
        )
    if method == "jax_gradient":
        settings = built.spec.settings.get("optimizer", {})
        factory_path = settings.get("value_and_grad_factory")
        if not factory_path:
            raise ProjectValidationError(
                "settings.fit_method='jax_gradient' requires "
                "settings.optimizer.value_and_grad_factory pointing to a fixed-shape "
                "factory callback; ProjectSpec v1 does not synthesize one automatically."
            )
        from swanx.fitting import JaxGradientOptimizerSettings, optimize_with_jax_gradient

        value_and_grad = _load_callable(factory_path)(built.fitting_problem)
        return optimize_with_jax_gradient(
            built.fitting_problem.parameters,
            value_and_grad,
            settings=JaxGradientOptimizerSettings(
                maxiter=int(settings.get("maxiter", 100)),
                ftol=settings.get("ftol"),
                gtol=settings.get("gtol"),
                record_history=bool(settings.get("record_history", True)),
            ),
        )
    raise ProjectValidationError(f"unknown fit method {method!r}")


def _best_values(built: BuiltProject, result: Any) -> dict[str, float]:
    if result is not None and hasattr(result, "best_parameters"):
        values = dict(built.values)
        values.update({name: float(value) for name, value in result.best_parameters.items()})
        return values
    if built.fitting_problem is not None:
        return {
            parameter.name: float(value)
            for parameter, value in zip(
                built.fitting_problem.parameters,
                initial_vector(built.fitting_problem.parameters),
            )
        }
    return dict(built.values)


def _simulate(built: BuiltProject):
    if built.fitting_problem is not None:
        return built.fitting_problem.simulate(built.values)
    angles = angles_from_settings(built.spec)
    reflectivity = simulate_reflectivity(
        ReflectivityRequest(
            angles=angles,
            energy_ev=built.spec.photon_energy_ev,
            stack=built.stack,
            polarization=project_polarization(str(built.spec.settings.get("polarization", "s"))),
        )
    )
    rocking = None
    if built.core_levels:
        rocking = simulate_rocking_curves(
            RockingCurveRequest(
                angles=angles,
                photon_energy_ev=built.spec.photon_energy_ev,
                stack=built.stack,
                core_levels=built.core_levels,
                polarization=project_polarization(str(built.spec.settings.get("polarization", "s"))),
                normalization_mode=str(built.spec.settings.get("normalization", "mean")),
            )
        )
    return type(
        "ProjectSimulation",
        (),
        {
            "parameters": dict(built.values),
            "stack": built.stack,
            "reflectivity": reflectivity,
            "rocking_curves": rocking,
        },
    )()


def _load_callable(dotted_path: str):
    module_name, _, attribute = str(dotted_path).partition(":")
    if not module_name or not attribute:
        raise ProjectValidationError("optimizer factory must be written as 'module:function'")
    module = importlib.import_module(module_name)
    callback = getattr(module, attribute)
    if not callable(callback):
        raise ProjectValidationError(f"optimizer factory {dotted_path!r} is not callable")
    return callback
