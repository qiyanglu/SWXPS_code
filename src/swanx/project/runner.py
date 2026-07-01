"""Run and validate YAML-backed SWANX projects."""

from __future__ import annotations

from datetime import datetime
import importlib
from pathlib import Path
import sys
from typing import Any, Callable

from swanx.fitting.bo import BayesianOptimizationSettings, run_bayesian_fit
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
    write_identifiability_outputs,
    write_input_files,
    write_markdown_report,
    write_method_outputs,
    write_plots,
    write_resolved_files,
    write_simulation_files,
)
from .spec import ProjectSpec, ProjectValidationError, load_project_spec


ProgressReporter = bool | Callable[[str], None]


def validate_project(path: str | Path) -> ProjectSpec:
    """Parse and validate a YAML ProjectSpec."""

    spec = load_project_spec(path)
    build_project(spec)
    return spec


def run_project(path: str | Path, *, progress: ProgressReporter = False) -> Path:
    """Run a YAML ProjectSpec and write a report folder."""

    emit = _progress_emitter(progress)
    emit(f"Reading ProjectSpec: {path}")
    spec = load_project_spec(path)
    emit(f"Building project objects for {spec.name!r}")
    built = build_project(spec)
    dataset_count = (1 if built.reflectivity_data is not None else 0) + len(built.rocking_curve_data)
    emit(
        "Loaded "
        f"{len(built.spec.stack)} layers, {len(built.core_levels)} core levels, "
        f"{dataset_count} dataset(s), {len(built.spec.varying_parameters())} varying parameter(s)"
    )
    output = prepare_output_dir(spec)
    emit(f"Writing run inputs and resolved project files: {output}")
    timestamp = datetime.now().isoformat(timespec="seconds")
    write_input_files(output, built, timestamp)
    write_resolved_files(output, built)

    result = _run_backend(built, emit=emit)
    best_values = _best_values(built, result)
    emit("Rebuilding project with final parameter values")
    final_built = build_project(spec, best_values)
    emit("Simulating final curves")
    simulation = _simulate(final_built)
    emit("Evaluating final objective and residuals")
    evaluation = (
        None
        if final_built.fitting_problem is None
        else final_built.fitting_problem.evaluate(best_values)
    )

    emit("Writing simulation, data, fit, optimizer, and plot reports")
    write_simulation_files(output, simulation)
    write_experimental_files(output, final_built)
    write_fit_files(output, final_built, simulation, evaluation, result)
    write_fit_summary(output, final_built, timestamp=timestamp, result=result, evaluation=evaluation)
    method_notes = write_method_outputs(output, spec.fit_method, result, final_built)
    identifiability_notes = write_identifiability_outputs(output, result, final_built)
    skipped_outputs = method_notes + identifiability_notes + write_plots(output, final_built, simulation)
    write_markdown_report(
        output,
        final_built,
        timestamp=timestamp,
        result=result,
        evaluation=evaluation,
        skipped_outputs=skipped_outputs,
    )
    emit(f"Done. Results written to: {output}")
    return output


def _run_backend(built: BuiltProject, *, emit: Callable[[str], None] | None = None) -> Any:
    emit = _null_progress if emit is None else emit
    method = built.spec.fit_method
    if method == "simulate_only":
        emit("Fit method is simulate_only; skipping optimizer")
        return None
    if built.fitting_problem is None:
        raise ProjectValidationError(f"{method} requires at least one dataset")
    if method == "bayesian_optimization":
        settings = built.spec.optimizer_settings
        emit(
            "Running bayesian_optimization "
            f"(n_calls={int(settings.get('n_calls', 40))}, "
            f"n_initial_points={int(settings.get('n_initial_points', 10))})"
        )
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
        settings = built.spec.optimizer_settings
        factory_path = settings.get("residual_function_factory")
        from swanx.fitting import JaxLeastSquaresOptimizerSettings, optimize_with_jax_least_squares

        if factory_path:
            emit(f"Loading JAX least-squares residual factory: {factory_path}")
            residual_function = _load_callable(factory_path, built.spec.root_dir)(built.fitting_problem)
        else:
            residual = str(settings.get("residual", "auto_fixed_grid"))
            if residual not in {"auto", "auto_fixed_grid"}:
                raise ProjectValidationError(
                    "run.optimizer.residual must be 'auto_fixed_grid' or use "
                    "run.optimizer.residual_function_factory='module:function'"
                )
            from .jax_fixed_grid import build_projectspec_jax_residual_function

            emit("Building ProjectSpec auto fixed-grid JAX residual")
            residual_function = build_projectspec_jax_residual_function(built)
        emit(f"Running jax_least_squares (max_nfev={settings.get('max_nfev', 100)})")
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
        settings = built.spec.optimizer_settings
        factory_path = settings.get("value_and_grad_factory")
        if not factory_path:
            raise ProjectValidationError(
                "run.mode='jax_gradient' requires "
                "run.optimizer.value_and_grad_factory='module:function' for the "
                "fixed-shape JAX value-and-gradient callback. Install with python -m pip install -e "
                "\".[project,gradient]\" and provide a factory, or use "
                "run.mode: \"simulate_only\" for simulation-only projects. "
                "Bayesian optimization is not used as a fallback."
            )
        from swanx.fitting import JaxGradientOptimizerSettings, optimize_with_jax_gradient

        emit(f"Loading JAX gradient value-and-gradient factory: {factory_path}")
        value_and_grad = _load_callable(factory_path, built.spec.root_dir)(built.fitting_problem)
        emit(f"Running jax_gradient (maxiter={int(settings.get('maxiter', 100))})")
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


def _progress_emitter(progress: ProgressReporter) -> Callable[[str], None]:
    if progress is True:
        return lambda message: print(f"[swanx] {message}", flush=True)
    if callable(progress):
        return progress
    return _null_progress


def _null_progress(_message: str) -> None:
    return None


def _best_values(built: BuiltProject, result: Any) -> dict[str, float]:
    if result is not None and hasattr(result, "best_parameters"):
        values = dict(built.values)
        values.update({name: float(value) for name, value in result.best_parameters.items()})
        return values
    if built.fitting_problem is not None:
        values = dict(built.values)
        values.update(
            {
                parameter.name: float(value)
                for parameter, value in zip(
                    built.fitting_problem.parameters,
                    initial_vector(built.fitting_problem.parameters),
                )
            }
        )
        return values
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
                normalization_edge_fraction=float(
                    built.spec.settings.get("normalization_edge_fraction", 0.10)
                ),
                normalization_polynomial_order=int(
                    built.spec.settings.get("normalization_polynomial_order", 2)
                ),
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


def _load_callable(dotted_path: str, project_root: Path | None = None):
    module_name, _, attribute = str(dotted_path).partition(":")
    if not module_name or not attribute:
        raise ProjectValidationError("optimizer factory must be written as 'module:function'")
    if project_root is not None:
        project_path = str(Path(project_root).resolve())
        if project_path not in sys.path:
            sys.path.insert(0, project_path)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        hint = ""
        if project_root is not None:
            hint = f" Project-local factories are loaded relative to {Path(project_root).resolve()}."
        raise ProjectValidationError(
            f"could not import optimizer factory module {module_name!r}.{hint} "
            "Use 'module:function' and place the module next to project.yaml, install it, "
            "or run from a directory on PYTHONPATH."
        ) from error
    callback = getattr(module, attribute)
    if not callable(callback):
        raise ProjectValidationError(f"optimizer factory {dotted_path!r} is not callable")
    return callback
