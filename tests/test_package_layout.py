"""Package layout and compatibility import coverage."""

from __future__ import annotations

import inspect


def test_project_and_fitting_public_imports_remain_stable():
    import swanx as sx
    from swanx.project import init_project, inspect_project, run_project, validate_project
    from swanx.fitting import optimize_with_jax_least_squares, run_bayesian_fit

    assert sx.SimulationStack is not None
    assert init_project is not None
    assert inspect_project is not None
    assert validate_project is not None
    assert run_project is not None
    assert optimize_with_jax_least_squares is not None
    assert run_bayesian_fit is not None


def test_root_backend_shims_still_import():
    import swanx.bo as bo
    import swanx.jax_gradient as jax_gradient
    import swanx.jax_least_squares as jax_least_squares

    assert bo.run_bayesian_fit is not None
    assert jax_gradient.optimize_with_jax_gradient is not None
    assert jax_least_squares.optimize_with_jax_least_squares is not None


def test_project_runner_imports_fitting_backend_not_root_shim():
    import swanx.project.runner as runner

    source = inspect.getsource(runner)
    assert "from swanx.fitting.bo import BayesianOptimizationSettings, run_bayesian_fit" in source
    assert "from swanx.bo import" not in source
