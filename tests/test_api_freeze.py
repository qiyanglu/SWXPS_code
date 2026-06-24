"""Frozen beginner-facing API and official entry-pattern coverage."""

import numpy as np


FROZEN_API = {
    "CoreLevelRequest",
    "ReflectivityRequest",
    "RockingCurveRequest",
    "SimulationStack",
    "StackLayer",
    "compute_parameter_diagnostics",
    "plot_correlation_matrix",
    "plot_parameter_estimates",
    "simulate_reflectivity",
    "simulate_rocking_curves",
}


def test_top_level_api_is_exactly_the_frozen_surface():
    import swanx as sx

    assert set(sx.__all__) == FROZEN_API
    for name in FROZEN_API:
        assert getattr(sx, name) is not None

    for removed_name in (
        "FitParameter",
        "Layer",
        "LayerSlicingPolicy",
        "attenuation_factor",
        "optimize_with_jax_gradient",
        "optimize_with_jax_least_squares",
        "parratt_reflectivity",
        "run_bayesian_optimization",
        "stack_from_layers",
    ):
        assert not hasattr(sx, removed_name)


def test_official_import_swanx_as_sx_workflow_runs():
    import swanx as sx

    stack = sx.SimulationStack(
        (
            sx.StackLayer("vacuum", 0.0),
            sx.StackLayer("substrate", 0.0, delta=1.0e-5, beta=2.0e-7),
        )
    )
    request = sx.ReflectivityRequest(
        angles=np.array([0.5, 1.0]),
        energy_ev=8000.0,
        stack=stack,
    )
    result = sx.simulate_reflectivity(request)

    assert result.reflectivity.shape == (2,)
    assert np.all(np.isfinite(result.reflectivity))
