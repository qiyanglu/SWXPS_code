"""Stage 5 canonical stack-model and simulation-workflow locations."""

import numpy as np


def test_stack_model_and_compatibility_paths_share_objects():
    import swanx as sx
    from swanx.simulation import (
        SimulationStack as old_stack,
        StackLayer as old_layer,
        stack_from_layers as old_builder,
    )
    from swanx.stack import SimulationStack, StackLayer, stack_from_layers
    from swanx.stack.model import (
        SimulationStack as canonical_stack,
        StackLayer as canonical_layer,
        stack_from_layers as canonical_builder,
    )
    from swxps.simulation import SimulationStack as legacy_stack
    from swxps.simulation import StackLayer as legacy_layer

    assert canonical_stack.__module__ == "swanx.stack.model"
    assert canonical_layer.__module__ == "swanx.stack.model"
    assert canonical_builder.__module__ == "swanx.stack.model"
    assert SimulationStack is old_stack is legacy_stack is sx.SimulationStack
    assert StackLayer is old_layer is legacy_layer is sx.StackLayer
    assert stack_from_layers is old_builder is canonical_builder


def test_workflow_and_compatibility_paths_share_objects():
    import swanx as sx
    from swanx.simulation import ReflectivityRequest as old_request
    from swanx.simulation import simulate_reflectivity as old_simulate
    from swanx.workflows import ReflectivityRequest, simulate_reflectivity
    from swanx.workflows.simulate import ReflectivityRequest as canonical_request
    from swanx.workflows.simulate import simulate_reflectivity as canonical_simulate
    from swxps.simulation import ReflectivityRequest as legacy_request
    from swxps.simulation import simulate_reflectivity as legacy_simulate

    assert canonical_request.__module__ == "swanx.workflows.simulate"
    assert canonical_simulate.__module__ == "swanx.workflows.simulate"
    assert ReflectivityRequest is old_request is legacy_request is sx.ReflectivityRequest
    assert simulate_reflectivity is old_simulate is legacy_simulate
    assert simulate_reflectivity is canonical_simulate is sx.simulate_reflectivity


def test_workflow_facade_keeps_lazy_fitting_and_diagnostics_exports():
    from swanx.diagnostics import compute_parameter_diagnostics as diagnostics
    from swanx.fitting import FittingProblem as fitting_problem
    from swanx.workflows import FittingProblem, compute_parameter_diagnostics

    assert FittingProblem is fitting_problem
    assert compute_parameter_diagnostics is diagnostics


def test_canonical_workflow_reflectivity_smoke():
    from swanx.stack import SimulationStack, StackLayer
    from swanx.workflows import ReflectivityRequest, simulate_reflectivity

    stack = SimulationStack(
        (
            StackLayer("vacuum", 0.0),
            StackLayer("substrate", 0.0, delta=1.0e-5, beta=2.0e-7),
        )
    )
    result = simulate_reflectivity(
        ReflectivityRequest(
            angles=np.array([0.5, 1.0]),
            energy_ev=8000.0,
            stack=stack,
        )
    )

    assert result.reflectivity.shape == (2,)
    assert np.all(np.isfinite(result.reflectivity))
    assert np.all(result.reflectivity >= 0.0)
    assert np.all(result.reflectivity <= 1.0 + 1.0e-12)
