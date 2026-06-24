from importlib import import_module

import numpy as np
import pytest

from swxps import (
    FitParameter,
    FittingProblem,
    LayerSlicingPolicy,
    ReflectivityData,
    ReflectivityRequest,
    SimulationStack,
    StackLayer,
    fixed_layer_grid_plan,
    simulate_reflectivity,
)
from swxps.reflectivity import parratt_reflectivity


def test_unified_grid_vacuum_substrate_retains_fresnel_reflectivity():
    stack = SimulationStack(
        (
            StackLayer("vacuum", 0.0),
            StackLayer("substrate", 0.0, delta=1.0e-5, beta=2.0e-7),
        )
    )
    angles = np.linspace(0.5, 5.0, 21)

    actual = simulate_reflectivity(
        ReflectivityRequest(
            angles=angles,
            energy_ev=3000.0,
            stack=stack,
            slicing=LayerSlicingPolicy(),
        )
    ).reflectivity
    expected = parratt_reflectivity(angles, 3000.0, stack.optical_layers)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


def test_fitting_problem_propagates_fixed_grid_plan():
    def build_stack(values):
        return SimulationStack(
            (
                StackLayer("vacuum", 0.0),
                StackLayer(
                    "film",
                    values["thickness"],
                    delta=5.0e-6,
                    beta=1.0e-7,
                    roughness=1.0,
                ),
                StackLayer("substrate", 0.0, delta=1.0e-5, beta=2.0e-7),
            )
        )

    angles = np.linspace(0.8, 4.0, 16)
    capacity_stack = build_stack({"thickness": 6.0})
    plan = fixed_layer_grid_plan(capacity_stack.optical_layers)
    target_stack = build_stack({"thickness": 4.0})
    target = simulate_reflectivity(
        ReflectivityRequest(
            angles=angles,
            energy_ev=3000.0,
            stack=target_stack,
            slicing=plan,
        )
    ).reflectivity
    problem = FittingProblem(
        parameters=(FitParameter("thickness", 2.0, 6.0),),
        stack_builder=build_stack,
        photon_energy_ev=3000.0,
        reflectivity=ReflectivityData("R", angles, target),
        angle_offset_parameter=None,
        slicing=plan,
    )

    at_target = problem.evaluate({"thickness": 4.0})
    away = problem.evaluate({"thickness": 5.5})

    assert at_target.objective < 1.0e-24
    assert away.objective > at_target.objective


def _minimal_reflectivity_problem(**kwargs):
    stack = SimulationStack(
        (
            StackLayer("vacuum", 0.0),
            StackLayer("substrate", 0.0, delta=1.0e-5, beta=2.0e-7),
        )
    )
    angles = np.linspace(0.8, 2.0, 5)
    return FittingProblem(
        parameters=(FitParameter("unused", 0.0, 1.0),),
        stack_builder=lambda values: stack,
        photon_energy_ev=3000.0,
        reflectivity=ReflectivityData("R", angles, np.ones_like(angles)),
        angle_offset_parameter=None,
        **kwargs,
    )


def test_fitting_problem_defaults_to_unified_slicing(monkeypatch):
    problem = _minimal_reflectivity_problem()
    captured = {}

    def capture(request):
        captured["slicing"] = request.slicing
        return type("Result", (), {"reflectivity": np.ones_like(request.angles)})()

    fitting_module = import_module(FittingProblem.__module__)
    monkeypatch.setattr(fitting_module, "simulate_reflectivity", capture)
    problem.evaluate({"unused": 0.5})

    assert isinstance(problem.slicing, LayerSlicingPolicy)
    assert captured["slicing"] is problem.slicing


def test_fitting_problem_explicit_none_preserves_legacy_slicing(monkeypatch):
    problem = _minimal_reflectivity_problem(slicing=None, roughness_step=0.5)
    captured = {}

    def capture(request):
        captured["slicing"] = request.slicing
        captured["roughness_step"] = request.roughness_step
        return type("Result", (), {"reflectivity": np.ones_like(request.angles)})()

    fitting_module = import_module(FittingProblem.__module__)
    monkeypatch.setattr(fitting_module, "simulate_reflectivity", capture)
    problem.evaluate({"unused": 0.5})

    assert captured == {"slicing": None, "roughness_step": 0.5}


@pytest.mark.parametrize(
    ("argument", "message"),
    [
        ({"field_step": 2.0}, "field_step is only used by the legacy path"),
        ({"roughness_step": 0.5}, "roughness_step is only used by the legacy path"),
    ],
)
def test_fitting_problem_unified_slicing_rejects_legacy_step_overrides(
    argument,
    message,
):
    with pytest.raises(ValueError, match=message):
        _minimal_reflectivity_problem(**argument)
