import numpy as np

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
