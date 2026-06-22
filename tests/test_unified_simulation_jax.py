import numpy as np
import pytest

simulation_jax = pytest.importorskip("swxps.simulation_jax", exc_type=ImportError)

from swxps import (
    CoreLevelRequest,
    LayerSlicingPolicy,
    ReflectivityRequest,
    RockingCurveRequest,
    SimulationStack,
    StackLayer,
    fixed_layer_grid_plan,
    simulate_reflectivity,
    simulate_rocking_curves,
)


def make_stack() -> SimulationStack:
    return SimulationStack(
        (
            StackLayer("vacuum", 0.0),
            StackLayer("A", 4.0, delta=7.0e-6, beta=1.0e-7, roughness=1.0),
            StackLayer("B", 16.0, delta=2.5e-6, beta=8.0e-8, roughness=1.5),
            StackLayer("B", 0.0, delta=1.0e-5, beta=2.0e-7, roughness=2.0),
        )
    )


def test_unified_jax_reflectivity_matches_numpy():
    stack = make_stack()
    plan = fixed_layer_grid_plan(stack.optical_layers, LayerSlicingPolicy())
    request = ReflectivityRequest(
        angles=np.linspace(0.8, 4.0, 16),
        energy_ev=3000.0,
        stack=stack,
        slicing=plan,
    )

    expected = simulate_reflectivity(request)
    actual = simulation_jax.simulate_reflectivity_jax(request)

    np.testing.assert_allclose(actual.reflectivity, expected.reflectivity, rtol=1e-11, atol=1e-12)


def test_unified_jax_rocking_curve_matches_numpy():
    stack = make_stack()
    plan = fixed_layer_grid_plan(stack.optical_layers)
    core = CoreLevelRequest(
        name="A core",
        binding_energy_ev=100.0,
        concentration_by_material={"A": 1.0},
        imfp_by_material={"vacuum": 20.0, "A": 20.0, "B": 30.0},
    )
    request = RockingCurveRequest(
        angles=np.linspace(1.0, 3.0, 9),
        photon_energy_ev=3000.0,
        stack=stack,
        core_levels=(core,),
        slicing=plan,
    )

    expected = simulate_rocking_curves(request).core_levels[0].curve
    actual = simulation_jax.simulate_rocking_curves_jax(request).core_levels[0].curve

    np.testing.assert_allclose(actual.raw_intensity, expected.raw_intensity, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(actual.intensity, expected.intensity, rtol=1e-11, atol=1e-12)
