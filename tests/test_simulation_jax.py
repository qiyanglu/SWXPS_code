import numpy as np
import pytest

simulation_jax = pytest.importorskip(
    "swxps.simulation_jax",
    exc_type=ImportError,
)

from swxps import (
    CoreLevelRequest,
    ReflectivityRequest,
    RockingCurveRequest,
    SimulationStack,
    StackLayer,
    simulate_reflectivity,
    simulate_rocking_curves,
)


def make_rough_stack() -> SimulationStack:
    return SimulationStack(
        (
            StackLayer("vacuum", thickness=0.0),
            StackLayer("A", thickness=20.0, delta=5.0e-6, beta=1.0e-7, roughness=1.0),
            StackLayer("B", thickness=30.0, delta=2.5e-6, beta=8.0e-8, roughness=1.5),
            StackLayer("B", thickness=0.0, delta=1.0e-5, beta=2.0e-7, roughness=2.0),
        )
    )


def test_jax_simulate_reflectivity_matches_numpy_transfer_matrix():
    request = ReflectivityRequest(
        angles=np.linspace(0.8, 4.0, 16),
        energy_ev=3000.0,
        stack=make_rough_stack(),
        angle_offset=0.05,
        roughness_step=1.0,
    )

    actual = simulation_jax.simulate_reflectivity_jax(request)
    expected = simulate_reflectivity(request)

    np.testing.assert_allclose(actual.angle, expected.angle)
    np.testing.assert_allclose(actual.calculation_angle, expected.calculation_angle)
    np.testing.assert_allclose(
        actual.reflectivity,
        expected.reflectivity,
        rtol=1e-11,
        atol=1e-12,
    )


def test_jax_simulate_rocking_curves_matches_numpy_transfer_matrix():
    angles = np.array([1.0, 1.5, 2.0, 2.5])
    core_a = CoreLevelRequest(
        name="A core",
        binding_energy_ev=100.0,
        concentration_by_material={"A": 1.0},
        imfp_by_material={"vacuum": 20.0, "A": 20.0, "B": 30.0},
    )
    core_b = CoreLevelRequest(
        name="B top",
        binding_energy_ev=200.0,
        concentration_by_material={"B": 1.0},
        imfp_by_material={"vacuum": 20.0, "A": 20.0, "B": 30.0},
        emitting_layer_indices=(2,),
    )
    request = RockingCurveRequest(
        angles=angles,
        photon_energy_ev=3000.0,
        stack=make_rough_stack(),
        core_levels=(core_a, core_b),
        angle_offset=-0.03,
        field_step=2.0,
        roughness_step=1.0,
    )

    actual = simulation_jax.simulate_rocking_curves_jax(request)
    expected = simulate_rocking_curves(request)

    np.testing.assert_allclose(actual.angle, expected.angle)
    np.testing.assert_allclose(actual.calculation_angle, expected.calculation_angle)
    assert [core.name for core in actual.core_levels] == [
        core.name for core in expected.core_levels
    ]
    for actual_core, expected_core in zip(actual.core_levels, expected.core_levels):
        np.testing.assert_allclose(
            actual_core.curve.intensity,
            expected_core.curve.intensity,
            rtol=1e-11,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            actual_core.curve.raw_intensity,
            expected_core.curve.raw_intensity,
            rtol=1e-11,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            actual_core.curve.normalization,
            expected_core.curve.normalization,
            rtol=1e-11,
            atol=1e-12,
        )


def test_jax_edge_polynomial_rocking_curve_normalization_matches_numpy():
    angles = np.linspace(1.0, 3.5, 21)
    core = CoreLevelRequest(
        name="A core",
        binding_energy_ev=100.0,
        concentration_by_material={"A": 1.0},
        imfp_by_material={"vacuum": 20.0, "A": 20.0, "B": 30.0},
    )
    request = RockingCurveRequest(
        angles=angles,
        photon_energy_ev=3000.0,
        stack=make_rough_stack(),
        core_levels=(core,),
        field_step=2.0,
        roughness_step=1.0,
        normalization_mode="edge_polynomial",
        normalization_edge_fraction=0.10,
        normalization_polynomial_order=2,
    )

    actual = simulation_jax.simulate_rocking_curves_jax(request).core_levels[0].curve
    expected = simulate_rocking_curves(request).core_levels[0].curve

    np.testing.assert_allclose(actual.raw_intensity, expected.raw_intensity, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(actual.normalization, expected.normalization, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(actual.intensity, expected.intensity, rtol=1e-11, atol=1e-12)