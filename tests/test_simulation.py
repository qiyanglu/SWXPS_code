import numpy as np

from swxps import (
    CoreLevelRequest,
    ReflectivityRequest,
    RockingCurveRequest,
    SimulationStack,
    StackLayer,
    simulate_reflectivity,
    simulate_rocking_curves,
)


def make_test_stack() -> SimulationStack:
    return SimulationStack(
        (
            StackLayer("vacuum", thickness=0.0),
            StackLayer("A", thickness=20.0, delta=5.0e-6, beta=1.0e-7),
            StackLayer("B", thickness=20.0, delta=2.0e-6, beta=1.0e-7),
            StackLayer("B", thickness=0.0, delta=2.0e-6, beta=1.0e-7),
        )
    )


def test_simulate_reflectivity_returns_angle_offset():
    angles = np.array([1.0, 2.0, 3.0])
    request = ReflectivityRequest(
        angles=angles,
        energy_ev=3000.0,
        stack=make_test_stack(),
        angle_offset=0.1,
    )

    result = simulate_reflectivity(request)

    np.testing.assert_allclose(result.angle, angles)
    np.testing.assert_allclose(result.calculation_angle, angles + 0.1)
    assert result.reflectivity.shape == angles.shape
    assert np.all(result.reflectivity >= 0.0)


def test_simulate_multiple_rocking_curves_returns_named_results():
    angles = np.array([1.0, 2.0, 3.0])
    core_a = CoreLevelRequest(
        name="A core",
        binding_energy_ev=100.0,
        concentration_by_material={"A": 1.0},
        imfp_by_material={"vacuum": 20.0, "A": 20.0, "B": 20.0},
    )
    core_b = CoreLevelRequest(
        name="B core",
        binding_energy_ev=200.0,
        concentration_by_material={"B": 1.0},
        imfp_by_material={"vacuum": 20.0, "A": 20.0, "B": 20.0},
    )
    request = RockingCurveRequest(
        angles=angles,
        photon_energy_ev=3000.0,
        stack=make_test_stack(),
        core_levels=(core_a, core_b),
        angle_offset=-0.05,
        field_step=2.0,
    )

    result = simulate_rocking_curves(request)

    np.testing.assert_allclose(result.angle, angles)
    np.testing.assert_allclose(result.calculation_angle, angles - 0.05)
    assert [core.name for core in result.core_levels] == ["A core", "B core"]
    for core in result.core_levels:
        assert core.curve.intensity.shape == angles.shape
        assert np.all(np.isfinite(core.curve.intensity))


def test_simulate_rocking_curve_requires_imfp_for_materials():
    angles = np.array([1.0, 2.0])
    core = CoreLevelRequest(
        name="A core",
        binding_energy_ev=100.0,
        concentration_by_material={"A": 1.0},
        imfp_by_material={"A": 20.0},
    )
    request = RockingCurveRequest(
        angles=angles,
        photon_energy_ev=3000.0,
        stack=make_test_stack(),
        core_levels=(core,),
    )

    try:
        simulate_rocking_curves(request)
    except ValueError as error:
        assert "missing value for material" in str(error)
    else:
        raise AssertionError("missing IMFP values should raise ValueError")
