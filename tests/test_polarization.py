import numpy as np
import pytest

import swanx.simulation_jax as simulation_jax
from swanx.stack import SimulationStack, StackLayer
from swanx.workflows import (
    CoreLevelRequest,
    ReflectivityRequest,
    RockingCurveRequest,
    simulate_reflectivity,
    simulate_rocking_curves,
)


def _stack() -> SimulationStack:
    return SimulationStack(
        (
            StackLayer("vacuum", thickness=0.0, delta=0.0, beta=0.0),
            StackLayer("film", thickness=24.0, delta=5.0e-5, beta=2.0e-6),
            StackLayer("spacer", thickness=18.0, delta=2.0e-5, beta=1.0e-6),
            StackLayer("substrate", thickness=0.0, delta=8.0e-5, beta=3.0e-6),
        )
    )


def _core() -> CoreLevelRequest:
    return CoreLevelRequest(
        name="film core",
        binding_energy_ev=100.0,
        concentration_by_material={"film": 1.0, "spacer": 0.25},
        imfp_by_material={
            "vacuum": 20.0,
            "film": 20.0,
            "spacer": 25.0,
            "substrate": 30.0,
        },
    )


def test_default_reflectivity_matches_explicit_s_for_legacy_and_unified():
    angles = np.linspace(0.6, 3.0, 9)
    for slicing in (None,):
        default = simulate_reflectivity(
            ReflectivityRequest(angles, 3000.0, _stack(), slicing=slicing)
        ).reflectivity
        explicit_s = simulate_reflectivity(
            ReflectivityRequest(
                angles,
                3000.0,
                _stack(),
                polarization="s",
                slicing=slicing,
            )
        ).reflectivity
        np.testing.assert_allclose(default, explicit_s, rtol=0.0, atol=0.0)

    default_unified = simulate_reflectivity(
        ReflectivityRequest(angles, 3000.0, _stack())
    ).reflectivity
    explicit_unified = simulate_reflectivity(
        ReflectivityRequest(angles, 3000.0, _stack(), polarization="s")
    ).reflectivity
    np.testing.assert_allclose(default_unified, explicit_unified, rtol=0.0, atol=0.0)


def test_p_reflectivity_differs_from_s_for_multilayer():
    angles = np.linspace(0.6, 8.0, 50)
    s_reflectivity = simulate_reflectivity(
        ReflectivityRequest(angles, 3000.0, _stack(), polarization="s")
    ).reflectivity
    p_reflectivity = simulate_reflectivity(
        ReflectivityRequest(angles, 3000.0, _stack(), polarization="p")
    ).reflectivity

    assert np.max(np.abs(s_reflectivity - p_reflectivity)) > 1.0e-10


def test_mixed_reflectivity_is_raw_weighted_sum_before_normalization():
    angles = np.linspace(0.6, 4.0, 17)
    s_reflectivity = simulate_reflectivity(
        ReflectivityRequest(angles, 3000.0, _stack(), polarization="s")
    ).reflectivity
    p_reflectivity = simulate_reflectivity(
        ReflectivityRequest(angles, 3000.0, _stack(), polarization="p")
    ).reflectivity
    mixed = simulate_reflectivity(
        ReflectivityRequest(
            angles,
            3000.0,
            _stack(),
            polarization={"s": 0.7, "p": 0.3},
        )
    ).reflectivity

    np.testing.assert_allclose(mixed, 0.7 * s_reflectivity + 0.3 * p_reflectivity)


def test_mixed_rocking_curve_raw_intensity_is_weighted_before_normalization():
    angles = np.linspace(0.8, 3.0, 13)
    common = dict(
        angles=angles,
        photon_energy_ev=3000.0,
        stack=_stack(),
        core_levels=(_core(),),
    )
    s_curve = simulate_rocking_curves(
        RockingCurveRequest(**common, polarization="s")
    ).core_levels[0].curve
    p_curve = simulate_rocking_curves(
        RockingCurveRequest(**common, polarization="p")
    ).core_levels[0].curve
    mixed_curve = simulate_rocking_curves(
        RockingCurveRequest(**common, polarization={"s": 0.4, "p": 0.6})
    ).core_levels[0].curve

    np.testing.assert_allclose(
        mixed_curve.raw_intensity,
        0.4 * s_curve.raw_intensity + 0.6 * p_curve.raw_intensity,
    )


def test_reflectivity_and_rocking_curve_accept_all_polarization_modes():
    angles = np.linspace(0.8, 2.0, 5)
    for polarization in ("s", "p", {"s": 0.5, "p": 0.5}):
        reflectivity = simulate_reflectivity(
            ReflectivityRequest(angles, 3000.0, _stack(), polarization=polarization)
        )
        assert reflectivity.reflectivity.shape == angles.shape

        rocking_curves = simulate_rocking_curves(
            RockingCurveRequest(
                angles,
                3000.0,
                _stack(),
                (_core(),),
                polarization=polarization,
            )
        )
        assert rocking_curves.core_levels[0].curve.intensity.shape == angles.shape


def test_jax_reflectivity_accepts_polarization_modes_and_mixed_sum():
    pytest.importorskip("jax")
    angles = np.linspace(0.8, 2.0, 5)
    s_result = simulation_jax.simulate_reflectivity_jax(
        ReflectivityRequest(angles, 3000.0, _stack(), polarization="s")
    ).reflectivity
    p_result = simulation_jax.simulate_reflectivity_jax(
        ReflectivityRequest(angles, 3000.0, _stack(), polarization="p")
    ).reflectivity
    mixed = simulation_jax.simulate_reflectivity_jax(
        ReflectivityRequest(
            angles,
            3000.0,
            _stack(),
            polarization={"s": 0.25, "p": 0.75},
        )
    ).reflectivity

    np.testing.assert_allclose(mixed, 0.25 * s_result + 0.75 * p_result)


def test_jax_rocking_curve_mixed_raw_intensity_sum():
    pytest.importorskip("jax")
    angles = np.linspace(0.8, 2.0, 5)
    common = dict(
        angles=angles,
        photon_energy_ev=3000.0,
        stack=_stack(),
        core_levels=(_core(),),
    )
    s_curve = simulation_jax.simulate_rocking_curves_jax(
        RockingCurveRequest(**common, polarization="s")
    ).core_levels[0].curve
    p_curve = simulation_jax.simulate_rocking_curves_jax(
        RockingCurveRequest(**common, polarization="p")
    ).core_levels[0].curve
    mixed_curve = simulation_jax.simulate_rocking_curves_jax(
        RockingCurveRequest(**common, polarization={"s": 0.25, "p": 0.75})
    ).core_levels[0].curve

    np.testing.assert_allclose(
        mixed_curve.raw_intensity,
        0.25 * s_curve.raw_intensity + 0.75 * p_curve.raw_intensity,
    )
