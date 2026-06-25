import numpy as np
import pytest

import swanx.simulation_jax as simulation_jax
from swanx.fitting import FittingProblem, ReflectivityData, RockingCurveData
from swanx.optics import (
    energy_to_wavelength,
    kz_in_layers,
    transfer_matrix_reflectivity_array,
)
from swanx.stack import (
    Layer,
    LayerTemplate,
    SimulationStack,
    StackLayer,
    StackTemplate,
    SuperlatticeTemplate,
)
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


def _lno_sto_superlattice_layers(
    *,
    energy_ev: float,
    repeats: int = 20,
    lno_thickness: float = 20.0,
    sto_thickness: float = 20.0,
) -> tuple:
    template = StackTemplate(
        energy_ev=energy_ev,
        parts=(
            LayerTemplate.vacuum(),
            SuperlatticeTemplate(
                repeats=repeats,
                period=(
                    LayerTemplate.from_file(
                        "LNO",
                        "data/OPC/LaNiO3.dat",
                        lno_thickness,
                        3.0,
                    ),
                    LayerTemplate.from_file(
                        "STO",
                        "data/OPC/SrTiO3.dat",
                        sto_thickness,
                        3.0,
                    ),
                ),
            ),
            LayerTemplate.from_file("STO", "data/OPC/SrTiO3.dat", 0.0, 3.0),
        ),
    )
    return template.build().optical_layers


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


def test_lno_sto_superlattice_reflectivity_compares_s_p_and_mixed_polarization():
    energy_ev = 3000.0
    period = 40.0
    bragg_angle = np.rad2deg(np.arcsin(energy_to_wavelength(energy_ev) / (2.0 * period)))
    angles = np.linspace(bragg_angle - 1.0, bragg_angle + 1.0, 501)
    layers = _lno_sto_superlattice_layers(energy_ev=energy_ev)

    s_reflectivity = transfer_matrix_reflectivity_array(
        angles,
        energy_ev,
        layers,
        roughness_step=1.0,
        polarization="s",
    )
    p_reflectivity = transfer_matrix_reflectivity_array(
        angles,
        energy_ev,
        layers,
        roughness_step=1.0,
        polarization="p",
    )
    mixed_reflectivity = transfer_matrix_reflectivity_array(
        angles,
        energy_ev,
        layers,
        roughness_step=1.0,
        polarization={"s": 0.5, "p": 0.5},
    )

    peak_angle = angles[np.argmax(s_reflectivity)]
    assert abs(peak_angle - bragg_angle) < 0.25
    assert np.max(s_reflectivity) > 1.0e-2
    assert np.max(np.abs(s_reflectivity - p_reflectivity)) > 1.0e-5
    np.testing.assert_allclose(
        mixed_reflectivity,
        0.5 * s_reflectivity + 0.5 * p_reflectivity,
    )


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



def _polarized_fitting_problem(polarization):
    angles = np.linspace(0.8, 3.0, 9)
    return FittingProblem(
        parameters=(),
        stack_builder=lambda values: _stack(),
        photon_energy_ev=3000.0,
        reflectivity=ReflectivityData(
            "R",
            angles,
            np.full(angles.shape, 1.0e-4),
        ),
        rocking_curves=(
            RockingCurveData(
                "film core",
                angles,
                np.ones(angles.shape),
            ),
        ),
        core_levels=(_core(),),
        polarization=polarization,
    )


def test_fitting_problem_p_polarization_changes_reflectivity_and_rocking_curve():
    s_simulation = _polarized_fitting_problem("s").simulate({})
    p_simulation = _polarized_fitting_problem("p").simulate({})

    assert s_simulation.reflectivity is not None
    assert p_simulation.reflectivity is not None
    assert s_simulation.rocking_curves is not None
    assert p_simulation.rocking_curves is not None
    assert np.max(
        np.abs(
            s_simulation.reflectivity.reflectivity
            - p_simulation.reflectivity.reflectivity
        )
    ) > 1.0e-10
    s_raw = s_simulation.rocking_curves.core_levels[0].curve.raw_intensity
    p_raw = p_simulation.rocking_curves.core_levels[0].curve.raw_intensity
    assert np.max(np.abs(s_raw - p_raw)) > 1.0e-10


def test_fitting_problem_mixed_polarization_uses_raw_weighted_results():
    s_simulation = _polarized_fitting_problem("s").simulate({})
    p_simulation = _polarized_fitting_problem("p").simulate({})
    mixed_simulation = _polarized_fitting_problem({"s": 0.3, "p": 0.7}).simulate({})

    assert s_simulation.reflectivity is not None
    assert p_simulation.reflectivity is not None
    assert mixed_simulation.reflectivity is not None
    np.testing.assert_allclose(
        mixed_simulation.reflectivity.reflectivity,
        0.3 * s_simulation.reflectivity.reflectivity
        + 0.7 * p_simulation.reflectivity.reflectivity,
    )

    assert s_simulation.rocking_curves is not None
    assert p_simulation.rocking_curves is not None
    assert mixed_simulation.rocking_curves is not None
    s_raw = s_simulation.rocking_curves.core_levels[0].curve.raw_intensity
    p_raw = p_simulation.rocking_curves.core_levels[0].curve.raw_intensity
    mixed_raw = mixed_simulation.rocking_curves.core_levels[0].curve.raw_intensity
    np.testing.assert_allclose(mixed_raw, 0.3 * s_raw + 0.7 * p_raw)


def test_invalid_mixed_polarization_weights_must_sum_to_one():
    angles = np.linspace(0.8, 2.0, 5)
    with pytest.raises(ValueError, match="sum to 1"):
        ReflectivityRequest(angles, 3000.0, _stack(), polarization={"s": 70, "p": 30})
    with pytest.raises(ValueError, match="sum to 1"):
        _polarized_fitting_problem({"s": 0.2, "p": 0.2})


def test_single_interface_p_reflectivity_matches_admittance_formula():
    energy_ev = 3000.0
    angles = np.linspace(0.8, 4.0, 21)
    layers = [
        Layer(0.0),
        Layer(0.0, delta=5.0e-5, beta=2.0e-6),
    ]
    actual = transfer_matrix_reflectivity_array(
        angles,
        energy_ev,
        layers,
        polarization="p",
    )

    n = np.asarray([layer.n for layer in layers], dtype=complex)
    kz = kz_in_layers(angles, energy_to_wavelength(energy_ev), n)
    y_p = kz / (n[:, np.newaxis] ** 2)
    expected = np.abs((y_p[0] - y_p[1]) / (y_p[0] + y_p[1])) ** 2

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-14)
