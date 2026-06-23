import numpy as np
import pytest

from swxps import (
    CoreLevelRequest,
    LayerSlicingPolicy,
    ReflectivityRequest,
    RockingCurveRequest,
    SimulationStack,
    StackLayer,
    simulate_reflectivity,
    simulate_rocking_curves,
)
from swxps.fields import transfer_matrix_reflectivity_array


def make_stack() -> SimulationStack:
    return SimulationStack(
        (
            StackLayer("vacuum", 0.0),
            StackLayer("film", 12.0, delta=5.0e-6, beta=1.0e-7, roughness=1.0),
            StackLayer("substrate", 0.0, delta=2.0e-6, beta=2.0e-7, roughness=1.5),
        )
    )


def make_core() -> CoreLevelRequest:
    return CoreLevelRequest(
        name="film",
        binding_energy_ev=100.0,
        concentration_by_material={"film": 1.0},
        imfp_by_material={"vacuum": 20.0, "film": 20.0, "substrate": 25.0},
    )


def test_reflectivity_request_defaults_to_its_own_unified_policy():
    first = ReflectivityRequest(np.array([1.0]), 3000.0, make_stack())
    second = ReflectivityRequest(np.array([1.0]), 3000.0, make_stack())

    assert isinstance(first.slicing, LayerSlicingPolicy)
    assert isinstance(second.slicing, LayerSlicingPolicy)
    assert first.slicing is not second.slicing


def test_rocking_curve_request_defaults_to_unified_policy():
    request = RockingCurveRequest(
        np.array([1.0]), 3000.0, make_stack(), (make_core(),)
    )

    assert isinstance(request.slicing, LayerSlicingPolicy)


def test_explicit_none_uses_legacy_reflectivity_roughness_step():
    angles = np.linspace(0.8, 4.0, 9)
    request = ReflectivityRequest(
        angles,
        3000.0,
        make_stack(),
        roughness_step=0.5,
        slicing=None,
    )

    actual = simulate_reflectivity(request).reflectivity
    expected = transfer_matrix_reflectivity_array(
        angles,
        3000.0,
        request.stack.optical_layers,
        roughness_step=0.5,
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-14)


def test_explicit_none_retains_legacy_rocking_curve_steps():
    request = RockingCurveRequest(
        np.linspace(1.0, 3.0, 5),
        3000.0,
        make_stack(),
        (make_core(),),
        field_step=2.0,
        roughness_step=0.5,
        slicing=None,
    )

    result = simulate_rocking_curves(request).core_levels[0].curve

    assert np.all(np.isfinite(result.intensity))
    assert np.all(np.isfinite(result.raw_intensity))


@pytest.mark.parametrize("step", [0.5, (0.5, 1.0)])
def test_unified_reflectivity_rejects_nondefault_roughness_step(step):
    with pytest.raises(ValueError, match="roughness_step is only used by the legacy path"):
        ReflectivityRequest(
            np.array([1.0]),
            3000.0,
            make_stack(),
            roughness_step=step,
        )


def test_unified_rocking_curve_rejects_nondefault_field_step():
    with pytest.raises(ValueError, match="field_step is only used by the legacy path"):
        RockingCurveRequest(
            np.array([1.0]),
            3000.0,
            make_stack(),
            (make_core(),),
            field_step=2.0,
        )


def test_unified_rocking_curve_rejects_nondefault_roughness_step():
    with pytest.raises(ValueError, match="roughness_step is only used by the legacy path"):
        RockingCurveRequest(
            np.array([1.0]),
            3000.0,
            make_stack(),
            (make_core(),),
            roughness_step=0.5,
        )
