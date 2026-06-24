"""Stage 6 thin simulation shim and exact compatibility parity."""

import ast
from pathlib import Path

import numpy as np


def test_simulation_module_is_only_a_thin_compatibility_layer():
    import swanx.simulation as simulation

    source = Path(simulation.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    assert definitions == []
    assert len(source.splitlines()) < 50
    assert not hasattr(simulation, "_values_by_material")
    assert not hasattr(simulation, "_apply_emitting_layer_filter")


def test_xps_helpers_have_one_canonical_location():
    import swanx.simulation_jax as jax_workflow
    import swanx.simulation_unified as unified
    import swanx.workflows.simulate as workflow
    from swanx.xps.utils import (
        _apply_emitting_layer_filter,
        _values_by_material,
    )

    assert _values_by_material.__module__ == "swanx.xps.utils"
    assert _apply_emitting_layer_filter.__module__ == "swanx.xps.utils"
    assert workflow._values_by_material is _values_by_material
    assert workflow._apply_emitting_layer_filter is _apply_emitting_layer_filter
    assert jax_workflow._values_by_material is _values_by_material
    assert jax_workflow._apply_emitting_layer_filter is _apply_emitting_layer_filter
    assert unified._values_by_material is _values_by_material
    assert unified._apply_emitting_layer_filter is _apply_emitting_layer_filter


def _stack():
    from swanx.stack.model import SimulationStack, StackLayer

    return SimulationStack(
        (
            StackLayer("vacuum", 0.0),
            StackLayer("film", 10.0, delta=5.0e-6, beta=1.0e-7),
            StackLayer("substrate", 0.0, delta=1.0e-5, beta=2.0e-7),
        )
    )


def test_reflectivity_compatibility_and_canonical_results_match_exactly():
    from swanx.simulation import simulate_reflectivity as compatibility_simulate
    from swanx.workflows.simulate import ReflectivityRequest
    from swanx.workflows.simulate import simulate_reflectivity as canonical_simulate
    from swxps.simulation import simulate_reflectivity as legacy_simulate

    request = ReflectivityRequest(
        angles=np.array([0.5, 1.0, 1.5]),
        energy_ev=8000.0,
        stack=_stack(),
        slicing=None,
    )
    compatibility = compatibility_simulate(request)
    canonical = canonical_simulate(request)
    legacy = legacy_simulate(request)

    assert compatibility_simulate is canonical_simulate is legacy_simulate
    np.testing.assert_array_equal(compatibility.angle, canonical.angle)
    np.testing.assert_array_equal(
        compatibility.calculation_angle,
        canonical.calculation_angle,
    )
    np.testing.assert_array_equal(compatibility.reflectivity, canonical.reflectivity)
    np.testing.assert_array_equal(legacy.reflectivity, canonical.reflectivity)


def test_swxps_compatibility_and_canonical_results_match_exactly():
    from swanx.simulation import simulate_rocking_curves as compatibility_simulate
    from swanx.workflows.simulate import CoreLevelRequest, RockingCurveRequest
    from swanx.workflows.simulate import simulate_rocking_curves as canonical_simulate
    from swxps.simulation import simulate_rocking_curves as legacy_simulate

    core = CoreLevelRequest(
        name="film level",
        binding_energy_ev=100.0,
        concentration_by_material={
            "vacuum": 0.0,
            "film": 1.0,
            "substrate": 0.0,
        },
        imfp_by_material={
            "vacuum": 20.0,
            "film": 20.0,
            "substrate": 20.0,
        },
    )
    request = RockingCurveRequest(
        angles=np.array([0.8, 1.0, 1.2]),
        photon_energy_ev=1000.0,
        stack=_stack(),
        core_levels=(core,),
        slicing=None,
    )
    compatibility = compatibility_simulate(request)
    canonical = canonical_simulate(request)
    legacy = legacy_simulate(request)

    assert compatibility_simulate is canonical_simulate is legacy_simulate
    np.testing.assert_array_equal(compatibility.angle, canonical.angle)
    np.testing.assert_array_equal(
        compatibility.calculation_angle,
        canonical.calculation_angle,
    )
    np.testing.assert_array_equal(
        compatibility.core_levels[0].curve.raw_intensity,
        canonical.core_levels[0].curve.raw_intensity,
    )
    np.testing.assert_array_equal(
        compatibility.core_levels[0].curve.intensity,
        canonical.core_levels[0].curve.intensity,
    )
    np.testing.assert_array_equal(
        legacy.core_levels[0].curve.intensity,
        canonical.core_levels[0].curve.intensity,
    )
