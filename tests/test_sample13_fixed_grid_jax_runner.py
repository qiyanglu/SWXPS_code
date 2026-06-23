from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("jax")

from swxps import (
    LayerSlicingPolicy,
    ReflectivityRequest,
    RockingCurveRequest,
    fixed_layer_grid,
    fixed_layer_grid_plan,
    simulate_reflectivity,
    simulate_rocking_curves,
)


RUNNER_DIR = (
    Path(__file__).resolve().parents[1]
    / "case_studies"
    / "sample_13"
    / "jax_least_squares_fixed_grid"
)
if str(RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNER_DIR))

import fit_sample13_fixed_grid_jax_least_squares as runner  # noqa: E402


def _setup():
    runner.legacy_trf.fit13.patch_archived_context_paths()
    values = runner.legacy_trf.load_start_values(runner.START_SUMMARY)
    vector = runner.legacy_trf.interior_vector(values)
    values = runner.values_from_vector(vector)
    stack = runner.legacy_trf.fit13.build_cap3_stack(values)
    plan = fixed_layer_grid_plan(
        runner.capacity_layers(stack),
        LayerSlicingPolicy(min_slices=10, max_slice_thickness=2.0),
    )
    return values, vector, stack, plan


def test_sample13_capacity_plan_preserves_shape_across_coupled_cap_extremes():
    values, _, stack, plan = _setup()
    expected_cells = sum(plan.slice_counts)
    assert len(plan.slice_counts) == len(stack.optical_layers) - 2
    assert expected_cells == 1092

    for total, layer1 in ((65.0, 1.0), (65.0, 20.0)):
        trial_values = {
            **values,
            "top_lno_total_thickness": total,
            "top_lno_layer1_thickness": layer1,
            "sto_thickness_start": 19.0,
            "lno_thickness_start": 19.0,
            "sto_thickness_delta": 3.0,
            "lno_thickness_delta": 3.0,
        }
        trial = runner.legacy_trf.fit13.build_cap3_stack(trial_values)
        assert len(fixed_layer_grid(trial.optical_layers, plan).centers) == expected_cells


def test_sample13_jax_curves_match_numpy_unified_grid():
    values, vector, stack, plan = _setup()
    angles = np.array([12.6, 13.4])
    core_levels = (
        *runner.legacy_trf.fit13.core_level_requests(),
        runner.legacy_trf.la_core_level_request(),
    )
    model = runner.Sample13FixedGridJaxModel(
        angles, angles, core_levels, plan, stack
    )
    reflectivity, curves = model.simulate_curves(vector)

    numpy_reflectivity = simulate_reflectivity(
        ReflectivityRequest(
            angles=angles,
            energy_ev=runner.legacy_trf.fit13.sample13.PHOTON_ENERGY_EV,
            stack=stack,
            angle_offset=values["reflectivity_angle_offset"],
            slicing=plan,
        )
    )
    numpy_curves = simulate_rocking_curves(
        RockingCurveRequest(
            angles=angles,
            photon_energy_ev=runner.legacy_trf.fit13.sample13.PHOTON_ENERGY_EV,
            stack=stack,
            core_levels=core_levels,
            angle_offset=values["rc_angle_offset"],
            offpeak_mask=np.ones_like(angles, dtype=bool),
            slicing=plan,
        )
    )

    np.testing.assert_allclose(
        reflectivity, numpy_reflectivity.reflectivity, rtol=1.0e-11, atol=1.0e-13
    )
    for jax_curve, numpy_curve in zip(curves, numpy_curves.core_levels):
        np.testing.assert_allclose(
            jax_curve, numpy_curve.curve.intensity, rtol=1.0e-11, atol=1.0e-13
        )
