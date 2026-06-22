import numpy as np

from swxps import (
    CoreLevelRequest,
    Layer,
    LayerSlicingPolicy,
    ReflectivityRequest,
    RockingCurveRequest,
    SimulationStack,
    StackLayer,
    fixed_layer_grid_plan,
    simulate_reflectivity,
    simulate_rocking_curves,
)
from swxps.fields import transfer_matrix_reflectivity_array
from swxps.slicing import adaptive_layer_grid
from swxps.unified_grid import integrate_xps_on_grid


def make_rough_stack() -> SimulationStack:
    return SimulationStack(
        (
            StackLayer("vacuum", 0.0),
            StackLayer("surface", 4.0, delta=7.0e-6, beta=1.0e-7, roughness=1.0),
            StackLayer("film", 16.0, delta=2.5e-6, beta=8.0e-8, roughness=1.5),
            StackLayer("substrate", 0.0, delta=1.0e-5, beta=2.0e-7, roughness=2.0),
        )
    )


def test_unified_reflectivity_tracks_fine_legacy_roughness_reference():
    stack = make_rough_stack()
    angles = np.linspace(0.6, 5.0, 41)
    policy = LayerSlicingPolicy(min_slices=10, max_slice_thickness=2.0)

    unified = simulate_reflectivity(
        ReflectivityRequest(
            angles=angles,
            energy_ev=3000.0,
            stack=stack,
            slicing=policy,
        )
    ).reflectivity
    fine_reference = transfer_matrix_reflectivity_array(
        angles,
        3000.0,
        stack.optical_layers,
        roughness_step=0.1,
    )

    np.testing.assert_allclose(unified, fine_reference, rtol=2.0e-2, atol=2.0e-6)


def test_unified_uniform_slab_xps_matches_continuous_analytic_integral():
    thickness = 20.0
    attenuation_length = 20.0
    layers = [Layer(0.0), Layer(thickness), Layer(0.0)]
    grid = adaptive_layer_grid(layers)

    actual = integrate_xps_on_grid(
        np.ones((len(grid.centers), 1)),
        grid,
        np.ones(len(grid.centers)),
        np.full(len(grid.centers), attenuation_length),
    )[0]
    expected = attenuation_length * (1.0 - np.exp(-thickness / attenuation_length))

    np.testing.assert_allclose(actual, expected, rtol=5.0e-4)


def test_unified_no_contrast_stack_has_flat_normalized_rocking_curve():
    stack = SimulationStack(
        (
            StackLayer("vacuum", 0.0),
            StackLayer("film", 4.0),
            StackLayer("substrate", 0.0),
        )
    )
    angles = np.array([1.0, 2.0, 3.0])
    core = CoreLevelRequest(
        name="film",
        binding_energy_ev=100.0,
        concentration_by_material={"film": 1.0},
        imfp_by_material={"vacuum": 20.0, "film": 20.0, "substrate": 20.0},
    )

    curve = simulate_rocking_curves(
        RockingCurveRequest(
            angles=angles,
            photon_energy_ev=3000.0,
            stack=stack,
            core_levels=(core,),
            slicing=LayerSlicingPolicy(),
        )
    ).core_levels[0].curve

    np.testing.assert_allclose(curve.intensity, 1.0, rtol=1e-12, atol=1e-12)
    assert np.all(curve.raw_intensity > 0.0)


def test_fixed_plan_high_level_shape_stays_valid_across_trial_thicknesses():
    capacity = make_rough_stack()
    plan = fixed_layer_grid_plan(capacity.optical_layers)
    angles = np.linspace(1.0, 3.0, 7)

    results = []
    for surface_thickness in (2.5, 3.7, 4.0):
        trial = SimulationStack(
            (
                capacity.layers[0],
                StackLayer(
                    "surface",
                    surface_thickness,
                    delta=7.0e-6,
                    beta=1.0e-7,
                    roughness=1.0,
                ),
                capacity.layers[2],
                capacity.layers[3],
            )
        )
        results.append(
            simulate_reflectivity(
                ReflectivityRequest(
                    angles=angles,
                    energy_ev=3000.0,
                    stack=trial,
                    slicing=plan,
                )
            ).reflectivity
        )

    assert all(result.shape == angles.shape for result in results)
    assert not np.allclose(results[0], results[-1])
