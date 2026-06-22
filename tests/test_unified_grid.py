import numpy as np

from swxps import Layer, parratt_reflectivity
from swxps.slicing import LayerSlicingPolicy, adaptive_layer_grid
from swxps.unified_grid import (
    cell_centered_attenuation,
    effective_layers_from_grid,
    field_profiles_on_grid,
    integrate_xps_on_grid,
    reflectivity_on_grid,
)


def make_sharp_layers():
    return [
        Layer(0.0),
        Layer(4.0, delta=5.0e-6, beta=1.0e-7),
        Layer(16.0, delta=2.5e-6, beta=8.0e-8),
        Layer(0.0, delta=1.0e-5, beta=2.0e-7),
    ]


def test_subdividing_sharp_layers_preserves_reflectivity():
    layers = make_sharp_layers()
    angles = np.linspace(0.5, 5.0, 31)
    grid = adaptive_layer_grid(layers)
    effective = effective_layers_from_grid(layers, grid)

    actual = reflectivity_on_grid(angles, 3000.0, effective)
    expected = parratt_reflectivity(angles, 3000.0, layers)

    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-13)


def test_field_profiles_use_exact_grid_centers_and_one_sample_per_cell():
    layers = make_sharp_layers()
    grid = adaptive_layer_grid(layers)
    effective = effective_layers_from_grid(layers, grid)

    profile = field_profiles_on_grid(
        np.array([2.0]),
        3000.0,
        effective,
        grid,
    )[0]

    assert len(profile.depth) == sum(grid.slice_counts)
    np.testing.assert_allclose(profile.depth, grid.centers)
    np.testing.assert_array_equal(profile.layer_index, grid.effective_layer_index)


def test_cell_centered_attenuation_matches_uniform_slab_analytic_values():
    widths = np.full(4, 2.0)
    attenuation = cell_centered_attenuation(widths, np.full(4, 20.0))
    centers = np.array([1.0, 3.0, 5.0, 7.0])

    np.testing.assert_allclose(attenuation, np.exp(-centers / 20.0))


def test_cell_centered_xps_integral_matches_uniform_slab_midpoint_formula():
    layers = [Layer(0.0), Layer(20.0), Layer(0.0)]
    grid = adaptive_layer_grid(
        layers,
        LayerSlicingPolicy(min_slices=10, max_slice_thickness=2.0),
    )
    actual = integrate_xps_on_grid(
        np.ones((10, 1)),
        grid,
        np.ones(10),
        np.full(10, 20.0),
    )[0]
    expected = np.sum(np.exp(-grid.centers / 20.0) * grid.widths)

    np.testing.assert_allclose(actual, expected)
