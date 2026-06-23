import numpy as np

from swxps import Layer, LayerSlicingPolicy
from swxps.fields import effective_layers_with_roughness
from swxps.slicing import adaptive_layer_grid
from swxps.unified_grid import effective_layers_from_grid


def test_unified_optical_cells_match_legacy_grading_on_identical_grid():
    layers = [
        Layer(0.0),
        Layer(20.0, delta=7.0e-6, beta=1.0e-7, roughness=3.0),
        Layer(20.0, delta=2.5e-6, beta=8.0e-8, roughness=3.0),
        Layer(0.0, delta=1.0e-5, beta=2.0e-7, roughness=3.0),
    ]
    grid = adaptive_layer_grid(
        layers,
        LayerSlicingPolicy(min_slices=1, max_slice_thickness=1.0),
    )

    unified = effective_layers_from_grid(layers, grid)
    legacy = effective_layers_with_roughness(layers, step=1.0)

    np.testing.assert_allclose(
        [layer.thickness for layer in unified],
        [layer.thickness for layer in legacy],
    )
    np.testing.assert_allclose(
        [layer.delta for layer in unified],
        [layer.delta for layer in legacy],
    )
    np.testing.assert_allclose(
        [layer.beta for layer in unified],
        [layer.beta for layer in legacy],
    )
