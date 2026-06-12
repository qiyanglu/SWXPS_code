import numpy as np

from swxps import (
    FieldProfile,
    Layer,
    attenuation_factor,
    graded_layer_property_at_depth,
    nominal_layer_index_at_depth,
    normalized_rocking_curve,
)


def test_attenuation_factor_decreases_with_depth():
    depth = np.linspace(0.0, 50.0, 11)
    attenuation = attenuation_factor(depth, np.full(depth.shape, 20.0))

    assert attenuation[0] == 1.0
    assert np.all(np.diff(attenuation) < 0.0)


def test_nominal_layer_index_at_depth_maps_finite_layers():
    layers = [
        Layer(thickness=0.0),
        Layer(thickness=20.0),
        Layer(thickness=30.0),
        Layer(thickness=0.0),
    ]
    depth = np.array([0.0, 19.9, 20.0, 49.9])

    np.testing.assert_array_equal(nominal_layer_index_at_depth(layers, depth), [1, 1, 2, 2])


def test_graded_layer_property_smooths_rough_interface():
    layers = [
        Layer(thickness=0.0),
        Layer(thickness=20.0, roughness=2.0),
        Layer(thickness=20.0, roughness=2.0),
        Layer(thickness=0.0),
    ]
    values = [0.0, 1.0, 0.0, 0.0]
    depth = np.array([18.0, 20.0, 22.0])

    sampled = graded_layer_property_at_depth(layers, values, depth)

    assert sampled[0] > sampled[1] > sampled[2]


def test_constant_field_uniform_concentration_normalizes_flat():
    layers = [
        Layer(thickness=0.0),
        Layer(thickness=30.0),
        Layer(thickness=0.0),
    ]
    angles = np.array([1.0, 2.0, 3.0])

    curve = normalized_rocking_curve(
        angles=angles,
        energy_ev=3000.0,
        layers=layers,
        concentration_by_layer=[0.0, 1.0, 0.0],
        imfp_by_layer=[20.0, 20.0, 20.0],
        field_step=2.0,
    )

    np.testing.assert_allclose(curve.intensity, 1.0, rtol=1e-12, atol=1e-12)


def test_xps_integration_rejects_shape_mismatch():
    profile = FieldProfile(
        depth=np.array([0.0, 1.0]),
        electric_field=np.array([1.0, 1.0], dtype=complex),
        intensity=np.array([1.0, 1.0]),
        layer_index=np.array([1, 1]),
    )

    from swxps import integrate_xps_intensity

    with np.testing.assert_raises(ValueError):
        integrate_xps_intensity(profile, np.array([1.0]), np.array([20.0, 20.0]))
