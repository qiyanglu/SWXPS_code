import numpy as np
import pytest

jax_reflectivity = pytest.importorskip(
    "swxps.reflectivity_jax",
    exc_type=ImportError,
)

from swxps import (
    Layer,
    depth_grid,
    electric_field_profile,
    nominal_layer_index_at_depth,
    normalized_rocking_curve,
    parratt_reflectivity,
    vacuum,
)


def test_jax_parratt_reflectivity_matches_numpy_core():
    energy_ev = 8000.0
    angles = np.linspace(0.3, 4.0, 64)
    layers = [
        vacuum(),
        Layer(thickness=24.0, delta=8.0e-6, beta=1.0e-7, roughness=1.5),
        Layer(thickness=31.0, delta=3.0e-6, beta=7.0e-8, roughness=2.0),
        Layer(thickness=0.0, delta=7.0e-6, beta=1.5e-7, roughness=2.5),
    ]
    thicknesses, deltas, betas, roughnesses = jax_reflectivity.layer_arrays_from_layers(
        layers
    )

    actual = np.asarray(
        jax_reflectivity.jitted_parratt_reflectivity(
            angles,
            energy_ev,
            thicknesses,
            deltas,
            betas,
            roughnesses,
        )
    )
    expected = parratt_reflectivity(angles, energy_ev, layers)

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-13)


def test_jax_reflectivity_loss_gradient_matches_finite_difference():
    energy_ev = 8000.0
    angles = np.linspace(0.4, 3.5, 48)
    layers = [
        vacuum(),
        Layer(thickness=22.0, delta=8.0e-6, beta=1.0e-7, roughness=0.5),
        Layer(thickness=34.0, delta=3.0e-6, beta=7.0e-8, roughness=0.75),
        Layer(thickness=0.0, delta=7.0e-6, beta=1.5e-7, roughness=1.0),
    ]
    thicknesses, deltas, betas, roughnesses = jax_reflectivity.layer_arrays_from_layers(
        layers
    )
    target = parratt_reflectivity(angles, energy_ev, layers) * 1.01

    value, gradient = jax_reflectivity.jitted_value_and_grad_reflectivity_loss(
        thicknesses,
        angles,
        energy_ev,
        deltas,
        betas,
        roughnesses,
        target,
    )

    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(gradient)))

    step = 1.0e-3
    plus = thicknesses.copy()
    minus = thicknesses.copy()
    plus[1] += step
    minus[1] -= step
    loss_plus = jax_reflectivity.reflectivity_mse_loss(
        plus,
        angles,
        energy_ev,
        deltas,
        betas,
        roughnesses,
        target,
    )
    loss_minus = jax_reflectivity.reflectivity_mse_loss(
        minus,
        angles,
        energy_ev,
        deltas,
        betas,
        roughnesses,
        target,
    )
    finite_difference = (float(loss_plus) - float(loss_minus)) / (2.0 * step)

    np.testing.assert_allclose(
        np.asarray(gradient)[1],
        finite_difference,
        rtol=1e-3,
        atol=1e-14,
    )


def test_jax_electric_field_intensity_matches_numpy_parratt_profile():
    energy_ev = 3000.0
    angles = np.array([1.0, 2.0, 3.0])
    layers = [
        vacuum(),
        Layer(thickness=20.0, delta=5.0e-6, beta=1.0e-7),
        Layer(thickness=30.0, delta=2.5e-6, beta=8.0e-8),
        Layer(thickness=0.0, delta=1.0e-5, beta=2.0e-7),
    ]
    depth, layer_index = depth_grid(layers, step=2.0)
    thicknesses, deltas, betas, roughnesses = jax_reflectivity.layer_arrays_from_layers(
        layers
    )

    actual = np.asarray(
        jax_reflectivity.jitted_electric_field_intensity(
            angles,
            energy_ev,
            thicknesses,
            deltas,
            betas,
            roughnesses,
            depth,
            layer_index,
        )
    )
    expected = np.column_stack(
        [
            electric_field_profile(
                angle,
                energy_ev,
                layers,
                step=2.0,
            ).intensity
            for angle in angles
        ]
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


def test_jax_normalized_rocking_curve_matches_numpy_sharp_stack():
    energy_ev = 3000.0
    angles = np.array([1.0, 1.5, 2.0, 2.5])
    layers = [
        vacuum(),
        Layer(thickness=20.0, delta=5.0e-6, beta=1.0e-7),
        Layer(thickness=30.0, delta=2.5e-6, beta=8.0e-8),
        Layer(thickness=0.0, delta=1.0e-5, beta=2.0e-7),
    ]
    concentration_by_layer = np.array([0.0, 1.0, 0.25, 0.0])
    imfp_by_layer = np.array([20.0, 20.0, 30.0, 30.0])
    depth, layer_index = depth_grid(layers, step=2.0)
    sampled_layers = nominal_layer_index_at_depth(layers, depth)
    concentration = concentration_by_layer[sampled_layers]
    attenuation_length = imfp_by_layer[sampled_layers]
    offpeak_mask = np.ones(angles.shape, dtype=bool)
    thicknesses, deltas, betas, roughnesses = jax_reflectivity.layer_arrays_from_layers(
        layers
    )

    actual, raw, normalization = jax_reflectivity.jitted_normalized_rocking_curve(
        angles,
        energy_ev,
        thicknesses,
        deltas,
        betas,
        roughnesses,
        depth,
        layer_index,
        concentration,
        attenuation_length,
        0.0,
        offpeak_mask,
    )
    expected = normalized_rocking_curve(
        angles=angles,
        energy_ev=energy_ev,
        layers=layers,
        concentration_by_layer=concentration_by_layer,
        imfp_by_layer=imfp_by_layer,
        field_step=2.0,
    )

    np.testing.assert_allclose(np.asarray(actual), expected.intensity, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(np.asarray(raw), expected.raw_intensity, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(float(normalization), expected.normalization, rtol=1e-11, atol=1e-12)


def test_jax_rocking_curve_loss_gradient_matches_finite_difference():
    energy_ev = 3000.0
    angles = np.array([1.0, 1.5, 2.0, 2.5])
    layers = [
        vacuum(),
        Layer(thickness=20.0, delta=5.0e-6, beta=1.0e-7),
        Layer(thickness=30.0, delta=2.5e-6, beta=8.0e-8),
        Layer(thickness=0.0, delta=1.0e-5, beta=2.0e-7),
    ]
    depth, layer_index = depth_grid(layers, step=2.0)
    sampled_layers = nominal_layer_index_at_depth(layers, depth)
    concentration = np.array([0.0, 1.0, 0.25, 0.0])[sampled_layers]
    attenuation_length = np.array([20.0, 20.0, 30.0, 30.0])[sampled_layers]
    offpeak_mask = np.ones(angles.shape, dtype=bool)
    thicknesses, deltas, betas, roughnesses = jax_reflectivity.layer_arrays_from_layers(
        layers
    )
    target, _, _ = jax_reflectivity.normalized_rocking_curve_jax(
        angles,
        energy_ev,
        thicknesses,
        deltas,
        betas,
        roughnesses,
        depth,
        layer_index,
        concentration,
        attenuation_length,
        0.0,
        offpeak_mask,
    )
    target = np.asarray(target) * np.array([1.01, 0.99, 1.005, 0.995])

    value, gradient = jax_reflectivity.jitted_value_and_grad_rocking_curve_loss(
        thicknesses,
        angles,
        energy_ev,
        deltas,
        betas,
        roughnesses,
        depth,
        layer_index,
        concentration,
        attenuation_length,
        0.0,
        offpeak_mask,
        target,
    )

    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(gradient)))

    step = 1.0e-3
    plus = thicknesses.copy()
    minus = thicknesses.copy()
    plus[1] += step
    minus[1] -= step
    loss_plus = jax_reflectivity.rocking_curve_mse_loss(
        plus,
        angles,
        energy_ev,
        deltas,
        betas,
        roughnesses,
        depth,
        layer_index,
        concentration,
        attenuation_length,
        0.0,
        offpeak_mask,
        target,
    )
    loss_minus = jax_reflectivity.rocking_curve_mse_loss(
        minus,
        angles,
        energy_ev,
        deltas,
        betas,
        roughnesses,
        depth,
        layer_index,
        concentration,
        attenuation_length,
        0.0,
        offpeak_mask,
        target,
    )
    finite_difference = (float(loss_plus) - float(loss_minus)) / (2.0 * step)

    np.testing.assert_allclose(
        np.asarray(gradient)[1],
        finite_difference,
        rtol=1e-3,
        atol=1e-14,
    )
