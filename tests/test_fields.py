import numpy as np

from swanx.optics import (
    depth_grid,
    effective_layers_with_roughness,
    electric_field_profile,
    parratt_amplitude,
    parratt_reflection_amplitudes,
    parratt_reflectivity,
    transfer_matrix_electric_field_profile,
    transfer_matrix_reflectivity,
)
from swanx.stack import Layer
from swanx.fields import (
    transfer_matrix_electric_field_profiles,
    transfer_matrix_reflectivity_array,
)


def test_depth_grid_covers_finite_layers_only():
    layers = [
        Layer(thickness=0.0),
        Layer(thickness=10.0),
        Layer(thickness=5.0),
        Layer(thickness=0.0),
    ]

    depth, layer_index = depth_grid(layers, step=5.0)

    np.testing.assert_allclose(depth, [0.0, 5.0, 10.0, 15.0])
    np.testing.assert_array_equal(layer_index, [1, 1, 1, 2])


def test_identical_index_stack_has_unit_field_intensity():
    layers = [
        Layer(thickness=0.0),
        Layer(thickness=20.0),
        Layer(thickness=30.0),
        Layer(thickness=0.0),
    ]

    profile = electric_field_profile(
        angle_deg=2.0,
        energy_ev=3000.0,
        layers=layers,
        step=2.0,
    )

    np.testing.assert_allclose(profile.intensity, 1.0, rtol=1e-12, atol=1e-12)


def test_surface_reflection_amplitude_matches_parratt_amplitude():
    layers = [
        Layer(thickness=0.0),
        Layer(thickness=20.0, delta=5.0e-6, beta=1.0e-7),
        Layer(thickness=0.0, delta=1.0e-5, beta=2.0e-7),
    ]

    amplitudes = parratt_reflection_amplitudes(
        angle_deg=2.5,
        energy_ev=3000.0,
        layers=layers,
    )

    assert amplitudes[0] == parratt_amplitude(2.5, 3000.0, layers)


def test_field_profile_rejects_array_angles():
    layers = [
        Layer(thickness=0.0),
        Layer(thickness=20.0),
        Layer(thickness=0.0),
    ]

    with np.testing.assert_raises(ValueError):
        electric_field_profile(np.array([1.0, 2.0]), 3000.0, layers)


def test_transfer_matrix_reflectivity_matches_parratt_for_sharp_stack():
    layers = [
        Layer(thickness=0.0),
        Layer(thickness=20.0, delta=5.0e-6, beta=1.0e-7),
        Layer(thickness=30.0, delta=2.5e-6, beta=8.0e-8),
        Layer(thickness=0.0, delta=1.0e-5, beta=2.0e-7),
    ]
    angles = np.linspace(0.5, 5.0, 25)

    transfer = np.array(
        [
            transfer_matrix_reflectivity(angle, 3000.0, layers)
            for angle in angles
        ]
    )
    parratt = parratt_reflectivity(angles, 3000.0, layers)

    np.testing.assert_allclose(transfer, parratt, rtol=1e-11, atol=1e-13)


def test_batched_transfer_matrix_reflectivity_matches_scalar():
    layers = [
        Layer(thickness=0.0),
        Layer(thickness=20.0, delta=5.0e-6, beta=1.0e-7, roughness=2.0),
        Layer(thickness=30.0, delta=2.5e-6, beta=8.0e-8, roughness=3.0),
        Layer(thickness=0.0, delta=1.0e-5, beta=2.0e-7, roughness=2.0),
    ]
    angles = np.linspace(0.5, 5.0, 25)

    batched = transfer_matrix_reflectivity_array(
        angles,
        3000.0,
        layers,
        roughness_step=0.5,
    )
    scalar = np.array(
        [
            transfer_matrix_reflectivity(
                angle,
                3000.0,
                layers,
                roughness_step=0.5,
            )
            for angle in angles
        ]
    )

    np.testing.assert_allclose(batched, scalar, rtol=1e-12, atol=1e-14)


def test_transfer_matrix_identical_index_stack_has_unit_field_intensity():
    layers = [
        Layer(thickness=0.0),
        Layer(thickness=20.0),
        Layer(thickness=30.0),
        Layer(thickness=0.0),
    ]

    profile = transfer_matrix_electric_field_profile(
        angle_deg=2.0,
        energy_ev=3000.0,
        layers=layers,
        step=2.0,
    )

    np.testing.assert_allclose(profile.intensity, 1.0, rtol=1e-12, atol=1e-12)


def test_batched_transfer_matrix_field_profiles_match_scalar():
    layers = [
        Layer(thickness=0.0),
        Layer(thickness=20.0, delta=5.0e-6, beta=1.0e-7, roughness=2.0),
        Layer(thickness=30.0, delta=2.5e-6, beta=8.0e-8, roughness=3.0),
        Layer(thickness=0.0, delta=1.0e-5, beta=2.0e-7, roughness=2.0),
    ]
    angles = np.array([1.0, 2.0, 3.0])

    batched = transfer_matrix_electric_field_profiles(
        angles,
        3000.0,
        layers,
        step=2.0,
        roughness_step=1.0,
    )

    for angle, profile in zip(angles, batched):
        scalar = transfer_matrix_electric_field_profile(
            angle,
            3000.0,
            layers,
            step=2.0,
            roughness_step=1.0,
        )
        np.testing.assert_allclose(profile.depth, scalar.depth)
        np.testing.assert_array_equal(profile.layer_index, scalar.layer_index)
        np.testing.assert_allclose(profile.electric_field, scalar.electric_field)
        np.testing.assert_allclose(profile.intensity, scalar.intensity)


def test_effective_layers_leave_zero_roughness_stack_unchanged():
    layers = [
        Layer(thickness=0.0),
        Layer(thickness=20.0, delta=5.0e-6, beta=1.0e-7),
        Layer(thickness=0.0),
    ]

    effective = effective_layers_with_roughness(layers, step=1.0)

    assert effective == layers


def test_effective_layers_discretize_rough_interfaces():
    layers = [
        Layer(thickness=0.0),
        Layer(thickness=20.0, delta=5.0e-6, beta=1.0e-7, roughness=2.0),
        Layer(thickness=0.0, delta=1.0e-5, beta=2.0e-7, roughness=2.0),
    ]

    effective = effective_layers_with_roughness(layers, step=1.0)

    assert len(effective) > len(layers)
    assert sum(layer.thickness for layer in effective[1:-1]) == 20.0
    assert all(layer.roughness == 0.0 for layer in effective)


def test_effective_layers_accept_per_finite_layer_steps():
    layers = [
        Layer(thickness=0.0),
        Layer(thickness=10.0, delta=5.0e-6, roughness=2.0),
        Layer(thickness=20.0, delta=2.0e-6, roughness=2.0),
        Layer(thickness=0.0, delta=1.0e-5, roughness=2.0),
    ]

    effective = effective_layers_with_roughness(layers, step=[2.0, 5.0])

    assert len(effective[1:-1]) == 9
    np.testing.assert_allclose(
        [layer.thickness for layer in effective[1:-1]],
        [2.0] * 5 + [5.0] * 4,
    )


def test_effective_layers_reject_wrong_number_of_layer_steps():
    layers = [
        Layer(thickness=0.0),
        Layer(thickness=10.0, roughness=2.0),
        Layer(thickness=20.0, roughness=2.0),
        Layer(thickness=0.0, roughness=2.0),
    ]

    with np.testing.assert_raises(ValueError):
        effective_layers_with_roughness(layers, step=[1.0])


def test_transfer_matrix_roughness_damps_high_angle_reflectivity():
    sharp_layers = [
        Layer(thickness=0.0),
        Layer(thickness=20.0, delta=8.0e-6, beta=1.0e-7),
        Layer(thickness=20.0, delta=2.0e-6, beta=8.0e-8),
        Layer(thickness=0.0, delta=7.0e-6, beta=1.5e-7),
    ]
    rough_layers = [
        Layer(thickness=0.0),
        Layer(thickness=20.0, delta=8.0e-6, beta=1.0e-7, roughness=4.0),
        Layer(thickness=20.0, delta=2.0e-6, beta=8.0e-8, roughness=4.0),
        Layer(thickness=0.0, delta=7.0e-6, beta=1.5e-7, roughness=4.0),
    ]

    sharp = transfer_matrix_reflectivity(5.0, 3000.0, sharp_layers)
    rough = transfer_matrix_reflectivity(5.0, 3000.0, rough_layers, roughness_step=0.5)

    assert rough < sharp
