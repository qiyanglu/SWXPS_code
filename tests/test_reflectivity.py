import numpy as np

from swanx.optics import (
    apply_roughness,
    energy_to_wavelength,
    fresnel_r_s,
    kz_in_layers,
    parratt_reflectivity,
)
from swanx.stack import (
    Layer,
    vacuum,
)


def test_two_layer_stack_reproduces_fresnel_reflectivity():
    energy_ev = 8000.0
    angles = np.linspace(0.2, 5.0, 100)
    layers = [
        vacuum(),
        Layer(thickness=0.0, delta=7.5e-6, beta=1.0e-7),
    ]

    wavelength = energy_to_wavelength(energy_ev)
    kz = kz_in_layers(angles, wavelength, [layer.n for layer in layers])
    expected = np.abs(fresnel_r_s(kz[0], kz[1])) ** 2
    actual = parratt_reflectivity(angles, energy_ev, layers)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


def test_identical_refractive_indices_give_zero_reflectivity():
    energy_ev = 8000.0
    angles = np.linspace(0.2, 6.0, 120)
    layers = [
        Layer(thickness=0.0, delta=2.0e-6, beta=5.0e-8),
        Layer(thickness=30.0, delta=2.0e-6, beta=5.0e-8),
        Layer(thickness=45.0, delta=2.0e-6, beta=5.0e-8),
        Layer(thickness=0.0, delta=2.0e-6, beta=5.0e-8),
    ]

    reflectivity = parratt_reflectivity(angles, energy_ev, layers)

    assert np.max(reflectivity) < 1e-24


def test_periodic_multilayer_has_first_bragg_peak_near_expected_angle():
    energy_ev = 8000.0
    wavelength = energy_to_wavelength(energy_ev)
    period = 40.0
    expected_theta = np.rad2deg(np.arcsin(wavelength / (2.0 * period)))

    layers = [vacuum()]
    for _ in range(24):
        layers.append(Layer(thickness=20.0, delta=8.0e-6, beta=2.0e-8))
        layers.append(Layer(thickness=20.0, delta=2.0e-6, beta=2.0e-8))
    layers.append(Layer(thickness=0.0, delta=2.0e-6, beta=2.0e-8))

    angles = np.linspace(0.5, 1.8, 900)
    reflectivity = parratt_reflectivity(angles, energy_ev, layers)
    peak_theta = angles[np.argmax(reflectivity)]

    assert abs(peak_theta - expected_theta) < 0.12


def test_reflectivity_does_not_exceed_one_for_absorbing_stack():
    energy_ev = 8000.0
    angles = np.linspace(0.1, 8.0, 300)
    layers = [
        vacuum(),
        Layer(thickness=35.0, delta=8.0e-6, beta=1.0e-7),
        Layer(thickness=55.0, delta=3.0e-6, beta=8.0e-8),
        Layer(thickness=0.0, delta=7.0e-6, beta=1.5e-7),
    ]

    reflectivity = parratt_reflectivity(angles, energy_ev, layers)

    assert np.all(reflectivity >= 0.0)
    assert np.max(reflectivity) <= 1.0 + 1e-12


def test_zero_roughness_matches_sharp_interface_behavior():
    energy_ev = 8000.0
    angles = np.linspace(0.2, 5.0, 100)
    sharp_layers = [
        vacuum(),
        Layer(thickness=35.0, delta=8.0e-6, beta=1.0e-7),
        Layer(thickness=0.0, delta=7.0e-6, beta=1.5e-7),
    ]
    zero_roughness_layers = [
        vacuum(),
        Layer(thickness=35.0, delta=8.0e-6, beta=1.0e-7, roughness=0.0),
        Layer(thickness=0.0, delta=7.0e-6, beta=1.5e-7, roughness=0.0),
    ]

    np.testing.assert_array_equal(
        parratt_reflectivity(angles, energy_ev, zero_roughness_layers),
        parratt_reflectivity(angles, energy_ev, sharp_layers),
    )


def test_positive_roughness_damps_fresnel_amplitude():
    energy_ev = 8000.0
    angle = 4.0
    roughness = 6.0
    wavelength = energy_to_wavelength(energy_ev)
    kz = kz_in_layers(angle, wavelength, [1.0 + 0.0j, 1.0 - 7.5e-6 + 1.0e-7j])
    sharp_amplitude = fresnel_r_s(kz[0], kz[1])
    rough_amplitude = apply_roughness(sharp_amplitude, kz[0], kz[1], roughness)

    assert abs(rough_amplitude) < abs(sharp_amplitude)


def test_negative_roughness_is_rejected():
    layers = [
        vacuum(),
        Layer(thickness=0.0, delta=7.5e-6, beta=1.0e-7, roughness=-1.0),
    ]

    with np.testing.assert_raises(ValueError):
        parratt_reflectivity(1.0, 8000.0, layers)
