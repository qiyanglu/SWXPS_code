import jax
import jax.numpy as jnp
import numpy as np

from swxps import Layer, parratt_reflectivity, reflectivity_forward_jax, vacuum


def _arrays_from_layers(layers):
    return (
        np.asarray([layer.thickness for layer in layers], dtype=float),
        np.asarray([layer.delta for layer in layers], dtype=float),
        np.asarray([layer.beta for layer in layers], dtype=float),
        np.asarray([layer.roughness for layer in layers], dtype=float),
    )


def test_jax_forward_matches_numpy_two_layer_reflectivity():
    energy_ev = 8000.0
    angles = np.linspace(0.2, 5.0, 80)
    layers = [
        vacuum(),
        Layer(thickness=0.0, delta=7.5e-6, beta=1.0e-7),
    ]
    thickness, delta, beta, roughness = _arrays_from_layers(layers)

    expected = parratt_reflectivity(angles, energy_ev, layers)
    actual = np.asarray(
        reflectivity_forward_jax(angles, energy_ev, thickness, delta, beta, roughness)
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-13)


def test_jax_forward_matches_numpy_multilayer_reflectivity():
    energy_ev = 8000.0
    angles = np.linspace(0.25, 4.0, 100)
    layers = [
        vacuum(),
        Layer(thickness=35.0, delta=8.0e-6, beta=1.0e-7, roughness=3.0),
        Layer(thickness=55.0, delta=3.0e-6, beta=8.0e-8, roughness=4.0),
        Layer(thickness=0.0, delta=7.0e-6, beta=1.5e-7, roughness=2.0),
    ]
    thickness, delta, beta, roughness = _arrays_from_layers(layers)

    expected = parratt_reflectivity(angles, energy_ev, layers)
    actual = np.asarray(
        reflectivity_forward_jax(angles, energy_ev, thickness, delta, beta, roughness)
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-13)


def test_jax_forward_identical_indices_give_near_zero_reflectivity():
    angles = np.linspace(0.2, 6.0, 120)
    thickness = np.asarray([0.0, 30.0, 45.0, 0.0])
    delta = np.asarray([2.0e-6, 2.0e-6, 2.0e-6, 2.0e-6])
    beta = np.asarray([5.0e-8, 5.0e-8, 5.0e-8, 5.0e-8])
    roughness = np.zeros_like(thickness)

    reflectivity = np.asarray(
        reflectivity_forward_jax(angles, 8000.0, thickness, delta, beta, roughness)
    )

    assert np.max(reflectivity) < 1e-24


def test_jax_forward_is_differentiable_for_scalar_loss():
    angles = jnp.linspace(0.4, 3.0, 40)
    energy_ev = 8000.0
    thickness = jnp.asarray([0.0, 35.0, 55.0, 0.0], dtype=jnp.float64)
    delta = jnp.asarray([0.0, 8.0e-6, 3.0e-6, 7.0e-6], dtype=jnp.float64)
    beta = jnp.asarray([0.0, 1.0e-7, 8.0e-8, 1.5e-7], dtype=jnp.float64)
    roughness = jnp.asarray([0.0, 3.0, 4.0, 2.0], dtype=jnp.float64)

    def loss(thickness_value, delta_value, beta_value, roughness_value):
        reflectivity = reflectivity_forward_jax(
            angles,
            energy_ev,
            thickness_value,
            delta_value,
            beta_value,
            roughness_value,
        )
        return jnp.mean(jnp.log10(reflectivity + 1e-16))

    gradients = jax.grad(loss, argnums=(0, 1, 2, 3))(thickness, delta, beta, roughness)

    for gradient in gradients:
        assert gradient.shape == thickness.shape
        assert bool(jnp.all(jnp.isfinite(gradient)))



def test_jax_forward_can_be_explicitly_jitted():
    energy_ev = 8000.0
    angles = jnp.linspace(0.25, 4.0, 64)
    thickness = jnp.asarray([0.0, 35.0, 55.0, 0.0], dtype=jnp.float64)
    delta = jnp.asarray([0.0, 8.0e-6, 3.0e-6, 7.0e-6], dtype=jnp.float64)
    beta = jnp.asarray([0.0, 1.0e-7, 8.0e-8, 1.5e-7], dtype=jnp.float64)
    roughness = jnp.asarray([0.0, 3.0, 4.0, 2.0], dtype=jnp.float64)

    jitted_forward = jax.jit(reflectivity_forward_jax)
    expected = reflectivity_forward_jax(angles, energy_ev, thickness, delta, beta, roughness)
    actual = jitted_forward(angles, energy_ev, thickness, delta, beta, roughness)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-12, atol=1e-14)


def test_jit_compiled_objective_has_finite_gradients():
    angles = jnp.linspace(0.4, 3.0, 48)
    target = jnp.linspace(0.02, 0.08, 48)
    energy_ev = 8000.0
    thickness = jnp.asarray([0.0, 35.0, 55.0, 0.0], dtype=jnp.float64)
    delta = jnp.asarray([0.0, 8.0e-6, 3.0e-6, 7.0e-6], dtype=jnp.float64)
    beta = jnp.asarray([0.0, 1.0e-7, 8.0e-8, 1.5e-7], dtype=jnp.float64)
    roughness = jnp.asarray([0.0, 3.0, 4.0, 2.0], dtype=jnp.float64)

    @jax.jit
    def objective(thickness_value, delta_value, beta_value, roughness_value):
        model = reflectivity_forward_jax(
            angles,
            energy_ev,
            thickness_value,
            delta_value,
            beta_value,
            roughness_value,
        )
        residual = jnp.log10(model + 1e-16) - jnp.log10(target + 1e-16)
        return jnp.mean(residual**2)

    gradients = jax.grad(objective, argnums=(0, 1, 2, 3))(thickness, delta, beta, roughness)

    for gradient in gradients:
        assert gradient.shape == thickness.shape
        assert bool(jnp.all(jnp.isfinite(gradient)))


def test_jax_forward_preserves_scalar_angle_shape_without_branch():
    reflectivity = reflectivity_forward_jax(
        1.0,
        8000.0,
        jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        jnp.asarray([0.0, 7.5e-6], dtype=jnp.float64),
        jnp.asarray([0.0, 1.0e-7], dtype=jnp.float64),
        jnp.asarray([0.0, 0.0], dtype=jnp.float64),
    )

    assert reflectivity.shape == ()
    assert bool(jnp.isfinite(reflectivity))
