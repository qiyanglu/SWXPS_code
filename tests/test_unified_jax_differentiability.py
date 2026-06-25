import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from swanx.stack import (  # noqa: E402
    Layer,
    LayerSlicingPolicy,
    SimulationStack,
    StackLayer,
    fixed_layer_grid,
    fixed_layer_grid_plan,
)
from swanx.workflows.simulate import ReflectivityRequest  # noqa: E402
from swanx.reflectivity_jax import transfer_matrix_reflectivity_jax  # noqa: E402
from swanx.simulation_jax import simulate_reflectivity_jax  # noqa: E402


def make_capacity_layers():
    return (
        Layer(0.0),
        Layer(8.0, delta=5.0e-6, beta=1.0e-7),
        Layer(0.0, delta=2.0e-6, beta=2.0e-7),
    )


@pytest.mark.xfail(
    raises=jax.errors.ConcretizationTypeError,
    strict=True,
    reason=(
        "generic high-level unified grid materialization converts traced layer "
        "thicknesses through Python floats/NumPy and is not end-to-end JAX-traceable"
    ),
)
def test_high_level_fixed_plan_materialization_is_not_yet_jax_differentiable():
    plan = fixed_layer_grid_plan(make_capacity_layers(), LayerSlicingPolicy())
    angles = np.linspace(0.8, 3.0, 7)
    target = jnp.linspace(0.02, 0.01, angles.size)

    def objective(thickness_vector):
        stack = SimulationStack(
            (
                StackLayer("vacuum", 0.0),
                StackLayer(
                    "film",
                    thickness_vector[0],
                    delta=5.0e-6,
                    beta=1.0e-7,
                ),
                StackLayer("substrate", 0.0, delta=2.0e-6, beta=2.0e-7),
            )
        )
        result = simulate_reflectivity_jax(
            ReflectivityRequest(
                angles=angles,
                energy_ev=3000.0,
                stack=stack,
                slicing=plan,
            )
        )
        residual = jnp.asarray(result.reflectivity) - target
        return jnp.sum(residual**2)

    x0 = jnp.array([6.0])
    assert jnp.isfinite(objective(x0))
    jax.grad(objective)(x0)


def test_jax_native_fixed_plan_reflectivity_objective_has_jitted_gradients():
    capacity_layers = make_capacity_layers()
    plan = fixed_layer_grid_plan(capacity_layers, LayerSlicingPolicy())
    capacity_grid = fixed_layer_grid(capacity_layers, plan)
    nominal_index = jnp.asarray(capacity_grid.nominal_layer_index, dtype=jnp.int32)
    counts = jnp.asarray(plan.slice_counts, dtype=jnp.float64)
    angles = jnp.linspace(0.8, 3.0, 11)
    nominal_delta = jnp.array([0.0, 5.0e-6, 2.0e-6])
    nominal_beta = jnp.array([0.0, 1.0e-7, 2.0e-7])

    def simulate(thickness_vector):
        widths = thickness_vector[nominal_index - 1] / counts[nominal_index - 1]
        effective_thickness = jnp.concatenate(
            (jnp.zeros((1,)), widths, jnp.zeros((1,)))
        )
        effective_delta = jnp.concatenate(
            (
                nominal_delta[:1],
                nominal_delta[nominal_index],
                nominal_delta[-1:],
            )
        )
        effective_beta = jnp.concatenate(
            (
                nominal_beta[:1],
                nominal_beta[nominal_index],
                nominal_beta[-1:],
            )
        )
        return transfer_matrix_reflectivity_jax(
            angles,
            3000.0,
            effective_thickness,
            effective_delta,
            effective_beta,
        )

    target = jax.lax.stop_gradient(simulate(jnp.array([7.0])))

    def objective(thickness_vector):
        residual = simulate(thickness_vector) - target
        return jnp.sum(residual**2)

    x0 = jnp.array([5.0])
    value = objective(x0)
    gradient = jax.grad(objective)(x0)
    objective_jit = jax.jit(objective)
    value_jit = objective_jit(x0)
    gradient_jit = jax.grad(objective_jit)(x0)

    assert np.isfinite(value)
    assert gradient.shape == x0.shape
    assert np.all(np.isfinite(gradient))
    assert np.any(np.abs(gradient) > 0.0)
    np.testing.assert_allclose(value_jit, value, rtol=1.0e-12, atol=1.0e-15)
    np.testing.assert_allclose(gradient_jit, gradient, rtol=1.0e-12, atol=1.0e-15)
