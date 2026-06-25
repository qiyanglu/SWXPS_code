import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from swanx.fitting import ReflectivityData
from swanx.jax_least_squares import (
    JaxLeastSquaresResidualSettings,
    build_jax_residual_function,
)


def test_compilation_counter_counts_shapes_not_repeated_values():
    data = ReflectivityData(
        name="R",
        angles=np.array([1.0, 2.0, 3.0]),
        reflectivity=np.array([1.0, 2.0, 3.0]),
    )

    def simulate_curves(parameters):
        return parameters[0] * jnp.array([1.0, 2.0, 3.0]), ()

    residual = build_jax_residual_function(
        simulate_curves,
        reflectivity=data,
        settings=JaxLeastSquaresResidualSettings(reflectivity_log=False),
    )

    assert residual.compilation_counter.total_compilations == 0
    residual([0.5])
    residual([0.8])
    assert residual.compilation_counter.residual_compilations == 1
    residual.jacobian([0.5])
    residual.jacobian([0.8])
    assert residual.compilation_counter.jacobian_compilations == 1
    assert residual.compilation_counter.total_compilations == 2
