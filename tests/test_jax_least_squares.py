import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("scipy.optimize")

from swxps.fitting import FitParameter, ReflectivityData, RockingCurveData
from swxps.jax_least_squares import (
    JaxLeastSquaresOptimizerSettings,
    JaxLeastSquaresResidualSettings,
    build_jax_residual_function,
    optimize_with_jax_least_squares,
)


def _simulate_linear_curves(parameters):
    thickness, offset = parameters
    reflectivity = jnp.array(
        [1.0 + thickness, 1.5 + 2.0 * thickness + offset]
    )
    rocking_curve_1 = jnp.array(
        [thickness + offset, 2.0 * thickness - offset, offset]
    )
    rocking_curve_2 = jnp.array([3.0 * thickness + offset])
    return reflectivity, (rocking_curve_1, rocking_curve_2)


def _synthetic_data():
    true_parameters = jnp.array([0.7, 0.3])
    reflectivity, rocking_curves = _simulate_linear_curves(true_parameters)
    reflectivity_data = ReflectivityData(
        name="reflectivity",
        angles=np.array([1.0, 2.0]),
        reflectivity=np.asarray(reflectivity),
        weight=2.0,
    )
    rocking_curve_data = (
        RockingCurveData(
            name="rc-1",
            angles=np.array([1.0, 2.0, 3.0]),
            intensity=np.asarray(rocking_curves[0]),
            weight=1.5,
        ),
        RockingCurveData(
            name="rc-2",
            angles=np.array([1.0]),
            intensity=np.asarray(rocking_curves[1]),
            weight=0.5,
        ),
    )
    settings = JaxLeastSquaresResidualSettings(
        reflectivity_log=False,
        rocking_curve_normalization="none",
    )
    return reflectivity_data, rocking_curve_data, settings


def test_residual_and_jacobian_shapes_for_reflectivity_and_multiple_rcs():
    reflectivity, rocking_curves, settings = _synthetic_data()
    residual_function = build_jax_residual_function(
        _simulate_linear_curves,
        reflectivity=reflectivity,
        rocking_curves=rocking_curves,
        settings=settings,
    )

    residuals = residual_function(np.array([0.2, 0.8]))
    jacobian = residual_function.jacobian(np.array([0.2, 0.8]))

    assert residuals.shape == (6,)
    assert jacobian.shape == (6, 2)
    assert np.all(np.isfinite(residuals))
    assert np.all(np.isfinite(jacobian))


def test_jax_least_squares_reduces_cost_and_respects_bounds():
    reflectivity, rocking_curves, settings = _synthetic_data()
    residual_function = build_jax_residual_function(
        _simulate_linear_curves,
        reflectivity=reflectivity,
        rocking_curves=rocking_curves,
        settings=settings,
    )
    parameters = (
        FitParameter("thickness", 0.0, 1.0, initial=0.2),
        FitParameter("offset", 0.0, 1.0, initial=0.8),
    )
    initial_vector = np.array([0.2, 0.8])
    initial_residuals = residual_function(initial_vector)
    initial_cost = 0.5 * float(np.dot(initial_residuals, initial_residuals))

    result = optimize_with_jax_least_squares(
        parameters,
        residual_function,
        settings=JaxLeastSquaresOptimizerSettings(max_nfev=50),
    )

    assert result.success
    assert result.final_cost < initial_cost
    assert result.final_residuals.shape == (6,)
    assert result.final_jacobian.shape == (6, 2)
    assert result.nfev > 0
    assert result.njev is not None
    assert result.history
    assert result.covariance is not None
    for parameter in parameters:
        value = result.best_parameters[parameter.name]
        assert parameter.lower <= value <= parameter.upper
    np.testing.assert_allclose(
        [result.best_parameters["thickness"], result.best_parameters["offset"]],
        [0.7, 0.3],
        atol=1.0e-7,
    )
