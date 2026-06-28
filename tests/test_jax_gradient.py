from types import SimpleNamespace

import numpy as np
import pytest

from swanx.fitting import (
    FitParameter,
    JaxGradientOptimizerSettings,
    optimize_with_jax_gradient,
    physical_to_scaled,
    scaled_to_physical,
)


def test_parameter_scaling_round_trips():
    lower = np.array([-1.0, 10.0])
    upper = np.array([1.0, 30.0])
    physical = np.array([0.0, 20.0])

    scaled = physical_to_scaled(physical, lower, upper)

    np.testing.assert_allclose(scaled, [0.5, 0.5])
    np.testing.assert_allclose(scaled_to_physical(scaled, lower, upper), physical)


def test_jax_gradient_optimizer_reduces_loss_and_records_history(monkeypatch):
    import swanx.fitting.jax_gradient as jax_gradient

    captured = {}

    def fake_minimize(func, x0, method, jac, bounds, callback, options):
        assert method == "L-BFGS-B"
        assert jac is True
        assert bounds == [(0.0, 1.0)]
        captured["options"] = options
        loss0, grad0 = func(x0)
        x1 = np.clip(x0 - 0.01 * grad0, 0.0, 1.0)
        loss1, _ = func(x1)
        callback(x1)
        return SimpleNamespace(
            x=x1,
            fun=loss1,
            status=0,
            message="converged",
            success=True,
            nit=1,
            nfev=2,
            initial_loss=loss0,
        )

    monkeypatch.setattr(jax_gradient, "_load_scipy_minimize", lambda: fake_minimize)

    parameters = (FitParameter("thickness", 0.0, 10.0, initial=8.0),)

    def value_and_grad(vector):
        residual = vector[0] - 2.0
        return residual**2, np.array([2.0 * residual])

    initial_loss = value_and_grad(np.array([8.0]))[0]
    result = optimize_with_jax_gradient(
        parameters,
        value_and_grad,
        settings=JaxGradientOptimizerSettings(maxiter=5),
    )

    assert result.best_loss < initial_loss
    assert result.best_parameters["thickness"] == 0.0
    assert result.success is True
    assert result.nit == 1
    assert result.nfev == 2
    assert captured["options"]["maxiter"] == 5
    assert len(result.history) == 1
    assert result.history[0].iteration == 1
    assert result.history[0].gradient_norm >= 0.0


def test_jax_gradient_optimizer_keeps_result_inside_bounds(monkeypatch):
    import swanx.fitting.jax_gradient as jax_gradient

    def fake_minimize(func, x0, method, jac, bounds, callback, options):
        del method, jac, callback, options
        loss, _ = func(np.array([bounds[0][1]]))
        return SimpleNamespace(
            x=np.array([bounds[0][1]]),
            fun=loss,
            status=0,
            message="bounded",
            success=True,
            nit=1,
            nfev=1,
        )

    monkeypatch.setattr(jax_gradient, "_load_scipy_minimize", lambda: fake_minimize)

    parameters = (FitParameter("roughness", 1.0, 5.0, initial=2.0),)

    def value_and_grad(vector):
        residual = vector[0] - 10.0
        return residual**2, np.array([2.0 * residual])

    result = optimize_with_jax_gradient(parameters, value_and_grad)

    assert 1.0 <= result.best_parameters["roughness"] <= 5.0


def test_jax_gradient_optimizer_can_call_existing_jax_value_and_grad(monkeypatch):
    jax_reflectivity = pytest.importorskip(
        "swanx.reflectivity_jax",
        exc_type=ImportError,
    )
    import swanx.fitting.jax_gradient as jax_gradient

    def fake_minimize(func, x0, method, jac, bounds, callback, options):
        del method, jac, bounds, callback, options
        loss, _ = func(x0)
        return SimpleNamespace(
            x=x0,
            fun=loss,
            status=0,
            message="evaluated",
            success=True,
            nit=0,
            nfev=1,
        )

    monkeypatch.setattr(jax_gradient, "_load_scipy_minimize", lambda: fake_minimize)

    angles = np.linspace(0.5, 3.0, 16)
    thicknesses = np.array([0.0, 20.0, 0.0])
    deltas = np.array([0.0, 5.0e-6, 1.0e-5])
    betas = np.array([0.0, 1.0e-7, 2.0e-7])
    roughnesses = np.array([0.0, 0.0, 0.0])
    target = np.asarray(
        jax_reflectivity.jitted_parratt_reflectivity(
            angles,
            3000.0,
            thicknesses,
            deltas,
            betas,
            roughnesses,
        )
    )

    def value_and_grad(vector):
        trial_thicknesses = thicknesses.copy()
        trial_thicknesses[1] = vector[0]
        loss, gradient = jax_reflectivity.jitted_value_and_grad_reflectivity_loss(
            trial_thicknesses,
            angles,
            3000.0,
            deltas,
            betas,
            roughnesses,
            target,
        )
        return float(loss), np.asarray([gradient[1]], dtype=float)

    result = optimize_with_jax_gradient(
        (FitParameter("film_thickness", 5.0, 40.0, initial=20.0),),
        value_and_grad,
        initial=np.array([20.0]),
    )

    assert np.isfinite(result.best_loss)
