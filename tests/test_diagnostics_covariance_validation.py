import numpy as np
import pytest

from swanx import FitParameter, JaxLeastSquaresOptimizationResult
from swanx.diagnostics import (
    compute_parameter_diagnostics,
    diagnostics_from_least_squares_result,
)


def test_valid_covariance_produces_symmetric_bounded_correlation():
    diagnostics = compute_parameter_diagnostics(
        [1.0, 2.0],
        covariance=np.array([[4.0, 1.5], [1.5, 9.0]]),
    )

    np.testing.assert_allclose(diagnostics.correlation, diagnostics.correlation.T)
    np.testing.assert_allclose(np.diag(diagnostics.correlation), 1.0)
    assert np.nanmax(np.abs(diagnostics.correlation)) <= 1.0
    assert diagnostics.correlation[0, 1] == pytest.approx(0.25)


def test_non_symmetric_covariance_warns_and_is_symmetrized():
    covariance = np.array([[4.0, 2.0], [1.0, 9.0]])

    with pytest.warns(RuntimeWarning, match="not approximately symmetric"):
        diagnostics = compute_parameter_diagnostics(
            [1.0, 2.0],
            covariance=covariance,
        )

    expected = 0.5 * (covariance + covariance.T)
    np.testing.assert_allclose(diagnostics.covariance, expected)
    np.testing.assert_allclose(diagnostics.correlation, diagnostics.correlation.T)
    assert diagnostics.correlation[0, 1] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        (np.array([[1.0, np.nan], [np.nan, 1.0]]), "finite values"),
        (np.array([[1.0, 0.0], [0.0, -0.1]]), "negative variances"),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), "positive semidefinite"),
    ],
)
def test_bad_external_covariance_is_rejected(covariance, message):
    with pytest.raises(ValueError, match=message):
        compute_parameter_diagnostics([1.0, 2.0], covariance=covariance)


def test_tiny_covariance_roundoff_is_projected_and_correlation_is_clipped():
    covariance = np.array([[1.0, 1.0 + 1.0e-13], [1.0 + 1.0e-13, 1.0]])

    with pytest.warns(RuntimeWarning, match="tiny negative eigenvalues"):
        diagnostics = compute_parameter_diagnostics(
            [1.0, 2.0],
            covariance=covariance,
        )

    np.testing.assert_allclose(diagnostics.correlation, np.ones((2, 2)))
    assert np.nanmax(np.abs(diagnostics.correlation)) <= 1.0


def test_least_squares_adapter_ignores_invalid_cached_covariance():
    residuals = np.array([0.1, -0.2, 0.05, 0.02])
    jacobian = np.array(
        [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [0.5, -0.25]]
    )
    result = JaxLeastSquaresOptimizationResult(
        best_parameters={"a": 1.25, "b": -0.25},
        final_cost=0.5 * float(residuals @ residuals),
        final_residuals=residuals,
        final_jacobian=jacobian,
        status=1,
        message="test",
        success=True,
        nfev=2,
        njev=2,
        optimality=0.0,
        history=(),
        covariance=np.array([[1.0, 3.0], [-2.0, 1.0]]),
        raw_result=object(),
        total_seconds=0.1,
    )
    parameters = (
        FitParameter("a", 0.0, 2.0),
        FitParameter("b", -1.0, 1.0),
    )

    diagnostics = diagnostics_from_least_squares_result(result, parameters)

    dof = residuals.size - len(parameters)
    expected = (float(residuals @ residuals) / dof) * np.linalg.pinv(
        jacobian.T @ jacobian,
        rcond=1.0e-12,
    )
    expected = 0.5 * (expected + expected.T)
    np.testing.assert_allclose(diagnostics.covariance, expected)
    np.testing.assert_allclose(diagnostics.correlation, diagnostics.correlation.T)
    assert np.nanmax(np.abs(diagnostics.correlation)) <= 1.0
