import numpy as np
import pytest

from swanx.diagnostics import (
    compute_parameter_diagnostics,
    diagnostics_from_least_squares_result,
    plot_correlation_matrix,
    plot_parameter_estimates,
    plot_singular_values,
)
from swanx.fitting import (
    FitParameter,
    JaxLeastSquaresOptimizationResult,
)


def test_full_rank_covariance_stderr_correlation_names_and_bounds():
    values = np.array([1.5, -0.5])
    residuals = np.array([1.0, -2.0, 0.5])
    jacobian = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    bounds = ((0.0, 2.0), (None, 1.0))

    diagnostics = compute_parameter_diagnostics(
        values,
        names=("thickness", "offset"),
        bounds=bounds,
        residuals=residuals,
        jacobian=jacobian,
    )

    residual_variance = float(residuals @ residuals)
    expected_covariance = residual_variance * np.linalg.pinv(jacobian.T @ jacobian)
    expected_stderr = np.sqrt(np.diag(expected_covariance))
    expected_correlation = expected_covariance / np.outer(
        expected_stderr, expected_stderr
    )
    np.testing.assert_allclose(diagnostics.covariance, expected_covariance)
    np.testing.assert_allclose(diagnostics.stderr, expected_stderr)
    np.testing.assert_allclose(diagnostics.correlation, expected_correlation)
    assert diagnostics.names == ("thickness", "offset")
    assert diagnostics.bounds == bounds
    assert diagnostics.dof == 1
    assert diagnostics.residual_variance == residual_variance
    assert np.isfinite(diagnostics.condition_number)


def test_correlation_detects_strongly_coupled_parameters():
    jacobian = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.001],
            [3.0, 3.002],
        ]
    )
    diagnostics = compute_parameter_diagnostics(
        [0.0, 0.0],
        residuals=[0.1, -0.1, 0.05],
        jacobian=jacobian,
    )

    assert abs(diagnostics.correlation[0, 1]) > 0.99
    assert diagnostics.condition_number > 1.0e3


def test_rank_deficient_jacobian_is_supported_and_reports_infinite_condition():
    diagnostics = compute_parameter_diagnostics(
        [1.0, 1.0],
        residuals=[0.1, -0.2, 0.1],
        jacobian=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
    )

    assert np.all(np.isfinite(diagnostics.covariance))
    assert np.all(np.isfinite(diagnostics.stderr))
    assert np.isinf(diagnostics.condition_number)
    assert diagnostics.singular_values[-1] < 1.0e-12


def test_supplied_covariance_handles_zero_uncertainty_correlation():
    diagnostics = compute_parameter_diagnostics(
        [1.0, 2.0],
        covariance=np.array([[0.25, 0.0], [0.0, 0.0]]),
    )

    np.testing.assert_allclose(diagnostics.stderr, [0.5, 0.0])
    assert diagnostics.correlation[0, 0] == 1.0
    assert diagnostics.correlation[1, 1] == 1.0
    assert np.isnan(diagnostics.correlation[0, 1])
    assert diagnostics.names == ("p0", "p1")
    assert diagnostics.bounds is None
    assert diagnostics.singular_values.size == 0
    assert np.isnan(diagnostics.condition_number)
    assert diagnostics.dof == 0
    assert np.isnan(diagnostics.residual_variance)


def test_diagnostics_from_least_squares_result_uses_declared_parameters():
    residuals = np.array([0.1, -0.2, 0.05])
    jacobian = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
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
        covariance=np.eye(2),
        raw_result=object(),
        total_seconds=0.1,
    )
    parameters = (
        FitParameter("a", 0.0, 2.0),
        FitParameter("b", -1.0, 1.0),
    )

    diagnostics = diagnostics_from_least_squares_result(result, parameters)

    assert diagnostics.names == ("a", "b")
    assert diagnostics.bounds == ((0.0, 2.0), (-1.0, 1.0))
    np.testing.assert_allclose(diagnostics.values, [1.25, -0.25])
    expected_covariance = float(residuals @ residuals) * np.linalg.pinv(
        jacobian.T @ jacobian
    )
    np.testing.assert_allclose(diagnostics.covariance, expected_covariance)
    assert not np.allclose(diagnostics.covariance, result.covariance)


def test_diagnostics_requires_residuals_and_jacobian_without_covariance():
    with pytest.raises(ValueError, match="residuals and jacobian are required"):
        compute_parameter_diagnostics([1.0])


def test_plotting_functions_return_matplotlib_figure_and_axes():
    plt = pytest.importorskip("matplotlib.pyplot")
    diagnostics = compute_parameter_diagnostics(
        [1.0, 2.0],
        names=("a", "b"),
        bounds=((0.0, 3.0), (1.0, 4.0)),
        residuals=[0.1, -0.2, 0.05],
        jacobian=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
    )

    figures = []
    for plotter in (
        plot_parameter_estimates,
        plot_correlation_matrix,
        plot_singular_values,
    ):
        fig, ax = plotter(diagnostics)
        figures.append(fig)
        assert ax.figure is fig
    for fig in figures:
        plt.close(fig)
