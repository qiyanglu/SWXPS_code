import numpy as np
import pytest

from swanx.diagnostics import (
    compute_parameter_diagnostics,
    plot_parameter_estimates,
)


def _diagnostics_with_different_parameter_scales():
    return compute_parameter_diagnostics(
        [20.0, 0.25, -0.1],
        names=("thickness", "fraction", "offset"),
        bounds=((10.0, 30.0), (0.0, 1.0), (-0.5, 0.5)),
        covariance=np.diag([4.0, 0.01, 0.0025]),
    )


def _estimate_line(ax):
    return next(line for line in ax.lines if line.get_marker() == "o")


def test_parameter_plot_normalizes_values_by_bounds_and_labels_raw_endpoints():
    plt = pytest.importorskip("matplotlib.pyplot")
    diagnostics = _diagnostics_with_different_parameter_scales()

    fig, ax = plot_parameter_estimates(diagnostics, ci=None)

    np.testing.assert_allclose(_estimate_line(ax).get_xdata(), [0.5, 0.25, 0.4])
    assert ax.get_xlabel() == "Position within parameter bounds"
    assert {text.get_text() for text in ax.texts} == {
        "10",
        "30",
        "0",
        "1",
        "-0.5",
        "0.5",
    }
    plt.close(fig)


def test_parameter_plot_can_show_raw_values_without_bound_labels():
    plt = pytest.importorskip("matplotlib.pyplot")
    diagnostics = _diagnostics_with_different_parameter_scales()

    fig, ax = plot_parameter_estimates(
        diagnostics,
        ci=None,
        normalization=None,
        show_bound_labels=False,
    )

    np.testing.assert_allclose(_estimate_line(ax).get_xdata(), diagnostics.values)
    assert ax.get_xlabel() == "Parameter value"
    assert not ax.texts
    plt.close(fig)


def test_parameter_plot_requires_complete_bounds_for_normalization():
    plt = pytest.importorskip("matplotlib.pyplot")
    diagnostics = compute_parameter_diagnostics(
        [1.0],
        bounds=((None, 2.0),),
        covariance=np.eye(1),
    )

    with pytest.raises(ValueError, match="finite lower < upper"):
        plot_parameter_estimates(diagnostics)
    plt.close("all")


def test_parameter_plot_rejects_unknown_normalization():
    plt = pytest.importorskip("matplotlib.pyplot")
    diagnostics = _diagnostics_with_different_parameter_scales()

    with pytest.raises(ValueError, match="normalization"):
        plot_parameter_estimates(diagnostics, normalization="mean")
    plt.close("all")
