import numpy as np
import pytest

from swanx.diagnostics import compute_parameter_diagnostics, plot_parameter_estimates


def test_parameter_plot_legend_is_above_axes_and_bound_bars_are_wider():
    plt = pytest.importorskip("matplotlib.pyplot")
    diagnostics = compute_parameter_diagnostics(
        [0.4, 15.0],
        names=("fraction", "thickness"),
        bounds=((0.0, 1.0), (5.0, 25.0)),
        covariance=np.diag([0.01, 1.0]),
    )

    fig, ax = plot_parameter_estimates(diagnostics)

    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_bbox_to_anchor()._bbox.y0 > 1.0
    assert all(width == pytest.approx(9.0) for width in ax.collections[0].get_linewidths())
    assert all(label.get_fontsize() == pytest.approx(11.0) for label in ax.get_yticklabels())
    plt.close(fig)
