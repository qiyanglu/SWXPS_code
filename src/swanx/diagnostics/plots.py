"""Parameter uncertainty, correlation, and identifiability plots."""

from __future__ import annotations

import numpy as np

from .covariance import ParameterDiagnostics
def plot_parameter_estimates(
    diagnostics: ParameterDiagnostics,
    ci: float | None = 0.95,
    ax=None,
    show_bounds: bool = True,
    normalization: str | None = "bounds",
    show_bound_labels: bool = True,
    show_value_labels: bool = False,
    value_format: str = ".4g",
):
    """Plot parameter estimates and intervals on a comparable scale.

    ``normalization="bounds"`` maps every finite parameter range to ``[0, 1]``
    and scales its uncertainty by that range. Use ``normalization=None`` for
    the former raw-value view. Bound and value labels always use raw values.
    """

    plt = _load_pyplot()
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(11.5, max(3.5, 0.64 * len(diagnostics.names) + 1.8))
        )
    else:
        fig = ax.figure
    multiplier = _ci_multiplier(ci)
    y = np.arange(len(diagnostics.names))
    plot_values = np.asarray(diagnostics.values, dtype=float)
    plot_stderr = np.asarray(diagnostics.stderr, dtype=float)
    plot_bounds = diagnostics.bounds

    if normalization == "bounds":
        if diagnostics.bounds is None:
            raise ValueError(
                "bounds normalization requires finite lower and upper bounds"
            )
        numeric_bounds = []
        for lower, upper in diagnostics.bounds:
            if (
                lower is None
                or upper is None
                or not np.isfinite(lower)
                or not np.isfinite(upper)
                or upper <= lower
            ):
                raise ValueError(
                    "bounds normalization requires finite lower < upper for "
                    "every parameter"
                )
            numeric_bounds.append((float(lower), float(upper)))
        lower_values = np.asarray([item[0] for item in numeric_bounds])
        upper_values = np.asarray([item[1] for item in numeric_bounds])
        ranges = upper_values - lower_values
        plot_values = (plot_values - lower_values) / ranges
        plot_stderr = plot_stderr / ranges
        plot_bounds = tuple((0.0, 1.0) for _ in diagnostics.names)
        ax.set_xlabel("Position within parameter bounds")
    elif normalization is None:
        ax.set_xlabel("Parameter value")
    else:
        raise ValueError("normalization must be 'bounds' or None")

    if show_bounds and plot_bounds is not None:
        for index, (plot_bound, raw_bound) in enumerate(
            zip(plot_bounds, diagnostics.bounds or plot_bounds)
        ):
            lower, upper = plot_bound
            if lower is not None and upper is not None:
                ax.hlines(
                    index,
                    lower,
                    upper,
                    color="tab:blue",
                    alpha=0.22,
                    linewidth=9.0,
                    zorder=1,
                )
                if show_bound_labels:
                    raw_lower, raw_upper = raw_bound
                    ax.annotate(
                        format(raw_lower, value_format),
                        (lower, index),
                        xytext=(-7, 0),
                        textcoords="offset points",
                        ha="right",
                        va="center",
                        color="tab:blue",
                        fontsize=10,
                    )
                    ax.annotate(
                        format(raw_upper, value_format),
                        (upper, index),
                        xytext=(7, 0),
                        textcoords="offset points",
                        ha="left",
                        va="center",
                        color="tab:blue",
                        fontsize=10,
                    )
            else:
                for bound in (lower, upper):
                    if bound is not None:
                        ax.plot(
                            bound,
                            index,
                            marker="|",
                            color="tab:blue",
                            markersize=12,
                            zorder=1,
                        )

    if multiplier is None:
        ax.plot(plot_values, y, "o", color="black", label="best fit", zorder=3)
    else:
        errors = multiplier * plot_stderr
        errors = np.where(np.isfinite(errors), errors, 0.0)
        ax.errorbar(
            plot_values,
            y,
            xerr=errors,
            fmt="o",
            color="black",
            ecolor="tab:orange",
            capsize=4,
            markersize=6,
            elinewidth=1.5,
            label=f"{int(round(100 * ci))}% CI",
            zorder=3,
        )

    if show_value_labels:
        for index, (coordinate, raw_value) in enumerate(
            zip(plot_values, diagnostics.values)
        ):
            ax.annotate(
                format(raw_value, value_format),
                (coordinate, index),
                xytext=(6, -9),
                textcoords="offset points",
                ha="left",
                va="top",
                color="black",
                fontsize=10,
            )

    ax.set_yticks(y, labels=diagnostics.names)
    ax.tick_params(axis="both", labelsize=11)
    ax.xaxis.label.set_size(11)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0.0,
        fontsize=10,
    )
    fig.tight_layout()
    return fig, ax


def plot_correlation_matrix(
    diagnostics: ParameterDiagnostics,
    ax=None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    annotate: bool = True,
):
    """Plot the parameter correlation matrix without a seaborn dependency."""

    plt = _load_pyplot()
    if ax is None:
        size = max(4.0, 0.55 * len(diagnostics.names) + 2.0)
        fig, ax = plt.subplots(figsize=(size, size))
    else:
        fig = ax.figure
    image = ax.imshow(
        diagnostics.correlation,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
    )
    indices = np.arange(len(diagnostics.names))
    ax.set_xticks(indices, labels=diagnostics.names, rotation=45, ha="right")
    ax.set_yticks(indices, labels=diagnostics.names)
    if annotate:
        midpoint = 0.5 * (vmin + vmax)
        for row in indices:
            for column in indices:
                value = diagnostics.correlation[row, column]
                if np.isfinite(value):
                    color = (
                        "white"
                        if abs(value - midpoint) > 0.25 * (vmax - vmin)
                        else "black"
                    )
                    ax.text(
                        column,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=10,
                    )
    ax.set_title("Parameter correlation")
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    return fig, ax


def plot_singular_values(diagnostics: ParameterDiagnostics, ax=None):
    """Plot available Jacobian singular values on a logarithmic scale."""

    if diagnostics.singular_values.size == 0:
        raise ValueError("Jacobian singular values are not available")
    plt = _load_pyplot()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
    else:
        fig = ax.figure
    indices = np.arange(1, diagnostics.singular_values.size + 1)
    plotted_values = np.where(diagnostics.singular_values > 0, diagnostics.singular_values, np.nan)
    ax.plot(indices, plotted_values, "o-", color="tab:blue")
    ax.set_yscale("log")
    ax.set_xlabel("Singular-value index")
    ax.set_ylabel("Singular value")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    return fig, ax



def _ci_multiplier(ci: float | None) -> float | None:
    if ci is None:
        return None
    if np.isclose(ci, 0.68):
        return 1.0
    if np.isclose(ci, 0.95):
        return 1.96
    raise ValueError("ci must be None, 0.68, or 0.95")



def _load_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("matplotlib is required for parameter diagnostic plots") from error
    return plt

__all__ = [
    "plot_correlation_matrix",
    "plot_parameter_estimates",
    "plot_singular_values",
]