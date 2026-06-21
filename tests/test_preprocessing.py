import numpy as np

from swxps import (
    normalize_by_background,
    normalize_by_mean,
    normalize_rocking_curve,
    subtract_edge_polynomial_background,
)


def test_edge_polynomial_background_accepts_percent_or_fraction():
    x = np.linspace(-1.0, 1.0, 21)
    background = 2.0 + 0.3 * x + 0.2 * x**2
    signal = np.zeros_like(x)
    signal[8:13] = 1.0
    values = background + signal

    by_fraction = subtract_edge_polynomial_background(x, values, edge_fraction=0.10)
    by_percent = subtract_edge_polynomial_background(x, values, edge_fraction=10)

    np.testing.assert_allclose(by_fraction.background, background, atol=1e-12)
    np.testing.assert_allclose(by_percent.background, background, atol=1e-12)
    np.testing.assert_allclose(by_fraction.corrected, signal, atol=1e-12)
    np.testing.assert_allclose(by_fraction.normalized[:3], 1.0, atol=1e-12)
    np.testing.assert_allclose(by_fraction.normalized[-3:], 1.0, atol=1e-12)
    assert np.count_nonzero(by_fraction.edge_mask) == 6


def test_background_normalization_keeps_valleys_below_one():
    raw = np.array([10.0, 8.0, 10.0])
    background = np.array([10.0, 10.0, 10.0])

    np.testing.assert_allclose(normalize_by_background(raw, background), [1.0, 0.8, 1.0])


def test_edge_polynomial_background_rejects_order_without_enough_points():
    x = np.linspace(0.0, 1.0, 5)
    values = np.ones_like(x)

    with np.testing.assert_raises(ValueError):
        subtract_edge_polynomial_background(x, values, edge_fraction=0.20, order=2)


def test_normalize_by_mean():
    np.testing.assert_allclose(normalize_by_mean(np.array([1.0, 2.0, 3.0])), [0.5, 1.0, 1.5])


def test_edge_polynomial_rocking_curve_normalization_matches_background_helper():
    x = np.linspace(10.0, 15.0, 31)
    background = 4.0 + 0.2 * x + 0.03 * x**2
    values = background * (1.0 + 0.25 * np.exp(-((x - 12.5) / 0.35) ** 2))

    actual, normalization = normalize_rocking_curve(
        x,
        values,
        mode="edge_polynomial",
        edge_fraction=0.10,
        polynomial_order=2,
    )
    expected = subtract_edge_polynomial_background(
        x,
        values,
        edge_fraction=0.10,
        order=2,
    )

    np.testing.assert_allclose(normalization, expected.background)
    np.testing.assert_allclose(actual, expected.normalized)