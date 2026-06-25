import numpy as np
import pytest

from swanx.fitting import ReflectivityData, RockingCurveData
from swanx.io import read_reflectivity_data, read_rocking_curve_data
from swanx.preprocessing import normalize_rocking_curve


def test_read_reflectivity_csv_with_header(tmp_path):
    path = tmp_path / "reflectivity.csv"
    path.write_text(
        "angle_deg,reflectivity,sigma\n5.1,0.012,0.2\n5.0,0.010,0.1\n",
        encoding="utf-8",
    )

    data = read_reflectivity_data(path, sigma_column="sigma")

    assert isinstance(data, ReflectivityData)
    assert data.name == "reflectivity"
    np.testing.assert_allclose(data.angles, [5.0, 5.1])
    np.testing.assert_allclose(data.reflectivity, [0.010, 0.012])
    np.testing.assert_allclose(data.sigma, [0.1, 0.2])


def test_read_reflectivity_whitespace_with_header_and_common_intensity_name(tmp_path):
    path = tmp_path / "reflectivity.txt"
    path.write_text("angle_deg R\n5.0 0.010\n5.1 0.012\n", encoding="utf-8")

    data = read_reflectivity_data(path)

    np.testing.assert_allclose(data.angles, [5.0, 5.1])
    np.testing.assert_allclose(data.reflectivity, [0.010, 0.012])
    assert data.sigma is None


def test_read_reflectivity_headerless_with_explicit_column_indices(tmp_path):
    path = tmp_path / "reflectivity.dat"
    path.write_text("7 5.0 0.010\n7 5.1 0.012\n", encoding="utf-8")

    data = read_reflectivity_data(path, angle_column=1, intensity_column=2)

    np.testing.assert_allclose(data.angles, [5.0, 5.1])
    np.testing.assert_allclose(data.reflectivity, [0.010, 0.012])


def test_read_rocking_curve_csv_with_header(tmp_path):
    path = tmp_path / "la4d.csv"
    path.write_text("angle_deg,intensity\n5.0,1.0\n5.1,1.03\n", encoding="utf-8")

    data = read_rocking_curve_data(path)

    assert isinstance(data, RockingCurveData)
    np.testing.assert_allclose(data.angles, [5.0, 5.1])
    np.testing.assert_allclose(data.intensity, [1.0, 1.03])


def test_read_rocking_curve_whitespace_with_header_and_common_intensity_name(tmp_path):
    path = tmp_path / "la4d.txt"
    path.write_text("angle_deg counts\n5.0 100\n5.1 103\n", encoding="utf-8")

    data = read_rocking_curve_data(path)

    np.testing.assert_allclose(data.intensity, [100.0, 103.0])


def test_read_rocking_curve_headerless_with_explicit_column_indices(tmp_path):
    path = tmp_path / "la4d.dat"
    path.write_text("0 5.0 1.00\n0 5.1 1.03\n", encoding="utf-8")

    data = read_rocking_curve_data(path, angle_column=1, intensity_column=2)

    np.testing.assert_allclose(data.angles, [5.0, 5.1])
    np.testing.assert_allclose(data.intensity, [1.0, 1.03])


def test_curve_loader_sorts_by_angle(tmp_path):
    path = tmp_path / "curve.csv"
    path.write_text(
        "angle_deg,intensity\n5.2,3\n5.0,1\n5.1,2\n",
        encoding="utf-8",
    )

    data = read_rocking_curve_data(path)

    np.testing.assert_allclose(data.angles, [5.0, 5.1, 5.2])
    np.testing.assert_allclose(data.intensity, [1.0, 2.0, 3.0])


def test_duplicate_angles_raise_value_error(tmp_path):
    path = tmp_path / "duplicate.csv"
    path.write_text("angle_deg,intensity\n5.0,1\n5.0,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate angle"):
        read_rocking_curve_data(path)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("angle_deg,intensity\nnan,1\n", "angle"),
        ("angle_deg,intensity\n5,inf\n", "intensity"),
        ("angle_deg,intensity,sigma\n5,1,nan\n", "sigma"),
    ],
)
def test_nan_or_inf_raise_value_error(tmp_path, contents, message):
    path = tmp_path / "invalid.csv"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_rocking_curve_data(path, sigma_column="sigma" if "sigma" in contents else None)


def test_negative_sigma_raises_value_error(tmp_path):
    path = tmp_path / "negative_sigma.csv"
    path.write_text("angle_deg,intensity,sigma\n5,1,-0.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="negative sigma"):
        read_rocking_curve_data(path, sigma_column="sigma")


def test_negative_intensity_is_allowed(tmp_path):
    path = tmp_path / "background_subtracted.csv"
    path.write_text("angle_deg,intensity\n5,-0.1\n5.1,0.1\n", encoding="utf-8")

    data = read_rocking_curve_data(path)

    np.testing.assert_allclose(data.intensity, [-0.1, 0.1])


def test_malformed_row_inside_data_block_raises_value_error(tmp_path):
    path = tmp_path / "malformed.csv"
    path.write_text("angle_deg,intensity\n5.0,1.0\nbroken,row\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"malformed curve row .* line 3"):
        read_rocking_curve_data(path)


def test_normalization_none_preserves_raw_intensity(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text("angle_deg,intensity\n5.0,2\n5.1,4\n", encoding="utf-8")

    data = read_rocking_curve_data(path, normalization_mode=None)

    np.testing.assert_allclose(data.intensity, [2.0, 4.0])


def test_mean_normalization_reuses_preprocessing_behavior(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text("angle_deg,intensity\n5.0,2\n5.1,4\n", encoding="utf-8")

    data = read_rocking_curve_data(path, normalization_mode="mean")
    expected, _ = normalize_rocking_curve(
        np.asarray([5.0, 5.1]),
        np.asarray([2.0, 4.0]),
        mode="mean",
    )

    np.testing.assert_allclose(data.intensity, expected)


def test_io_namespace_does_not_export_preprocessing_functions():
    import swanx.io as sxio

    assert not hasattr(sxio, "normalize_rocking_curve")
    assert "normalize_rocking_curve" not in sxio.__all__
