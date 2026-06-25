from pathlib import Path

import numpy as np
import pytest

from swanx.io import IMFPTable, read_imfp


def test_read_imfp_supports_existing_ang_file():
    table = read_imfp(Path("data/IMFP") / "LNO.ANG")

    assert isinstance(table, IMFPTable)
    assert table.material == "LaNiO3"
    assert table.kinetic_energy_ev[0] == pytest.approx(50.0)
    assert table.imfp_angstrom[0] == pytest.approx(4.63)


def test_read_imfp_supports_csv_and_interpolation(tmp_path):
    path = tmp_path / "imfp.csv"
    path.write_text(
        "kinetic_energy_ev,imfp_angstrom\n100,5.0\n200,8.0\n300,10.5\n",
        encoding="utf-8",
    )

    table = read_imfp(path)

    assert table.at_kinetic_energy(150.0) == pytest.approx(6.5)


def test_read_imfp_supports_whitespace_and_headerless_explicit_columns(tmp_path):
    whitespace = tmp_path / "imfp.txt"
    whitespace.write_text(
        "energy imfp\n200 8.0\n100 5.0\n",
        encoding="utf-8",
    )
    table = read_imfp(whitespace)
    np.testing.assert_allclose(table.kinetic_energy_ev, [100.0, 200.0])

    headerless = tmp_path / "headerless.txt"
    headerless.write_text("100 9.9 5.0\n200 9.9 8.0\n", encoding="utf-8")
    table = read_imfp(headerless, energy_column=0, imfp_column=2)
    assert table.at_kinetic_energy(150.0) == pytest.approx(6.5)

    with_metadata = tmp_path / "headerless_with_metadata.txt"
    with_metadata.write_text(
        "material LNO\nunits eV Angstrom\n100 9.9 5.0\n200 9.9 8.0\n",
        encoding="utf-8",
    )
    table = read_imfp(with_metadata, energy_column=0, imfp_column=2)
    assert table.at_kinetic_energy(150.0) == pytest.approx(6.5)


def test_read_imfp_rejects_out_of_range_duplicate_invalid_and_nonpositive(tmp_path):
    valid = tmp_path / "valid.csv"
    valid.write_text("energy,imfp\n100,5\n200,8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        read_imfp(valid).at_kinetic_energy(50.0)

    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("energy,imfp\n100,5\n100,8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_imfp(duplicate)

    invalid = tmp_path / "invalid.csv"
    invalid.write_text("energy,imfp\n100,inf\n", encoding="utf-8")
    with pytest.raises(ValueError, match="NaN|infinite"):
        read_imfp(invalid)

    nonpositive_energy = tmp_path / "nonpositive_energy.csv"
    nonpositive_energy.write_text("energy,imfp\n0,5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="positive"):
        read_imfp(nonpositive_energy)

    nonpositive_imfp = tmp_path / "nonpositive_imfp.csv"
    nonpositive_imfp.write_text("energy,imfp\n100,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="positive"):
        read_imfp(nonpositive_imfp)


def test_read_imfp_raises_on_malformed_row_after_numeric_data_start(tmp_path):
    path = tmp_path / "malformed.csv"
    path.write_text(
        "kinetic_energy_ev,imfp_angstrom\n100,5.0\nnot,a-number\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"invalid IMFP row .* line 3"):
        read_imfp(path)
