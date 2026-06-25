from pathlib import Path

import numpy as np
import pytest

from swanx.io import OpticalConstantTable, read_optical_constants


def test_cxro_reader_parses_material_density_and_comma_rows(tmp_path):
    path = tmp_path / "LaNiO3.dat"
    path.write_text(
        "LaNiO3 Density=7.25\n"
        "Energy(eV), Delta, Beta\n"
        "100, 0.10, 0.01\n"
        "200, 0.20, 0.02\n",
        encoding="utf-8",
    )

    table = read_optical_constants(path)

    assert isinstance(table, OpticalConstantTable)
    assert table.material == "LaNiO3"
    assert table.density == 7.25
    assert table.at_energy(150.0) == pytest.approx((0.15, 0.015))


def test_cxro_reader_supports_whitespace_rows_and_sorts(tmp_path):
    path = tmp_path / "material.dat"
    path.write_text(
        "Material Density=1.0\n"
        "Energy(eV), Delta, Beta\n"
        "200 0.20 0.02\n"
        "100 0.10 0.01\n",
        encoding="utf-8",
    )

    table = read_optical_constants(path)

    np.testing.assert_allclose(table.energy_ev, [100.0, 200.0])
    assert table.at_energy(150.0) == pytest.approx((0.15, 0.015))


def test_cxro_reader_rejects_out_of_range_duplicate_invalid_and_nonpositive(tmp_path):
    valid = tmp_path / "valid.dat"
    valid.write_text(
        "M Density=1\nEnergy(eV), Delta, Beta\n100 0.1 0.01\n200 0.2 0.02\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside"):
        read_optical_constants(valid).at_energy(50.0)

    duplicate = tmp_path / "duplicate.dat"
    duplicate.write_text(
        "M Density=1\nEnergy(eV), Delta, Beta\n100 0.1 0.01\n100 0.2 0.02\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        read_optical_constants(duplicate)

    invalid = tmp_path / "invalid.dat"
    invalid.write_text(
        "M Density=1\nEnergy(eV), Delta, Beta\n100 nan 0.01\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="NaN|infinite"):
        read_optical_constants(invalid)

    nonpositive = tmp_path / "nonpositive.dat"
    nonpositive.write_text(
        "M Density=1\nEnergy(eV), Delta, Beta\n0 0.1 0.01\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="positive"):
        read_optical_constants(nonpositive)

    negative_beta = tmp_path / "negative_beta.dat"
    negative_beta.write_text(
        "M Density=1\nEnergy(eV), Delta, Beta\n100 0.1 -0.01\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="beta.*non-negative"):
        read_optical_constants(negative_beta)

    negative_delta = tmp_path / "negative_delta.dat"
    negative_delta.write_text(
        "M Density=1\nEnergy(eV), Delta, Beta\n100 -0.1 0.01\n200 -0.2 0.02\n",
        encoding="utf-8",
    )
    table = read_optical_constants(negative_delta)
    assert table.at_energy(150.0) == pytest.approx((-0.15, 0.015))


def test_table_reader_uses_header_or_explicit_columns_for_beta_delta_order(tmp_path):
    header = tmp_path / "header.csv"
    header.write_text(
        "energy_ev,beta,delta\n100,0.01,0.10\n200,0.02,0.20\n",
        encoding="utf-8",
    )
    table = read_optical_constants(header, format="table")
    assert table.at_energy(150.0) == pytest.approx((0.15, 0.015))

    explicit = tmp_path / "explicit.txt"
    explicit.write_text("100 0.01 0.10\n200 0.02 0.20\n", encoding="utf-8")
    table = read_optical_constants(
        explicit,
        format="table",
        energy_column=0,
        delta_column=2,
        beta_column=1,
    )
    assert table.at_energy(150.0) == pytest.approx((0.15, 0.015))
