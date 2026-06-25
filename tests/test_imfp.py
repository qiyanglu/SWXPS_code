from pathlib import Path

import pytest

from swanx.imfp import (
    imfp_from_file,
    load_imfp,
)


def test_load_imfp_parses_header_and_rows():
    table = load_imfp(Path("data/IMFP") / "LNO.ANG")

    assert table.material == "LaNiO3"
    assert table.kinetic_energy_ev[0] == 50.0
    assert table.imfp[0] == 4.63


def test_imfp_from_file_interpolates_between_rows():
    table = load_imfp(Path("data/IMFP") / "LNO.ANG")
    energy = 0.5 * (table.kinetic_energy_ev[0] + table.kinetic_energy_ev[1])

    assert table.imfp_at(energy) == pytest.approx(0.5 * (table.imfp[0] + table.imfp[1]))


def test_imfp_from_file_rejects_out_of_range_energy():
    table = load_imfp(Path("data/IMFP") / "LNO.ANG")

    with pytest.raises(ValueError, match="outside the table range"):
        table.imfp_at(1.0)


def test_imfp_from_file_returns_positive_value_for_la4d_case():
    imfp = imfp_from_file(Path("data/IMFP") / "LNO.ANG", kinetic_energy_ev=895.0)

    assert imfp > 0.0
