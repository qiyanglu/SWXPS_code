from pathlib import Path

import pytest

from swanx.optical_constants import (
    constants_from_file,
    layer_from_file,
    load_optical_constants,
)


def test_load_optical_constants_parses_header_and_rows():
    table = load_optical_constants(Path("examples/data/OPC") / "LaNiO3.dat")

    assert table.material == "LaNiO3"
    assert table.density == 7.25
    assert table.energy_ev[0] == 30.0
    assert table.delta[0] == 0.310286105
    assert table.beta[0] == 0.399127007


def test_constants_from_file_returns_exact_tabulated_values():
    delta, beta = constants_from_file(Path("examples/data/OPC") / "LaNiO3.dat", energy_ev=5000.00049)

    assert delta == pytest.approx(4.89944032e-05)
    assert beta == pytest.approx(2.89918785e-06)


def test_constants_from_file_linearly_interpolates_between_rows():
    table = load_optical_constants(Path("examples/data/OPC") / "LaNiO3.dat")
    energy = 0.5 * (table.energy_ev[0] + table.energy_ev[1])

    delta, beta = table.constants_at(energy)

    assert delta == pytest.approx(0.5 * (table.delta[0] + table.delta[1]))
    assert beta == pytest.approx(0.5 * (table.beta[0] + table.beta[1]))


def test_constants_from_file_rejects_out_of_range_energy():
    table = load_optical_constants(Path("examples/data/OPC") / "LaNiO3.dat")

    with pytest.raises(ValueError, match="outside the table range"):
        table.constants_at(8000.0)


def test_layer_from_file_uses_interpolated_constants():
    layer = layer_from_file(
        Path("examples/data/OPC") / "SrTiO3.dat",
        energy_ev=5000.00049,
        thickness=12.0,
        roughness=3.0,
    )

    assert layer.thickness == 12.0
    assert layer.roughness == 3.0
    assert layer.delta > 0.0
    assert layer.beta > 0.0
