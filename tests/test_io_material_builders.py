import numpy as np
import pytest

from swanx.io import (
    IMFPTable,
    OpticalConstantTable,
    core_level_from_tables,
    core_levels_from_specs,
    load_material_tables,
    stack_from_layer_specs,
)


def _write_opc(path, delta):
    path.write_text(
        "M Density=1\n"
        "Energy(eV), Delta, Beta\n"
        f"100 {delta} 0.01\n"
        f"200 {2 * delta} 0.02\n",
        encoding="utf-8",
    )


def _write_imfp(path, first=5.0, second=9.0):
    path.write_text(
        "energy imfp\n"
        f"100 {first}\n"
        f"200 {second}\n",
        encoding="utf-8",
    )


def test_load_material_tables_explicit_and_directory_modes(tmp_path):
    opc = tmp_path / "OPC"
    imfp = tmp_path / "IMFP"
    opc.mkdir()
    imfp.mkdir()
    _write_opc(opc / "LNO.dat", 0.1)
    _write_opc(opc / "SrTiO3.dat", 0.2)
    _write_imfp(imfp / "LNO.ANG")
    _write_imfp(imfp / "STO.ANG")

    explicit = load_material_tables(
        opc_files={"LNO": opc / "LNO.dat", "STO": opc / "SrTiO3.dat"},
        imfp_files={"LNO": imfp / "LNO.ANG", "STO": imfp / "STO.ANG"},
    )
    assert set(explicit.optical_constants) == {"LNO", "STO"}
    assert set(explicit.imfp) == {"LNO", "STO"}

    directory = load_material_tables(
        opc_dir=opc,
        imfp_dir=imfp,
        materials=["LNO"],
    )
    assert set(directory.optical_constants) == {"LNO"}
    assert set(directory.imfp) == {"LNO"}


def test_load_material_tables_missing_files_are_clear(tmp_path):
    with pytest.raises(FileNotFoundError, match="OPC.*LNO"):
        load_material_tables(opc_dir=tmp_path, materials=["LNO"])
    with pytest.raises(FileNotFoundError, match="IMFP.*LNO"):
        load_material_tables(imfp_dir=tmp_path, materials=["LNO"])


def test_stack_from_layer_specs_uses_opc_vacuum_and_explicit_override():
    table = OpticalConstantTable(
        energy_ev=np.array([100.0, 200.0]),
        delta=np.array([0.1, 0.2]),
        beta=np.array([0.01, 0.02]),
    )

    stack = stack_from_layer_specs(
        [
            {"material": "vacuum", "thickness": 0.0},
            {"material": "LNO", "thickness": 40.0, "roughness": 3.0},
            {"material": "STO", "thickness": 0.0, "delta": 9.0, "beta": 8.0},
        ],
        optical_constants={"LNO": table},
        energy_ev=150.0,
    )

    assert stack.layers[0].delta == 0.0
    assert stack.layers[1].delta == pytest.approx(0.15)
    assert stack.layers[1].beta == pytest.approx(0.015)
    assert stack.layers[2].delta == 9.0
    assert stack.layers[2].beta == 8.0


def test_stack_from_layer_specs_validates_boundaries():
    table = OpticalConstantTable(
        energy_ev=np.array([100.0, 200.0]),
        delta=np.array([0.1, 0.2]),
        beta=np.array([0.01, 0.02]),
    )
    with pytest.raises(ValueError, match="first layer"):
        stack_from_layer_specs(
            [{"material": "LNO", "thickness": 1.0}, {"material": "STO", "thickness": 0.0}],
            optical_constants={"LNO": table, "STO": table},
            energy_ev=150.0,
        )
    with pytest.raises(ValueError, match="thickness=0.0"):
        stack_from_layer_specs(
            [{"material": "vacuum", "thickness": 0.0}, {"material": "STO", "thickness": 1.0}],
            optical_constants={"STO": table},
            energy_ev=150.0,
        )


def test_core_level_helpers_interpolate_imfp_and_use_binding_energy():
    imfp_tables = {
        "LNO": IMFPTable(
            kinetic_energy_ev=np.array([700.0, 800.0]),
            imfp_angstrom=np.array([7.0, 9.0]),
        )
    }

    core = core_level_from_tables(
        name="La 4d",
        binding_energy_ev=150.0,
        photon_energy_ev=900.0,
        concentration_by_material={"LNO": 1.0},
        imfp_tables=imfp_tables,
    )

    assert core.imfp_by_material["LNO"] == pytest.approx(8.0)
    assert np.isinf(core.imfp_by_material["vacuum"])

    cores = core_levels_from_specs(
        [
            {"name": "A", "binding_energy_ev": 100.0, "concentration_by_material": {"LNO": 1.0}},
            {"name": "B", "binding_energy_ev": 200.0, "concentration_by_material": {"LNO": 1.0}},
        ],
        photon_energy_ev=900.0,
        imfp_tables=imfp_tables,
    )
    assert cores[0].imfp_by_material["LNO"] != cores[1].imfp_by_material["LNO"]


def test_core_level_from_tables_rejects_nonpositive_kinetic_energy():
    with pytest.raises(ValueError, match="kinetic energy"):
        core_level_from_tables(
            name="bad",
            binding_energy_ev=1000.0,
            photon_energy_ev=900.0,
            concentration_by_material={},
            imfp_tables={},
        )
