"""Quickstart: build a SW-XPS workflow from OPC and IMFP files."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

import swanx as sx
from swanx.io import (
    core_level_from_tables,
    load_material_tables,
    stack_from_layer_specs,
)


DATA_DIR = REPO_ROOT / "data"


def main() -> None:
    energy_ev = 900.0
    angles = np.linspace(5.0, 15.0, 201)

    tables = load_material_tables(
        opc_files={
            "LNO": DATA_DIR / "OPC" / "LaNiO3.dat",
            "STO": DATA_DIR / "OPC" / "SrTiO3.dat",
        },
        imfp_files={
            "LNO": DATA_DIR / "IMFP" / "LNO.ANG",
            "STO": DATA_DIR / "IMFP" / "STO.ANG",
        },
    )

    stack = stack_from_layer_specs(
        [
            {"material": "vacuum", "thickness": 0.0},
            {"material": "LNO", "thickness": 40.0, "roughness": 3.0},
            {"material": "STO", "thickness": 0.0},
        ],
        optical_constants=tables.optical_constants,
        energy_ev=energy_ev,
    )

    reflectivity = sx.simulate_reflectivity(
        sx.ReflectivityRequest(
            angles=angles,
            energy_ev=energy_ev,
            stack=stack,
        )
    )

    la4d = core_level_from_tables(
        name="La 4d",
        binding_energy_ev=105.0,
        photon_energy_ev=energy_ev,
        concentration_by_material={"LNO": 1.0},
        imfp_tables=tables.imfp,
    )
    rocking_curves = sx.simulate_rocking_curves(
        sx.RockingCurveRequest(
            angles=angles,
            photon_energy_ev=energy_ev,
            stack=stack,
            core_levels=(la4d,),
        )
    )

    print(f"reflectivity points: {reflectivity.reflectivity.size}")
    print(f"R range: {reflectivity.reflectivity.min():.3e} to {reflectivity.reflectivity.max():.3e}")
    for core in rocking_curves.core_levels:
        print(
            f"{core.name}: kinetic energy {core.kinetic_energy_ev:.1f} eV, "
            f"normalized RC mean {core.curve.intensity.mean():.3f}"
        )


if __name__ == "__main__":
    main()
