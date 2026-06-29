"""Load tutorial experimental curves into SWANX fitting data objects."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swanx.fitting import FittingProblem
from swanx.io import (
    core_level_from_tables,
    load_material_tables,
    read_reflectivity_data,
    read_rocking_curve_data,
    stack_from_layer_specs,
)


DATA_DIR = REPO_ROOT / "data"


def main() -> None:
    reflectivity = read_reflectivity_data(
        DATA_DIR / "curves" / "lno_sto_reflectivity.csv",
        name="tutorial reflectivity",
    )
    la4d_raw = read_rocking_curve_data(
        DATA_DIR / "curves" / "la4d_rocking_curve.csv",
        name="La 4d",
    )
    la4d_mean = read_rocking_curve_data(
        DATA_DIR / "curves" / "la4d_rocking_curve.csv",
        name="La 4d",
        normalization_mode="mean",
    )

    print(f"reflectivity shape: {reflectivity.angles.shape}")
    print(f"first reflectivity point: {reflectivity.angles[0]:.1f} deg, {reflectivity.reflectivity[0]:.3g}")
    print(f"raw La 4d shape: {la4d_raw.angles.shape}")
    print(f"mean-normalized La 4d mean: {la4d_mean.intensity.mean():.3f}")

    energy_ev = 900.0
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
    la4d_request = core_level_from_tables(
        name="La 4d",
        binding_energy_ev=105.0,
        photon_energy_ev=energy_ev,
        concentration_by_material={"LNO": 1.0},
        imfp_tables=tables.imfp,
    )
    problem = FittingProblem(
        parameters=(),
        stack_builder=lambda values: stack,
        photon_energy_ev=energy_ev,
        reflectivity=reflectivity,
        rocking_curves=(la4d_mean,),
        core_levels=(la4d_request,),
    )
    print(
        "FittingProblem datasets: "
        f"reflectivity={problem.reflectivity.name}, "
        f"rocking_curves={len(problem.rocking_curves)}"
    )


if __name__ == "__main__":
    main()
