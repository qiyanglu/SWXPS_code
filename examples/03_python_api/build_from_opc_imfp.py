"""Quickstart: build the synthetic C/LaNiO3/SrTiO3 workflow from OPC and IMFP files."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from examples.synthetic_case import (  # noqa: E402
    PHOTON_ENERGY_EV,
    RC_EDGE_FRACTION,
    RC_NORMALIZATION,
    RC_POLYNOMIAL_ORDER,
    build_stack,
    core_level_requests,
    load_synthetic_data,
)
from swanx.io import load_material_tables  # noqa: E402
from swanx.workflows import (  # noqa: E402
    ReflectivityRequest,
    RockingCurveRequest,
    simulate_reflectivity,
    simulate_rocking_curves,
)


DATA_DIR = REPO_ROOT / "data"


def main() -> None:
    tables = load_material_tables(
        opc_files={
            "C": DATA_DIR / "OPC" / "C.dat",
            "LNO": DATA_DIR / "OPC" / "LaNiO3.dat",
            "STO": DATA_DIR / "OPC" / "SrTiO3.dat",
        },
        imfp_files={
            "C": DATA_DIR / "IMFP" / "C.ANG",
            "LNO": DATA_DIR / "IMFP" / "LNO.ANG",
            "STO": DATA_DIR / "IMFP" / "STO.ANG",
        },
    )
    data = load_synthetic_data(stride=4)
    angles = data["angle_deg"]
    stack = build_stack()

    reflectivity = simulate_reflectivity(
        ReflectivityRequest(
            angles=angles,
            energy_ev=PHOTON_ENERGY_EV,
            stack=stack,
            roughness_step=1.0,
            slicing=None,
        )
    )
    rocking_curves = simulate_rocking_curves(
        RockingCurveRequest(
            angles=angles,
            photon_energy_ev=PHOTON_ENERGY_EV,
            stack=stack,
            core_levels=core_level_requests(),
            normalization_mode=RC_NORMALIZATION,
            normalization_edge_fraction=RC_EDGE_FRACTION,
            normalization_polynomial_order=RC_POLYNOMIAL_ORDER,
            field_step=1.0,
            roughness_step=1.0,
            slicing=None,
        )
    )

    print(f"materials loaded: {', '.join(sorted(tables.optical_constants))}")
    print(f"stack layers: {len(stack.optical_layers)}")
    print(f"reflectivity points: {reflectivity.reflectivity.size}")
    print(f"R range: {reflectivity.reflectivity.min():.3e} to {reflectivity.reflectivity.max():.3e}")
    for core in rocking_curves.core_levels:
        print(
            f"{core.name}: kinetic energy {core.kinetic_energy_ev:.1f} eV, "
            f"normalized RC mean {core.curve.intensity.mean():.3f}"
        )


if __name__ == "__main__":
    main()
