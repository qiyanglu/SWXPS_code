"""Plot reflectivity from a LaNiO3/SrTiO3 multilayer mirror."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swanx.stack import (
    LayerTemplate,
    StackTemplate,
    SuperlatticeTemplate,
)
from swanx.optics import (
    energy_to_wavelength,
    transfer_matrix_reflectivity,
)


def make_lno_sto_superlattice(
    energy_ev: float,
    repeats: int,
    lno_thickness: float,
    sto_thickness: float,
) -> list:
    """Return vacuum / [LaNiO3 / SrTiO3]xN / SrTiO3 substrate."""

    template = StackTemplate(
        energy_ev=energy_ev,
        base_dir=REPO_ROOT,
        parts=(
            LayerTemplate.vacuum(),
            SuperlatticeTemplate(
                repeats=repeats,
                period=(
                    LayerTemplate.from_file("LNO", "data/OPC/LaNiO3.dat", lno_thickness, 3.0),
                    LayerTemplate.from_file("STO", "data/OPC/SrTiO3.dat", sto_thickness, 3.0),
                ),
            ),
            LayerTemplate.from_file("STO", "data/OPC/SrTiO3.dat", 0.0, 3.0),
        ),
    )
    return template.build().optical_layers


def main() -> None:
    energy_ev = 3000.0
    lno_thickness = 20.0
    sto_thickness = 20.0
    period = lno_thickness + sto_thickness
    repeats = 20

    layers = make_lno_sto_superlattice(
        energy_ev=energy_ev,
        repeats=repeats,
        lno_thickness=lno_thickness,
        sto_thickness=sto_thickness,
    )

    angles = np.linspace(0.05, 5.0, 500)
    reflectivity = np.array(
        [
            transfer_matrix_reflectivity(
                angle,
                energy_ev,
                layers,
                roughness_step=1.0,
            )
            for angle in angles
        ]
    )

    wavelength = energy_to_wavelength(energy_ev)
    first_bragg_angle = np.rad2deg(np.arcsin(wavelength / (2.0 * period)))

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.semilogy(angles, reflectivity, color="black", linewidth=1.4)
    ax.axvline(
        first_bragg_angle,
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label=f"m=1 Bragg estimate: {first_bragg_angle:.2f} deg",
    )

    ax.set_xlabel("Grazing incidence angle (deg)")
    ax.set_ylabel("Reflectivity")
    ax.set_title("LaNiO3/SrTiO3 multilayer reflectivity")
    ax.set_xlim(angles.min(), angles.max())
    ax.set_ylim(1e-4, 1.2)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path = Path(__file__).resolve().parent / "lno_sto_reflectivity.png"
    fig.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
