"""Compare LaNiO3/SrTiO3 reflectivity with and without interface roughness."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swanx.optics import (

    energy_to_wavelength,

    transfer_matrix_reflectivity,

)

from swanx.stack import (

    LayerTemplate,

    StackTemplate,

    SuperlatticeTemplate,

)


def make_lno_sto_superlattice(
    energy_ev: float,
    repeats: int,
    lno_thickness: float,
    sto_thickness: float,
    roughness: float,
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
                    LayerTemplate.from_file("LNO", "data/OPC/LaNiO3.dat", lno_thickness, roughness),
                    LayerTemplate.from_file("STO", "data/OPC/SrTiO3.dat", sto_thickness, roughness),
                ),
            ),
            LayerTemplate.from_file("STO", "data/OPC/SrTiO3.dat", 0.0, roughness),
        ),
    )
    return template.build().optical_layers


def main() -> None:
    energy_ev = 3000.0
    lno_thickness = 20.0
    sto_thickness = 20.0
    period = lno_thickness + sto_thickness
    repeats = 20

    angles = np.linspace(0.05, 5.0, 500)
    sharp_layers = make_lno_sto_superlattice(
        energy_ev=energy_ev,
        repeats=repeats,
        lno_thickness=lno_thickness,
        sto_thickness=sto_thickness,
        roughness=0.0,
    )
    rough_layers = make_lno_sto_superlattice(
        energy_ev=energy_ev,
        repeats=repeats,
        lno_thickness=lno_thickness,
        sto_thickness=sto_thickness,
        roughness=3.0,
    )

    sharp_reflectivity = np.array(
        [
            transfer_matrix_reflectivity(angle, energy_ev, sharp_layers)
            for angle in angles
        ]
    )
    rough_reflectivity = np.array(
        [
            transfer_matrix_reflectivity(
                angle,
                energy_ev,
                rough_layers,
                roughness_step=1.0,
            )
            for angle in angles
        ]
    )

    wavelength = energy_to_wavelength(energy_ev)
    first_bragg_angle = np.rad2deg(np.arcsin(wavelength / (2.0 * period)))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogy(
        angles,
        sharp_reflectivity,
        color="black",
        linewidth=1.3,
        label="0 Angstrom roughness",
    )
    ax.semilogy(
        angles,
        rough_reflectivity,
        color="tab:blue",
        linewidth=1.5,
        label="3 Angstrom roughness",
    )
    ax.axvline(
        first_bragg_angle,
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label=f"m=1 Bragg estimate: {first_bragg_angle:.2f} deg",
    )

    ax.set_xlabel("Grazing incidence angle (deg)")
    ax.set_ylabel("Reflectivity")
    ax.set_title("LaNiO3/SrTiO3 roughness comparison")
    ax.set_xlim(angles.min(), angles.max())
    ax.set_ylim(1e-6, 1.2)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path = Path(__file__).resolve().parent / "lno_sto_roughness_comparison.png"
    fig.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
