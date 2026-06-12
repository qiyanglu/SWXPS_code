"""Plot electric-field intensity versus depth and incidence angle."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swxps import (
    Layer,
    energy_to_wavelength,
    layer_from_file,
    transfer_matrix_reflectivity,
    transfer_matrix_electric_field_profile,
    vacuum,
)


def make_lno_sto_superlattice(
    energy_ev: float,
    repeats: int,
    lno_thickness: float,
    sto_thickness: float,
    roughness: float,
) -> list[Layer]:
    """Return vacuum / [LaNiO3 / SrTiO3]xN / SrTiO3 substrate."""

    lno_file = REPO_ROOT / "OPC" / "LaNiO3.dat"
    sto_file = REPO_ROOT / "OPC" / "SrTiO3.dat"

    layers = [vacuum()]
    for _ in range(repeats):
        layers.append(
            layer_from_file(
                lno_file,
                energy_ev,
                thickness=lno_thickness,
                roughness=roughness,
            )
        )
        layers.append(
            layer_from_file(
                sto_file,
                energy_ev,
                thickness=sto_thickness,
                roughness=roughness,
            )
        )
    layers.append(
        layer_from_file(
            sto_file,
            energy_ev,
            thickness=0.0,
            roughness=roughness,
        )
    )
    return layers


def draw_layer_boundaries(ax: plt.Axes, layers: list[Layer]) -> None:
    """Draw nominal layer boundaries on a depth axis."""

    depth = 0.0
    for layer in layers[1:-1]:
        depth += layer.thickness
        ax.axvline(depth, color="white", alpha=0.16, linewidth=0.7)


def main() -> None:
    energy_ev = 1000.0
    lno_thickness = 20.0
    sto_thickness = 20.0
    roughness = 3.0
    repeats = 20

    period = lno_thickness + sto_thickness
    wavelength = energy_to_wavelength(energy_ev)
    bragg_estimate = np.rad2deg(np.arcsin(wavelength / (2.0 * period)))

    layers = make_lno_sto_superlattice(
        energy_ev=energy_ev,
        repeats=repeats,
        lno_thickness=lno_thickness,
        sto_thickness=sto_thickness,
        roughness=roughness,
    )

    scan_angles = np.linspace(bragg_estimate - 1.5, bragg_estimate + 1.5, 320)
    reflectivity = np.array(
        [
            transfer_matrix_reflectivity(
                angle,
                energy_ev,
                layers,
                roughness_step=1.0,
            )
            for angle in scan_angles
        ]
    )
    peak_angle = scan_angles[np.argmax(reflectivity)]

    field_angles = np.linspace(bragg_estimate - 1.2, bragg_estimate + 1.2, 121)
    profiles = [
        transfer_matrix_electric_field_profile(
            angle_deg=angle,
            energy_ev=energy_ev,
            layers=layers,
            step=1.0,
            roughness_step=1.0,
        )
        for angle in field_angles
    ]
    depth = profiles[0].depth
    intensity = np.vstack([profile.intensity for profile in profiles])

    fig, (ax_r, ax_e) = plt.subplots(
        2,
        1,
        figsize=(8.4, 7.2),
        sharex=False,
        gridspec_kw={"height_ratios": [1.0, 1.7]},
    )

    ax_r.semilogy(scan_angles, reflectivity, color="black", linewidth=1.3)
    ax_r.axvline(
        bragg_estimate,
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label=f"Bragg estimate: {bragg_estimate:.2f} deg",
    )
    ax_r.scatter(
        [peak_angle],
        [reflectivity.max()],
        color="tab:blue",
        s=28,
        zorder=3,
        label=f"Peak maximum: {peak_angle:.2f} deg",
    )
    ax_r.set_ylabel("Reflectivity")
    ax_r.set_title(f"LaNiO3/SrTiO3 field map near first Bragg peak, {energy_ev:.0f} eV")
    ax_r.grid(True, which="both", alpha=0.25)
    ax_r.legend(loc="best")

    mesh = ax_e.pcolormesh(
        depth,
        field_angles,
        intensity,
        shading="auto",
        cmap="magma",
    )
    draw_layer_boundaries(ax_e, layers)
    ax_e.axhline(bragg_estimate, color="cyan", linestyle="--", linewidth=1.0)
    ax_e.axhline(peak_angle, color="white", linestyle="-", linewidth=1.0)
    ax_e.set_xlabel("Depth below surface (Angstrom)")
    ax_e.set_ylabel("Grazing incidence angle (deg)")
    ax_e.set_xlim(depth.min(), depth.max())
    ax_e.set_ylim(field_angles.min(), field_angles.max())
    colorbar = fig.colorbar(mesh, ax=ax_e, pad=0.02)
    colorbar.set_label("Field intensity |E|^2")
    fig.tight_layout()

    output_path = REPO_ROOT / "examples" / "lno_sto_field_profile.png"
    fig.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
