"""Plot electric-field intensity for the synthetic C/LaNiO3/SrTiO3 stack."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from examples.synthetic_case import PHOTON_ENERGY_EV, bragg_angle_deg, build_stack  # noqa: E402
from swanx.optics import transfer_matrix_electric_field_profile, transfer_matrix_reflectivity  # noqa: E402


def draw_layer_boundaries(ax: plt.Axes, stack) -> None:
    depth = 0.0
    for layer in stack.optical_layers[1:-1]:
        depth += layer.thickness
        ax.axvline(depth, color="white", alpha=0.16, linewidth=0.7)


def main() -> None:
    stack = build_stack()
    layers = stack.optical_layers
    bragg_angle = bragg_angle_deg()
    scan_angles = np.linspace(bragg_angle - 1.5, bragg_angle + 1.5, 320)
    reflectivity = np.array(
        [
            transfer_matrix_reflectivity(angle, PHOTON_ENERGY_EV, layers, roughness_step=1.0)
            for angle in scan_angles
        ]
    )
    peak_angle = scan_angles[np.argmax(reflectivity)]

    field_angles = np.linspace(bragg_angle - 1.2, bragg_angle + 1.2, 121)
    profiles = [
        transfer_matrix_electric_field_profile(
            angle_deg=angle,
            energy_ev=PHOTON_ENERGY_EV,
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
        gridspec_kw={"height_ratios": [1.0, 1.7]},
    )
    ax_r.semilogy(scan_angles, reflectivity, color="black", linewidth=1.3)
    ax_r.axvline(bragg_angle, color="tab:red", linestyle="--", linewidth=1.0, label="Bragg estimate")
    ax_r.axvline(peak_angle, color="tab:blue", linestyle="-", linewidth=1.0, label="peak")
    ax_r.set_ylabel("Reflectivity")
    ax_r.set_title(
        f"C/[LaNiO3/SrTiO3]x20 field map near first Bragg peak, {PHOTON_ENERGY_EV:.0f} eV"
    )
    ax_r.grid(True, which="both", alpha=0.25)
    ax_r.legend(loc="best")

    mesh = ax_e.pcolormesh(depth, field_angles, intensity, shading="auto", cmap="magma")
    draw_layer_boundaries(ax_e, stack)
    ax_e.axhline(bragg_angle, color="cyan", linestyle="--", linewidth=1.0)
    ax_e.axhline(peak_angle, color="white", linestyle="-", linewidth=1.0)
    ax_e.set_xlabel("Depth below surface (Angstrom)")
    ax_e.set_ylabel("Grazing incidence angle (deg)")
    ax_e.set_xlim(depth.min(), depth.max())
    ax_e.set_ylim(field_angles.min(), field_angles.max())
    fig.colorbar(mesh, ax=ax_e, pad=0.02).set_label("Field intensity |E|^2")
    fig.tight_layout()

    output_path = Path(__file__).resolve().parent / "synthetic_c_lno_sto_field_profile.png"
    fig.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
