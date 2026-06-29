"""Plot off-peak fields in the top 100 Angstrom of the C/LNO/STO stack."""

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


def shade_top_layers(ax: plt.Axes, stack, max_depth: float) -> None:
    depth = 0.0
    colors = ("tab:brown", "tab:green", "tab:orange")
    for index, layer in enumerate(stack.optical_layers[1:-1]):
        if depth >= max_depth:
            break
        color = colors[0] if index == 0 else colors[1] if index % 2 == 1 else colors[2]
        ax.axvspan(
            depth,
            min(depth + layer.thickness, max_depth),
            color=color,
            alpha=0.08,
            linewidth=0,
        )
        depth += layer.thickness


def main() -> None:
    stack = build_stack()
    layers = stack.optical_layers
    max_depth = 100.0
    bragg_angle = bragg_angle_deg()

    scan_angles = np.linspace(bragg_angle - 1.5, bragg_angle + 1.5, 500)
    reflectivity = np.array(
        [
            transfer_matrix_reflectivity(angle, PHOTON_ENERGY_EV, layers, roughness_step=1.0)
            for angle in scan_angles
        ]
    )
    peak_angle = scan_angles[np.argmax(reflectivity)]
    offpeak_angles = np.array(
        [peak_angle - 1.2, peak_angle - 0.8, peak_angle - 0.45, peak_angle + 0.45, peak_angle + 0.8, peak_angle + 1.2]
    )
    offpeak_angles = offpeak_angles[
        (offpeak_angles >= scan_angles.min()) & (offpeak_angles <= scan_angles.max())
    ]
    profiles = [
        transfer_matrix_electric_field_profile(
            angle_deg=angle,
            energy_ev=PHOTON_ENERGY_EV,
            layers=layers,
            step=0.25,
            roughness_step=1.0,
        )
        for angle in offpeak_angles
    ]

    fig, (ax_r, ax_e) = plt.subplots(
        2,
        1,
        figsize=(8.2, 7.0),
        gridspec_kw={"height_ratios": [1.0, 1.45]},
    )
    ax_r.semilogy(scan_angles, reflectivity, color="black", linewidth=1.3)
    ax_r.axvline(bragg_angle, color="tab:red", linestyle="--", linewidth=1.0, label="Bragg estimate")
    ax_r.axvline(peak_angle, color="tab:blue", linewidth=1.0, label="peak")
    for angle in offpeak_angles:
        ax_r.axvline(angle, color="tab:purple", alpha=0.35, linewidth=0.9)
    ax_r.set_ylabel("Reflectivity")
    ax_r.set_title(f"Off-peak fields near first Bragg peak, {PHOTON_ENERGY_EV:.0f} eV")
    ax_r.grid(True, which="both", alpha=0.25)
    ax_r.legend(loc="best")

    shade_top_layers(ax_e, stack, max_depth)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(profiles)))
    for angle, profile, color in zip(offpeak_angles, profiles, colors):
        mask = profile.depth <= max_depth
        ax_e.plot(profile.depth[mask], profile.intensity[mask], color=color, linewidth=1.5, label=f"{angle:.2f} deg")

    ax_e.set_xlabel("Depth below surface (Angstrom)")
    ax_e.set_ylabel("Field intensity |E|^2")
    ax_e.set_xlim(0.0, max_depth)
    ax_e.grid(True, alpha=0.25)
    ax_e.legend(loc="best", ncols=2)
    fig.tight_layout()

    output_path = Path(__file__).resolve().parent / "synthetic_c_lno_sto_top100_offpeak_fields.png"
    fig.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
