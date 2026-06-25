"""Plot off-peak electric-field intensity in the top 100 Angstrom."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swanx.optics import (

    energy_to_wavelength,

    transfer_matrix_electric_field_profile,

    transfer_matrix_reflectivity,

)

from swanx.stack import (

    Layer,

    LayerTemplate,

    StackTemplate,

    SuperlatticeTemplate,

)


def make_lno_sto_superlattice(
    energy_ev: float,
    repeats: int,
    layer_thickness: float,
    roughness: float,
) -> list[Layer]:
    """Return vacuum / [LaNiO3 / SrTiO3]xN / SrTiO3 substrate."""

    template = StackTemplate(
        energy_ev=energy_ev,
        base_dir=REPO_ROOT,
        parts=(
            LayerTemplate.vacuum(),
            SuperlatticeTemplate(
                repeats=repeats,
                period=(
                    LayerTemplate.from_file("LNO", "data/OPC/LaNiO3.dat", layer_thickness, roughness),
                    LayerTemplate.from_file("STO", "data/OPC/SrTiO3.dat", layer_thickness, roughness),
                ),
            ),
            LayerTemplate.from_file("STO", "data/OPC/SrTiO3.dat", 0.0, roughness),
        ),
    )
    return template.build().optical_layers


def shade_top_layers(ax: plt.Axes, layer_thickness: float, max_depth: float) -> None:
    """Shade LNO/STO layers in the top depth window."""

    depth = 0.0
    layer_number = 0
    while depth < max_depth:
        is_lno = layer_number % 2 == 0
        color = "tab:green" if is_lno else "tab:orange"
        label = "LaNiO3" if layer_number == 0 else "SrTiO3" if layer_number == 1 else None
        ax.axvspan(
            depth,
            min(depth + layer_thickness, max_depth),
            color=color,
            alpha=0.08,
            linewidth=0,
            label=label,
        )
        depth += layer_thickness
        layer_number += 1


def main() -> None:
    energy_ev = 1000.0
    layer_thickness = 20.0
    period = 2.0 * layer_thickness
    roughness = 3.0
    repeats = 20
    max_depth = 100.0

    wavelength = energy_to_wavelength(energy_ev)
    bragg_estimate = np.rad2deg(np.arcsin(wavelength / (2.0 * period)))

    layers = make_lno_sto_superlattice(
        energy_ev=energy_ev,
        repeats=repeats,
        layer_thickness=layer_thickness,
        roughness=roughness,
    )

    scan_angles = np.linspace(bragg_estimate - 1.5, bragg_estimate + 1.5, 500)
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

    offpeak_angles = np.array(
        [
            peak_angle - 1.2,
            peak_angle - 0.8,
            peak_angle - 0.45,
            peak_angle + 0.45,
            peak_angle + 0.8,
            peak_angle + 1.2,
        ]
    )
    offpeak_angles = offpeak_angles[
        (offpeak_angles >= scan_angles.min()) & (offpeak_angles <= scan_angles.max())
    ]

    profiles = [
        transfer_matrix_electric_field_profile(
            angle_deg=angle,
            energy_ev=energy_ev,
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
    for angle in offpeak_angles:
        ax_r.axvline(angle, color="tab:purple", alpha=0.35, linewidth=0.9)
    ax_r.set_ylabel("Reflectivity")
    ax_r.set_title(f"Off-peak field profiles near first Bragg peak, {energy_ev:.0f} eV")
    ax_r.grid(True, which="both", alpha=0.25)
    ax_r.legend(loc="best")

    shade_top_layers(ax_e, layer_thickness, max_depth)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(profiles)))
    for angle, profile, color in zip(offpeak_angles, profiles, colors):
        mask = profile.depth <= max_depth
        ax_e.plot(
            profile.depth[mask],
            profile.intensity[mask],
            color=color,
            linewidth=1.5,
            label=f"{angle:.2f} deg",
        )

    ax_e.set_xlabel("Depth below surface (Angstrom)")
    ax_e.set_ylabel("Field intensity |E|^2")
    ax_e.set_xlim(0.0, max_depth)
    ax_e.grid(True, alpha=0.25)
    ax_e.legend(loc="best", ncols=2)
    fig.tight_layout()

    output_path = Path(__file__).resolve().parent / "lno_sto_top100_offpeak_fields.png"
    fig.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
