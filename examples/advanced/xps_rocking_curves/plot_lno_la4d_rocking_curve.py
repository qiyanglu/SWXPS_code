"""Plot normalized La 4d, O 1s, and Ti 2p standing-wave XPS rocking curves."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swanx.stack import (
    LayerTemplate,
    StackTemplate,
    SuperlatticeTemplate,
)
from swanx.optics import energy_to_wavelength
from swanx.io import read_imfp
from swanx.workflows import (
    CoreLevelRequest,
    ReflectivityRequest,
    RockingCurveRequest,
    simulate_reflectivity,
    simulate_rocking_curves,
)


def make_lno_sto_superlattice(
    energy_ev: float,
    repeats: int,
    layer_thickness: float,
    roughness: float,
) -> object:
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
    return template.build()


def main() -> None:
    photon_energy_ev = 1000.0
    la4d_binding_energy_ev = 105.0
    o1s_binding_energy_ev = 530.0
    ti2p_binding_energy_ev = 460.0
    la4d_kinetic_energy_ev = photon_energy_ev - la4d_binding_energy_ev
    o1s_kinetic_energy_ev = photon_energy_ev - o1s_binding_energy_ev
    ti2p_kinetic_energy_ev = photon_energy_ev - ti2p_binding_energy_ev
    layer_thickness = 20.0
    roughness = 3.0
    repeats = 20

    lno_imfp = read_imfp(REPO_ROOT / "data" / "IMFP" / "LNO.ANG")
    sto_imfp = read_imfp(REPO_ROOT / "data" / "IMFP" / "STO.ANG")
    lno_la4d_imfp = lno_imfp.at_kinetic_energy(la4d_kinetic_energy_ev)
    sto_la4d_imfp = sto_imfp.at_kinetic_energy(la4d_kinetic_energy_ev)
    lno_o1s_imfp = lno_imfp.at_kinetic_energy(o1s_kinetic_energy_ev)
    sto_o1s_imfp = sto_imfp.at_kinetic_energy(o1s_kinetic_energy_ev)
    lno_ti2p_imfp = lno_imfp.at_kinetic_energy(ti2p_kinetic_energy_ev)
    sto_ti2p_imfp = sto_imfp.at_kinetic_energy(ti2p_kinetic_energy_ev)

    stack = make_lno_sto_superlattice(
        energy_ev=photon_energy_ev,
        repeats=repeats,
        layer_thickness=layer_thickness,
        roughness=roughness,
    )

    period = 2.0 * layer_thickness
    wavelength = energy_to_wavelength(photon_energy_ev)
    bragg_estimate = np.rad2deg(np.arcsin(wavelength / (2.0 * period)))

    angles = np.linspace(bragg_estimate - 2.0, bragg_estimate + 2.0, 161)
    reflectivity_result = simulate_reflectivity(
        ReflectivityRequest(
            angles=angles,
            energy_ev=photon_energy_ev,
            stack=stack,
            angle_offset=0.0,
            roughness_step=1.0,
            slicing=None,
        )
    )
    reflectivity = reflectivity_result.reflectivity
    peak_angle = angles[np.argmax(reflectivity)]
    offpeak_mask = np.abs(angles - peak_angle) > 1.25

    core_levels = (
        CoreLevelRequest(
            name="La 4d",
            binding_energy_ev=la4d_binding_energy_ev,
            concentration_by_material={"LNO": 1.0},
            imfp_by_material={
                "vacuum": lno_la4d_imfp,
                "LNO": lno_la4d_imfp,
                "STO": sto_la4d_imfp,
            },
        ),
        CoreLevelRequest(
            name="O 1s",
            binding_energy_ev=o1s_binding_energy_ev,
            concentration_by_material={"LNO": 1.0, "STO": 1.0},
            imfp_by_material={
                "vacuum": lno_o1s_imfp,
                "LNO": lno_o1s_imfp,
                "STO": sto_o1s_imfp,
            },
        ),
        CoreLevelRequest(
            name="Ti 2p",
            binding_energy_ev=ti2p_binding_energy_ev,
            concentration_by_material={"STO": 1.0},
            imfp_by_material={
                "vacuum": lno_ti2p_imfp,
                "LNO": lno_ti2p_imfp,
                "STO": sto_ti2p_imfp,
            },
        ),
    )
    rc_result = simulate_rocking_curves(
        RockingCurveRequest(
            angles=angles,
            photon_energy_ev=photon_energy_ev,
            stack=stack,
            core_levels=core_levels,
            angle_offset=0.0,
            field_step=1.0,
            roughness_step=1.0,
            offpeak_mask=offpeak_mask,
            slicing=None,
        )
    )
    la4d_curve = rc_result.core_levels[0].curve
    o1s_curve = rc_result.core_levels[1].curve
    ti2p_curve = rc_result.core_levels[2].curve

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(7.6, 8.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 1.0]},
    )
    ax_r, ax_la, ax_o, ax_ti = axes

    ax_r.semilogy(angles, reflectivity, color="black", linewidth=1.3)
    ax_r.axvline(bragg_estimate, color="tab:red", linestyle="--", linewidth=1.0)
    ax_r.axvline(peak_angle, color="tab:blue", linestyle="-", linewidth=1.0)
    ax_r.set_ylabel("Reflectivity")
    ax_r.set_title("Normalized SW-XPS rocking curves for LNO/STO")
    ax_r.grid(True, which="both", alpha=0.25)

    core_axes = [
        (
            ax_la,
            la4d_curve,
            "tab:purple",
            f"La 4d, KE={la4d_kinetic_energy_ev:.0f} eV",
        ),
        (
            ax_o,
            o1s_curve,
            "tab:green",
            f"O 1s, KE={o1s_kinetic_energy_ev:.0f} eV",
        ),
        (
            ax_ti,
            ti2p_curve,
            "tab:orange",
            f"Ti 2p, KE={ti2p_kinetic_energy_ev:.0f} eV",
        ),
    ]

    for ax, curve, color, label in core_axes:
        ax.plot(curve.angle, curve.intensity, color=color, linewidth=1.8, label=label)
        ax.scatter(
            curve.angle[offpeak_mask],
            curve.intensity[offpeak_mask],
            color="tab:gray",
            s=10,
            alpha=0.45,
        )
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0)
        ax.axvline(bragg_estimate, color="tab:red", linestyle="--", linewidth=1.0)
        ax.axvline(peak_angle, color="tab:blue", linestyle="-", linewidth=1.0)
        ax.set_ylabel("Norm. intensity")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    ax_la.plot([], [], color="tab:red", linestyle="--", label="Bragg estimate")
    ax_la.plot([], [], color="tab:blue", linestyle="-", label="reflectivity peak")
    ax_la.legend(loc="best")

    ax_ti.set_xlabel("Grazing incidence angle (deg)")
    ax_ti.set_xlim(angles.min(), angles.max())
    fig.tight_layout()

    output_path = Path(__file__).resolve().parent / "lno_la4d_o1s_ti2p_rocking_curves.png"
    fig.savefig(output_path, dpi=200)
    print(f"LNO IMFP at La 4d KE {la4d_kinetic_energy_ev:.1f} eV: {lno_la4d_imfp:.2f} Angstrom")
    print(f"STO IMFP at La 4d KE {la4d_kinetic_energy_ev:.1f} eV: {sto_la4d_imfp:.2f} Angstrom")
    print(f"LNO IMFP at O 1s KE {o1s_kinetic_energy_ev:.1f} eV: {lno_o1s_imfp:.2f} Angstrom")
    print(f"STO IMFP at O 1s KE {o1s_kinetic_energy_ev:.1f} eV: {sto_o1s_imfp:.2f} Angstrom")
    print(f"LNO IMFP at Ti 2p KE {ti2p_kinetic_energy_ev:.1f} eV: {lno_ti2p_imfp:.2f} Angstrom")
    print(f"STO IMFP at Ti 2p KE {ti2p_kinetic_energy_ev:.1f} eV: {sto_ti2p_imfp:.2f} Angstrom")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
