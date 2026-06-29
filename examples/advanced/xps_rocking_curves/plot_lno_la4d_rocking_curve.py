"""Plot synthetic C/LaNiO3/SrTiO3 reflectivity and SW-XPS rocking curves."""

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

from examples.synthetic_case import (  # noqa: E402
    PHOTON_ENERGY_EV,
    RC_COLUMN_BY_NAME,
    angles,
    bragg_angle_deg,
    simulate_case,
)


PLOT_COLORS = {
    "La 4d": "tab:purple",
    "O 1s": "tab:green",
    "Ti 2p": "tab:orange",
    "C 1s": "tab:brown",
}


def main() -> None:
    scan_angles = angles()
    _, reflectivity, rc_result = simulate_case(angle_grid=scan_angles)
    bragg_angle = bragg_angle_deg()
    peak_angle = scan_angles[np.argmax(reflectivity.reflectivity)]
    offpeak_mask = np.abs(scan_angles - peak_angle) > 1.25

    fig, axes = plt.subplots(
        1 + len(RC_COLUMN_BY_NAME),
        1,
        figsize=(7.6, 9.6),
        sharex=True,
    )
    ax_r = axes[0]
    ax_r.semilogy(scan_angles, reflectivity.reflectivity, color="black", linewidth=1.3)
    ax_r.axvline(bragg_angle, color="tab:red", linestyle="--", linewidth=1.0)
    ax_r.axvline(peak_angle, color="tab:blue", linestyle="-", linewidth=1.0)
    ax_r.set_ylabel("Reflectivity")
    ax_r.set_title(
        f"C/[LaNiO3/SrTiO3]x20 reflectivity and SW-XPS RCs, {PHOTON_ENERGY_EV:.0f} eV"
    )
    ax_r.grid(True, which="both", alpha=0.25)

    for ax, core in zip(axes[1:], rc_result.core_levels):
        curve = core.curve
        ax.plot(
            curve.angle,
            curve.intensity,
            color=PLOT_COLORS[core.name],
            linewidth=1.8,
            label=f"{core.name}, KE={core.kinetic_energy_ev:.0f} eV",
        )
        ax.scatter(
            curve.angle[offpeak_mask],
            curve.intensity[offpeak_mask],
            color="tab:gray",
            s=10,
            alpha=0.45,
        )
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0)
        ax.axvline(bragg_angle, color="tab:red", linestyle="--", linewidth=1.0)
        ax.axvline(peak_angle, color="tab:blue", linestyle="-", linewidth=1.0)
        ax.set_ylabel("Norm. intensity")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Grazing incidence angle (deg)")
    axes[-1].set_xlim(scan_angles.min(), scan_angles.max())
    fig.tight_layout()

    output_path = Path(__file__).resolve().parent / "synthetic_c_lno_sto_rocking_curves.png"
    fig.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
