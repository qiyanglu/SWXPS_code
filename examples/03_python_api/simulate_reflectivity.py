"""Plot reflectivity from the synthetic C/LNO/STO superlattice case."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from examples.synthetic_case import (  # noqa: E402
    PHOTON_ENERGY_EV,
    angles,
    bragg_angle_deg,
    simulate_case,
)


def main() -> None:
    scan_angles = angles(count=321)
    _, reflectivity, _ = simulate_case(angle_grid=scan_angles)
    bragg_angle = bragg_angle_deg()
    peak_angle = scan_angles[reflectivity.reflectivity.argmax()]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.semilogy(scan_angles, reflectivity.reflectivity, color="black", linewidth=1.4)
    ax.axvline(
        bragg_angle,
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label=f"m=1 Bragg estimate: {bragg_angle:.2f} deg",
    )
    ax.axvline(
        peak_angle,
        color="tab:blue",
        linestyle="-",
        linewidth=1.0,
        label=f"simulated peak: {peak_angle:.2f} deg",
    )

    ax.set_xlabel("Grazing incidence angle (deg)")
    ax.set_ylabel("Reflectivity")
    ax.set_title(f"C/[LaNiO3/SrTiO3]x20 reflectivity, {PHOTON_ENERGY_EV:.0f} eV")
    ax.set_xlim(scan_angles.min(), scan_angles.max())
    ax.set_ylim(1e-4, 1.2)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path = Path(__file__).resolve().parent / "synthetic_c_lno_sto_reflectivity.png"
    fig.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
