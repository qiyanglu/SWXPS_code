"""Compare roughness-profile shapes for the synthetic C/LaNiO3/SrTiO3 stack."""

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
from swanx.optics import transfer_matrix_reflectivity  # noqa: E402


def reflectivity_curve(angles: np.ndarray, layers, roughness_profile: str) -> np.ndarray:
    return np.array(
        [
            transfer_matrix_reflectivity(
                angle,
                PHOTON_ENERGY_EV,
                layers,
                roughness_step=1.0,
                roughness_profile=roughness_profile,
            )
            for angle in angles
        ],
        dtype=float,
    )


def main() -> None:
    layers = build_stack().optical_layers
    scan_angles = np.linspace(bragg_angle_deg() - 2.0, bragg_angle_deg() + 2.0, 500)
    erf_reflectivity = reflectivity_curve(scan_angles, layers, roughness_profile="erf")
    linear_reflectivity = reflectivity_curve(scan_angles, layers, roughness_profile="linear")
    fractional_difference = (linear_reflectivity - erf_reflectivity) / np.maximum(erf_reflectivity, 1e-12)

    fig, (ax_reflectivity, ax_difference) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_reflectivity.semilogy(scan_angles, erf_reflectivity, color="black", linewidth=1.4, label="error-function roughness")
    ax_reflectivity.semilogy(scan_angles, linear_reflectivity, color="tab:blue", linewidth=1.2, linestyle="--", label="linear roughness")
    ax_reflectivity.axvline(bragg_angle_deg(), color="tab:red", linestyle=":", linewidth=1.0, label="Bragg estimate")
    ax_reflectivity.set_ylabel("Reflectivity")
    ax_reflectivity.set_title("C/[LaNiO3/SrTiO3]x20: RMS-matched roughness profiles")
    ax_reflectivity.set_xlim(scan_angles.min(), scan_angles.max())
    ax_reflectivity.set_ylim(1e-6, 1.2)
    ax_reflectivity.grid(True, which="both", alpha=0.25)
    ax_reflectivity.legend()

    ax_difference.axhline(0.0, color="black", linewidth=0.8)
    ax_difference.plot(scan_angles, fractional_difference, color="tab:green", linewidth=1.1)
    ax_difference.axvline(bragg_angle_deg(), color="tab:red", linestyle=":", linewidth=1.0)
    ax_difference.set_xlabel("Grazing incidence angle (deg)")
    ax_difference.set_ylabel("(linear - erf) / erf")
    ax_difference.grid(True, alpha=0.25)
    fig.tight_layout()

    output_path = Path(__file__).resolve().parent / "synthetic_c_lno_sto_roughness_profile_comparison.png"
    fig.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
