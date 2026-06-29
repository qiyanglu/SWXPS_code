"""Compare synthetic C/LaNiO3/SrTiO3 reflectivity with and without roughness."""

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
    TRUE_VALUES,
    bragg_angle_deg,
    build_stack,
    stack_template,
)
from swanx.optics import transfer_matrix_reflectivity  # noqa: E402


def reflectivity_curve(angles: np.ndarray, layers, roughness_step: float | None) -> np.ndarray:
    kwargs = {} if roughness_step is None else {"roughness_step": roughness_step}
    return np.array(
        [
            transfer_matrix_reflectivity(
                angle,
                PHOTON_ENERGY_EV,
                layers,
                **kwargs,
            )
            for angle in angles
        ],
        dtype=float,
    )


def main() -> None:
    sharp_values = {
        **TRUE_VALUES,
        "carbon_roughness": 0.0,
        "superlattice_roughness": 0.0,
        "substrate_roughness": 0.0,
    }
    scan_angles = np.linspace(bragg_angle_deg() - 2.0, bragg_angle_deg() + 2.0, 500)
    sharp_layers = stack_template(carbon_roughness=0.0).build(sharp_values).optical_layers
    rough_layers = build_stack().optical_layers
    sharp_reflectivity = reflectivity_curve(scan_angles, sharp_layers, roughness_step=None)
    rough_reflectivity = reflectivity_curve(scan_angles, rough_layers, roughness_step=1.0)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogy(scan_angles, sharp_reflectivity, color="black", linewidth=1.3, label="sharp interfaces")
    ax.semilogy(scan_angles, rough_reflectivity, color="tab:blue", linewidth=1.5, label="synthetic roughness")
    ax.axvline(
        bragg_angle_deg(),
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label=f"m=1 Bragg estimate: {bragg_angle_deg():.2f} deg",
    )
    ax.set_xlabel("Grazing incidence angle (deg)")
    ax.set_ylabel("Reflectivity")
    ax.set_title("C/[LaNiO3/SrTiO3]x20 roughness comparison")
    ax.set_xlim(scan_angles.min(), scan_angles.max())
    ax.set_ylim(1e-6, 1.2)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path = Path(__file__).resolve().parent / "synthetic_c_lno_sto_roughness_comparison.png"
    fig.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
