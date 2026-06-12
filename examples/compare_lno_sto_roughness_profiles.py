"""Compare error-function and linear roughness profiles in LNO/STO reflectivity."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swxps import (  # noqa: E402
    Layer,
    energy_to_wavelength,
    layer_from_file,
    transfer_matrix_reflectivity,
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


def reflectivity_curve(
    angles: np.ndarray,
    energy_ev: float,
    layers: list[Layer],
    roughness_profile: str,
) -> np.ndarray:
    """Return transfer-matrix reflectivity for one roughness profile shape."""

    return np.array(
        [
            transfer_matrix_reflectivity(
                angle,
                energy_ev,
                layers,
                roughness_step=1.0,
                roughness_profile=roughness_profile,
            )
            for angle in angles
        ],
        dtype=float,
    )


def main() -> None:
    energy_ev = 3000.0
    lno_thickness = 20.0
    sto_thickness = 20.0
    period = lno_thickness + sto_thickness
    repeats = 20
    roughness = 3.0

    layers = make_lno_sto_superlattice(
        energy_ev=energy_ev,
        repeats=repeats,
        lno_thickness=lno_thickness,
        sto_thickness=sto_thickness,
        roughness=roughness,
    )

    angles = np.linspace(0.05, 5.0, 500)
    erf_reflectivity = reflectivity_curve(
        angles,
        energy_ev,
        layers,
        roughness_profile="erf",
    )
    linear_reflectivity = reflectivity_curve(
        angles,
        energy_ev,
        layers,
        roughness_profile="linear",
    )

    fractional_difference = (
        linear_reflectivity - erf_reflectivity
    ) / np.maximum(erf_reflectivity, 1e-12)

    wavelength = energy_to_wavelength(energy_ev)
    first_bragg_angle = np.rad2deg(np.arcsin(wavelength / (2.0 * period)))

    fig, (ax_reflectivity, ax_difference) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_reflectivity.semilogy(
        angles,
        erf_reflectivity,
        color="black",
        linewidth=1.4,
        label="error-function roughness",
    )
    ax_reflectivity.semilogy(
        angles,
        linear_reflectivity,
        color="tab:blue",
        linewidth=1.2,
        linestyle="--",
        label="linear roughness, RMS-matched",
    )
    ax_reflectivity.axvline(
        first_bragg_angle,
        color="tab:red",
        linestyle=":",
        linewidth=1.0,
        label=f"m=1 Bragg estimate: {first_bragg_angle:.2f} deg",
    )
    ax_reflectivity.set_ylabel("Reflectivity")
    ax_reflectivity.set_title("LNO/STO reflectivity: RMS-matched roughness profiles")
    ax_reflectivity.set_xlim(angles.min(), angles.max())
    ax_reflectivity.set_ylim(1e-6, 1.2)
    ax_reflectivity.grid(True, which="both", alpha=0.25)
    ax_reflectivity.legend()

    ax_difference.axhline(0.0, color="black", linewidth=0.8)
    ax_difference.plot(
        angles,
        fractional_difference,
        color="tab:green",
        linewidth=1.1,
    )
    ax_difference.axvline(
        first_bragg_angle,
        color="tab:red",
        linestyle=":",
        linewidth=1.0,
    )
    ax_difference.set_xlabel("Grazing incidence angle (deg)")
    ax_difference.set_ylabel("(linear - erf) / erf")
    ax_difference.grid(True, alpha=0.25)

    fig.tight_layout()

    output_path = REPO_ROOT / "examples" / "lno_sto_roughness_profile_comparison.png"
    fig.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
