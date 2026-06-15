"""Compare error-function and linear roughness concentration profiles."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swxps import (  # noqa: E402
    SimulationStack,
    LayerTemplate,
    StackTemplate,
    SuperlatticeTemplate,
    sample_concentration_profiles,
)


def make_lno_sto_stack(
    energy_ev: float,
    repeats: int,
    layer_thickness: float,
    roughness: float,
) -> SimulationStack:
    """Return vacuum / [LaNiO3 / SrTiO3]xN / SrTiO3 substrate."""

    template = StackTemplate(
        energy_ev=energy_ev,
        base_dir=REPO_ROOT,
        parts=(
            LayerTemplate.vacuum(),
            SuperlatticeTemplate(
                repeats=repeats,
                period=(
                    LayerTemplate.from_file("LNO", "OPC/LaNiO3.dat", layer_thickness, roughness),
                    LayerTemplate.from_file("STO", "OPC/SrTiO3.dat", layer_thickness, roughness),
                ),
            ),
            LayerTemplate.from_file("STO", "OPC/SrTiO3.dat", 0.0, roughness),
        ),
    )
    return template.build()


def draw_layer_boundaries(ax: plt.Axes, repeats: int, layer_thickness: float) -> None:
    """Draw nominal layer boundaries."""

    total_layers = 2 * repeats
    for index in range(1, total_layers):
        ax.axhline(
            index * layer_thickness,
            color="0.85",
            linewidth=0.6,
            zorder=0,
        )


def plot_profiles(
    ax: plt.Axes,
    stack: SimulationStack,
    roughness_profile: str,
    repeats: int,
    layer_thickness: float,
    max_depth: float,
    panel_title: str,
) -> None:
    """Plot La and Ti profiles for one roughness model."""

    profiles = sample_concentration_profiles(
        stack,
        {
            "La": {"LNO": 1.0},
            "Ti": {"STO": 1.0},
        },
        step=0.1,
        roughness_profile=roughness_profile,
    )

    draw_layer_boundaries(ax, repeats, layer_thickness)
    ax.plot(
        profiles.profiles["La"],
        profiles.depth,
        color="tab:purple",
        linewidth=1.8,
        label="La",
    )
    ax.plot(
        profiles.profiles["Ti"],
        profiles.depth,
        color="tab:orange",
        linewidth=1.8,
        label="Ti",
    )
    ax.set_title(panel_title)
    ax.set_xlabel("Relative concentration")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.0, max_depth)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.25)


def main() -> None:
    energy_ev = 1000.0
    repeats = 20
    layer_thickness = 20.0
    roughness = 3.0
    max_depth = 100.0

    stack = make_lno_sto_stack(
        energy_ev=energy_ev,
        repeats=repeats,
        layer_thickness=layer_thickness,
        roughness=roughness,
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 6.2), sharey=True)
    plot_profiles(
        axes[0],
        stack,
        roughness_profile="erf",
        repeats=repeats,
        layer_thickness=layer_thickness,
        max_depth=max_depth,
        panel_title="error-function roughness",
    )
    plot_profiles(
        axes[1],
        stack,
        roughness_profile="linear",
        repeats=repeats,
        layer_thickness=layer_thickness,
        max_depth=max_depth,
        panel_title=f"linear roughness\nhalf-width = {np.sqrt(3.0) * roughness:.1f} A",
    )
    axes[0].set_ylabel("Depth below surface (Angstrom)")
    axes[1].legend(loc="lower right")
    fig.suptitle("LNO/STO concentration profiles with 3 A roughness", y=0.995)
    fig.tight_layout()

    output_path = Path(__file__).resolve().parent / "lno_sto_concentration_profile_comparison.png"
    fig.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
