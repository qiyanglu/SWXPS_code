"""Compare error-function and linear roughness concentration profiles."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from examples.synthetic_case import build_stack  # noqa: E402
from swanx.stack import sample_concentration_profiles  # noqa: E402


def draw_layer_boundaries(ax: plt.Axes, stack, max_depth: float) -> None:
    depth = 0.0
    for layer in stack.optical_layers[1:-1]:
        depth += layer.thickness
        if depth >= max_depth:
            break
        ax.axhline(depth, color="0.85", linewidth=0.6, zorder=0)


def plot_profiles(ax: plt.Axes, stack, roughness_profile: str, max_depth: float, title: str) -> None:
    profiles = sample_concentration_profiles(
        stack,
        {
            "C": {"C": 1.0},
            "La": {"LNO": 1.0},
            "Ti": {"STO": 1.0},
        },
        step=0.1,
        roughness_profile=roughness_profile,
    )
    draw_layer_boundaries(ax, stack, max_depth)
    ax.plot(profiles.profiles["C"], profiles.depth, color="tab:brown", linewidth=1.8, label="C")
    ax.plot(profiles.profiles["La"], profiles.depth, color="tab:purple", linewidth=1.8, label="La")
    ax.plot(profiles.profiles["Ti"], profiles.depth, color="tab:orange", linewidth=1.8, label="Ti")
    ax.set_title(title)
    ax.set_xlabel("Relative concentration")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.0, max_depth)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.25)


def main() -> None:
    stack = build_stack()
    max_depth = 120.0
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 6.2), sharey=True)
    plot_profiles(axes[0], stack, "erf", max_depth, "error-function roughness")
    plot_profiles(axes[1], stack, "linear", max_depth, "linear roughness")
    axes[0].set_ylabel("Depth below surface (Angstrom)")
    axes[1].legend(loc="lower right")
    fig.suptitle("Synthetic C/[LaNiO3/SrTiO3]x20 concentration profiles", y=0.995)
    fig.tight_layout()

    output_path = Path(__file__).resolve().parent / "synthetic_c_lno_sto_concentration_profile_comparison.png"
    fig.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
