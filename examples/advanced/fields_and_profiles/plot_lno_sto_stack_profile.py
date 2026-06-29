"""Plot roughness-broadened concentration profiles for C/LaNiO3/SrTiO3."""

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


def main() -> None:
    stack = build_stack()
    max_depth = 120.0
    profiles = sample_concentration_profiles(
        stack,
        {
            "C": {"C": 1.0},
            "La": {"LNO": 1.0},
            "Ti": {"STO": 1.0},
            "O": {"LNO": 1.0, "STO": 1.0},
        },
        step=0.25,
    )

    fig, ax = plt.subplots(figsize=(5.8, 6.6))
    draw_layer_boundaries(ax, stack, max_depth)
    ax.plot(profiles.profiles["C"], profiles.depth, color="tab:brown", linewidth=1.8, label="C")
    ax.plot(profiles.profiles["La"], profiles.depth, color="tab:purple", linewidth=1.8, label="La")
    ax.plot(profiles.profiles["Ti"], profiles.depth, color="tab:orange", linewidth=1.8, label="Ti")
    ax.plot(profiles.profiles["O"], profiles.depth, color="tab:green", linewidth=1.4, label="O")

    ax.set_xlabel("Relative concentration")
    ax.set_ylabel("Depth below surface (Angstrom)")
    ax.set_title("Synthetic C/[LaNiO3/SrTiO3]x20 concentration profile")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.0, max_depth)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    output_path = Path(__file__).resolve().parent / "synthetic_c_lno_sto_stack_profile.png"
    fig.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
