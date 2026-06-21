"""Plot roughness-graded top concentration profiles for the best Sample#13 fit."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CASE_DIR = SCRIPT_DIR.parent
REPO_ROOT = CASE_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from swxps.xps import graded_layer_property_at_depth  # noqa: E402

import fit_sample13_joint_cap3_jax_gradient as fit13  # noqa: E402


RUN_DIR = REPO_ROOT / "runs" / "sample_13" / "jax_gradient" / "layer1_1A_multistart_35iter"
SUMMARY_PATH = RUN_DIR / "sample13_joint_cap3_jax_gradient_best_summary.csv"
OUTPUT_CSV = RUN_DIR / "sample13_top100_concentration_profiles.csv"
OUTPUT_PNG = RUN_DIR / "sample13_top100_concentration_profiles.png"


def main() -> None:
    fit13.patch_archived_context_paths()
    values = load_best_values(SUMMARY_PATH)
    stack = fit13.build_cap3_stack(values)
    depth = np.linspace(0.0, 100.0, 1001)

    c_by_layer = []
    la_by_layer = []
    ni_by_layer = []
    for index, material in enumerate(stack.materials):
        c_by_layer.append(1.0 if material == "C" else 0.0)
        la_by_layer.append(1.0 if material == "LNO" else 0.0)
        ni_by_layer.append(1.0 if material == "LNO" and index != 2 else 0.0)

    profiles = {
        "C": graded_layer_property_at_depth(stack.optical_layers, c_by_layer, depth),
        "La": graded_layer_property_at_depth(stack.optical_layers, la_by_layer, depth),
        "Ni": graded_layer_property_at_depth(stack.optical_layers, ni_by_layer, depth),
    }
    save_csv(depth, profiles)
    save_plot(depth, profiles, values)
    print(f"Saved {OUTPUT_CSV}")
    print(f"Saved {OUTPUT_PNG}")


def load_best_values(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    return {
        parameter.name: float(row[parameter.name])
        for parameter in fit13.PARAMETERS
    }


def save_csv(depth: np.ndarray, profiles: dict[str, np.ndarray]) -> None:
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["depth_A", "C", "La", "Ni"])
        for index, z in enumerate(depth):
            writer.writerow([
                z,
                profiles["C"][index],
                profiles["La"][index],
                profiles["Ni"][index],
            ])


def save_plot(
    depth: np.ndarray,
    profiles: dict[str, np.ndarray],
    values: dict[str, float],
) -> None:
    import matplotlib.pyplot as plt

    layer1 = values["top_lno_layer1_thickness"]
    layer2 = values["top_lno_total_thickness"] - layer1
    carbon = values["carbon_thickness"]
    boundaries = [
        (carbon, "C/LNO-1"),
        (carbon + layer1, "LNO-1/LNO-2"),
        (carbon + layer1 + layer2, "LNO-2/LNO-bottom"),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(depth, profiles["La"], color="tab:orange", linewidth=2.0, label="La")
    ax.plot(depth, profiles["Ni"], color="tab:blue", linewidth=2.0, label="Ni")
    ax.plot(depth, profiles["C"], color="tab:green", linewidth=2.0, label="C")
    for z, label in boundaries:
        if z <= depth[-1]:
            ax.axvline(z, color="0.35", linestyle="--", linewidth=0.9, alpha=0.65)
            ax.text(
                z + 0.8,
                0.08,
                label,
                rotation=90,
                va="bottom",
                ha="left",
                fontsize=8,
                color="0.25",
            )
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("Depth from surface (A)")
    ax.set_ylabel("Relative concentration")
    ax.set_title("Sample#13 Best Fit Top 100 A Concentration Profiles")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
