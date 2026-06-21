"""Plot vertical concentration profiles for the Sample#13 single_60iter fit."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
CASE_DIR = SCRIPT_DIR.parent
REPO_ROOT = CASE_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from swxps import plot_vertical_concentration_profiles  # noqa: E402

import fit_sample13_joint_cap3_jax_gradient_lno2_roughness as fit13  # noqa: E402


RUN_DIR = REPO_ROOT / "runs" / "sample_13" / "jax_gradient_lno2_roughness" / "single_60iter"
SUMMARY_PATH = RUN_DIR / "sample13_joint_cap3_lnorough_jax_gradient_best_summary.csv"
OUTPUT_PNG = RUN_DIR / "sample13_single60_top30_vertical_concentration_profiles.png"
OUTPUT_CSV = RUN_DIR / "sample13_single60_top30_vertical_concentration_profiles.csv"


def main() -> None:
    fit13.patch_archived_context_paths()
    values = load_best_values(SUMMARY_PATH)
    stack = fit13.build_cap3_stack(values)
    concentrations = concentration_by_layer(stack)
    labels = {
        1: "C",
        2: "LNO-1 Ni-free",
        3: "LNO-2",
        4: "LNO-bottom",
    }
    profiles = plot_vertical_concentration_profiles(
        OUTPUT_PNG,
        stack,
        concentrations,
        max_depth=30.0,
        step=0.1,
        title="Sample#13 top 30 A concentration profiles",
        layer_labels=labels,
        show_layer_shading=False,
        layer_box_style=True,
        categorical_strips=True,
    )
    save_csv(profiles.depth, profiles.profiles)
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_CSV}")


def load_best_values(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    return {
        parameter.name: float(row[parameter.name])
        for parameter in fit13.PARAMETERS
    }


def concentration_by_layer(stack) -> dict[str, list[float]]:
    c = []
    la = []
    ni = []
    for index, material in enumerate(stack.materials):
        c.append(1.0 if material == "C" else 0.0)
        la.append(1.0 if material == "LNO" else 0.0)
        ni.append(1.0 if material == "LNO" and index != 2 else 0.0)
    return {"La": la, "Ni": ni, "C": c}


def save_csv(depth, profiles: dict[str, object]) -> None:
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["depth_A", *profiles])
        for row_index, z in enumerate(depth):
            writer.writerow([z, *[values[row_index] for values in profiles.values()]])


if __name__ == "__main__":
    main()
