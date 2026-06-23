"""Compare surface-selected and whole-stack La 4d/Ni 3p rocking curves."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


FIT_DIR = Path(__file__).resolve().parent
CASE_DIR = FIT_DIR.parent
REPO_ROOT = CASE_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
RUN_DIR = REPO_ROOT / "runs" / "sample_13" / "jax_gradient_without_la4d" / "single_60iter"
SUMMARY_PATH = RUN_DIR / "sample13_reflectivity_c1s_ni3p_jax_gradient_best_summary.csv"
OUTPUT_PREFIX = "sample13_surface_vs_whole_stack_rc"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(FIT_DIR) not in sys.path:
    sys.path.insert(0, str(FIT_DIR))

from swxps import (  # noqa: E402
    CoreLevelRequest,
    RockingCurveRequest,
    imfp_from_file,
    simulate_rocking_curves,
)

import fit_sample13_reflectivity_c1s_ni3p_jax_gradient as fit13  # noqa: E402


SURFACE_INDICES = {
    "La 4d": (2, 3),
    "Ni 3p": (3,),
}


def main() -> None:
    fit13.patch_archived_context_paths()
    values = load_best_values(SUMMARY_PATH)
    data = fit13.sample13.load_and_prepare_data(10.0, 2)
    stack = fit13.build_cap3_stack(values)
    lno_indices = tuple(
        index for index, layer in enumerate(stack.layers) if layer.material == "LNO"
    )
    whole_indices = {
        "La 4d": lno_indices,
        "Ni 3p": tuple(index for index in lno_indices if index != 2),
    }

    surface = simulate_model(data.rc_angle, stack, values, SURFACE_INDICES)
    whole = simulate_model(data.rc_angle, stack, values, whole_indices)
    surface_by_name = {core.name: core.curve for core in surface.core_levels}
    whole_by_name = {core.name: core.curve for core in whole.core_levels}

    save_curves(data, surface_by_name, whole_by_name)
    metrics = save_summary(data, surface_by_name, whole_by_name, whole_indices)
    save_overlay_plot(data, surface_by_name, whole_by_name)
    save_difference_plot(data.rc_angle, surface_by_name, whole_by_name)

    print(f"Nominal LNO layers in stack: {len(lno_indices)}")
    print(f"Surface La indices: {SURFACE_INDICES['La 4d']}")
    print(f"Whole-stack La layer count: {len(whole_indices['La 4d'])}")
    print(f"Surface Ni indices: {SURFACE_INDICES['Ni 3p']}")
    print(f"Whole-stack Ni layer count: {len(whole_indices['Ni 3p'])}")
    for metric in metrics:
        print(
            f"{metric['core']}: RMS normalized difference="
            f"{metric['rms_normalized_difference']:.6g}, max="
            f"{metric['max_abs_normalized_difference']:.6g}, mean deeper raw fraction="
            f"{metric['mean_deeper_raw_fraction']:.3%}"
        )


def load_best_values(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    return {parameter.name: float(row[parameter.name]) for parameter in fit13.PARAMETERS}


def core_request(name: str, layer_indices: tuple[int, ...]) -> CoreLevelRequest:
    imfp_files = {
        "C": REPO_ROOT / "IMFP" / "C.ANG",
        "LNO": REPO_ROOT / "IMFP" / "LNO.ANG",
        "STO": REPO_ROOT / "IMFP" / "STO.ANG",
    }
    kinetic_energy = fit13.sample13.PHOTON_ENERGY_EV - fit13.sample13.BINDING_ENERGIES[name]
    imfp = {
        material: imfp_from_file(path, kinetic_energy)
        for material, path in imfp_files.items()
    }
    return CoreLevelRequest(
        name=name,
        binding_energy_ev=fit13.sample13.BINDING_ENERGIES[name],
        concentration_by_material={"LNO": 1.0},
        imfp_by_material={"vacuum": imfp["C"], **imfp},
        emitting_layer_indices=layer_indices,
    )


def simulate_model(angles, stack, values, indices_by_name):
    return simulate_rocking_curves(
        RockingCurveRequest(
            angles=angles,
            photon_energy_ev=fit13.sample13.PHOTON_ENERGY_EV,
            stack=stack,
            core_levels=tuple(
                core_request(name, indices_by_name[name]) for name in ("La 4d", "Ni 3p")
            ),
            angle_offset=values["rc_angle_offset"],
            field_step=5.0,
            roughness_step=2.0,
            offpeak_mask=np.ones_like(angles, dtype=bool),
            slicing=None,
        )
    )


def save_curves(data, surface, whole) -> None:
    columns = ["angle_deg"]
    for name in ("La 4d", "Ni 3p"):
        key = name.lower().replace(" ", "_")
        columns.extend(
            [
                f"{key}_experimental",
                f"{key}_surface_normalized",
                f"{key}_whole_normalized",
                f"{key}_whole_minus_surface",
                f"{key}_surface_raw",
                f"{key}_whole_raw",
                f"{key}_deeper_raw_fraction",
            ]
        )
    path = RUN_DIR / f"{OUTPUT_PREFIX}_curves.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, angle in enumerate(data.rc_angle):
            row = {"angle_deg": float(angle)}
            for name in ("La 4d", "Ni 3p"):
                key = name.lower().replace(" ", "_")
                surface_curve = surface[name]
                whole_curve = whole[name]
                row.update(
                    {
                        f"{key}_experimental": float(data.rc_normalized[name][index]),
                        f"{key}_surface_normalized": float(surface_curve.intensity[index]),
                        f"{key}_whole_normalized": float(whole_curve.intensity[index]),
                        f"{key}_whole_minus_surface": float(
                            whole_curve.intensity[index] - surface_curve.intensity[index]
                        ),
                        f"{key}_surface_raw": float(surface_curve.raw_intensity[index]),
                        f"{key}_whole_raw": float(whole_curve.raw_intensity[index]),
                        f"{key}_deeper_raw_fraction": float(
                            1.0
                            - surface_curve.raw_intensity[index]
                            / whole_curve.raw_intensity[index]
                        ),
                    }
                )
            writer.writerow(row)


def save_summary(data, surface, whole, whole_indices):
    rows = []
    for name in ("La 4d", "Ni 3p"):
        difference = whole[name].intensity - surface[name].intensity
        deeper_fraction = 1.0 - surface[name].raw_intensity / whole[name].raw_intensity
        experimental = data.rc_normalized[name]
        surface_mse = float(np.mean((surface[name].intensity - experimental) ** 2))
        whole_mse = float(np.mean((whole[name].intensity - experimental) ** 2))
        rows.append(
            {
                "core": name,
                "surface_layer_count": len(SURFACE_INDICES[name]),
                "whole_layer_count": len(whole_indices[name]),
                "rms_normalized_difference": float(np.sqrt(np.mean(difference**2))),
                "max_abs_normalized_difference": float(np.max(np.abs(difference))),
                "mean_deeper_raw_fraction": float(np.mean(deeper_fraction)),
                "min_deeper_raw_fraction": float(np.min(deeper_fraction)),
                "max_deeper_raw_fraction": float(np.max(deeper_fraction)),
                "whole_to_surface_mean_raw_ratio": float(
                    np.mean(whole[name].raw_intensity)
                    / np.mean(surface[name].raw_intensity)
                ),
                "surface_experimental_mse": surface_mse,
                "whole_experimental_mse": whole_mse,
                "experimental_mse_change_percent": 100.0
                * (whole_mse / surface_mse - 1.0),
            }
        )
    path = RUN_DIR / f"{OUTPUT_PREFIX}_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows

def save_overlay_plot(data, surface, whole) -> None:
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14})
    colors = {"experiment": "#4f5457", "surface": "#5f86a6", "whole": "#c47a6c"}
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 8.2), sharex=True)
    for ax, name in zip(axes, ("La 4d", "Ni 3p")):
        ax.scatter(
            data.rc_angle,
            data.rc_normalized[name],
            s=25,
            facecolors="none",
            edgecolors=colors["experiment"],
            linewidths=1.0,
            label="Experiment",
            zorder=3,
        )
        ax.plot(data.rc_angle, surface[name].intensity, color=colors["surface"], lw=2.2, label="Surface layers")
        ax.plot(data.rc_angle, whole[name].intensity, color=colors["whole"], lw=2.2, ls="--", label="Whole LNO stack")
        ax.set_title(name)
        ax.set_ylabel("Normalized intensity")
        ax.grid(alpha=0.22)
    axes[0].legend(frameon=False, ncol=3, loc="best")
    axes[-1].set_xlabel("Grazing incidence angle (deg)")
    fig.suptitle("Sample #13: emission-depth sensitivity", fontsize=17)
    fig.tight_layout()
    fig.savefig(RUN_DIR / f"{OUTPUT_PREFIX}_overlay.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_difference_plot(angles, surface, whole) -> None:
    colors = {"La 4d": "#789f8a", "Ni 3p": "#a67892"}
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.4), sharex=True)
    for name in ("La 4d", "Ni 3p"):
        difference = whole[name].intensity - surface[name].intensity
        deeper_fraction = 1.0 - surface[name].raw_intensity / whole[name].raw_intensity
        axes[0].plot(angles, difference, lw=2.2, color=colors[name], label=name)
        axes[1].plot(angles, 100.0 * deeper_fraction, lw=2.2, color=colors[name], label=name)
    axes[0].axhline(0.0, color="#575d60", lw=1.0, ls=":")
    axes[0].set_ylabel("Whole - surface\n(normalized intensity)")
    axes[1].set_ylabel("Signal below selected\nlayers (%)")
    axes[1].set_xlabel("Grazing incidence angle (deg)")
    for ax in axes:
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
    fig.suptitle("Effect of including the buried LNO stack", fontsize=17)
    fig.tight_layout()
    fig.savefig(RUN_DIR / f"{OUTPUT_PREFIX}_difference.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
