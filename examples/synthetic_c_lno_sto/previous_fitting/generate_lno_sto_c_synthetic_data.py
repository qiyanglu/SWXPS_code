"""Generate synthetic reflectivity and SW-XPS data for C/LNO/STO."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swxps import (
    CoreLevelRequest,
    LayerTemplate,
    ReflectivityRequest,
    RockingCurveRequest,
    SimulationStack,
    StackTemplate,
    SuperlatticeTemplate,
    energy_to_wavelength,
    imfp_from_file,
    simulate_reflectivity,
    simulate_rocking_curves,
)


def c_lno_sto_template(
    energy_ev: float,
    repeats: int = 20,
    carbon_thickness: float | str = "carbon_thickness",
    carbon_roughness: float | str = "carbon_roughness",
    lno_thickness: float | str = "lno_thickness",
    sto_thickness: float | str = "sto_thickness",
    superlattice_roughness: float | str = "superlattice_roughness",
    substrate_roughness: float | str = "substrate_roughness",
) -> StackTemplate:
    """Return a declarative vacuum / C / [LNO / STO]xN / STO template."""

    return StackTemplate(
        energy_ev=energy_ev,
        base_dir=REPO_ROOT,
        parts=(
            LayerTemplate.vacuum(),
            LayerTemplate.from_file(
                "C",
                "OPC/C.dat",
                thickness=carbon_thickness,
                roughness=carbon_roughness,
            ),
            SuperlatticeTemplate(
                repeats=repeats,
                period=(
                    LayerTemplate.from_file(
                        "LNO",
                        "OPC/LaNiO3.dat",
                        thickness=lno_thickness,
                        roughness=superlattice_roughness,
                    ),
                    LayerTemplate.from_file(
                        "STO",
                        "OPC/SrTiO3.dat",
                        thickness=sto_thickness,
                        roughness=superlattice_roughness,
                    ),
                ),
            ),
            LayerTemplate.from_file(
                "STO",
                "OPC/SrTiO3.dat",
                thickness=0.0,
                roughness=substrate_roughness,
            ),
        ),
    )


def make_c_lno_sto_stack(
    energy_ev: float,
    repeats: int = 20,
    carbon_thickness: float = 10.0,
    carbon_roughness: float = 2.0,
    lno_thickness: float = 20.0,
    sto_thickness: float = 20.0,
    superlattice_roughness: float = 3.0,
    substrate_roughness: float = 3.0,
) -> SimulationStack:
    """Return vacuum / C / [LaNiO3 / SrTiO3]xN / SrTiO3 substrate."""

    return c_lno_sto_template(
        energy_ev=energy_ev,
        repeats=repeats,
        carbon_thickness=carbon_thickness,
        carbon_roughness=carbon_roughness,
        lno_thickness=lno_thickness,
        sto_thickness=sto_thickness,
        superlattice_roughness=superlattice_roughness,
        substrate_roughness=substrate_roughness,
    ).build()


def core_level_requests(photon_energy_ev: float) -> tuple[CoreLevelRequest, ...]:
    """Return La 4d, O 1s, Ti 2p, and C 1s core-level requests."""

    binding_energies = {
        "La 4d": 105.0,
        "O 1s": 530.0,
        "Ti 2p": 460.0,
        "C 1s": 285.0,
    }
    imfp_files = {
        "LNO": REPO_ROOT / "IMFP" / "LNO.ANG",
        "STO": REPO_ROOT / "IMFP" / "STO.ANG",
        "C": REPO_ROOT / "IMFP" / "C.ANG",
    }

    imfp_by_core = {}
    for core_name, binding_energy in binding_energies.items():
        kinetic_energy = photon_energy_ev - binding_energy
        imfp_by_core[core_name] = {
            material: imfp_from_file(path, kinetic_energy)
            for material, path in imfp_files.items()
        }

    return (
        CoreLevelRequest(
            name="La 4d",
            binding_energy_ev=binding_energies["La 4d"],
            concentration_by_material={"LNO": 1.0},
            imfp_by_material={
                "vacuum": imfp_by_core["La 4d"]["C"],
                "C": imfp_by_core["La 4d"]["C"],
                "LNO": imfp_by_core["La 4d"]["LNO"],
                "STO": imfp_by_core["La 4d"]["STO"],
            },
        ),
        CoreLevelRequest(
            name="O 1s",
            binding_energy_ev=binding_energies["O 1s"],
            concentration_by_material={"LNO": 1.0, "STO": 1.0},
            imfp_by_material={
                "vacuum": imfp_by_core["O 1s"]["C"],
                "C": imfp_by_core["O 1s"]["C"],
                "LNO": imfp_by_core["O 1s"]["LNO"],
                "STO": imfp_by_core["O 1s"]["STO"],
            },
        ),
        CoreLevelRequest(
            name="Ti 2p",
            binding_energy_ev=binding_energies["Ti 2p"],
            concentration_by_material={"STO": 1.0},
            imfp_by_material={
                "vacuum": imfp_by_core["Ti 2p"]["C"],
                "C": imfp_by_core["Ti 2p"]["C"],
                "LNO": imfp_by_core["Ti 2p"]["LNO"],
                "STO": imfp_by_core["Ti 2p"]["STO"],
            },
        ),
        CoreLevelRequest(
            name="C 1s",
            binding_energy_ev=binding_energies["C 1s"],
            concentration_by_material={"C": 1.0},
            imfp_by_material={
                "vacuum": imfp_by_core["C 1s"]["C"],
                "C": imfp_by_core["C 1s"]["C"],
                "LNO": imfp_by_core["C 1s"]["LNO"],
                "STO": imfp_by_core["C 1s"]["STO"],
            },
        ),
    )


def main() -> None:
    photon_energy_ev = 1000.0
    lno_thickness = 20.0
    sto_thickness = 20.0
    period = lno_thickness + sto_thickness

    stack = make_c_lno_sto_stack(
        energy_ev=photon_energy_ev,
        lno_thickness=lno_thickness,
        sto_thickness=sto_thickness,
    )
    wavelength = energy_to_wavelength(photon_energy_ev)
    bragg_estimate = np.rad2deg(np.arcsin(wavelength / (2.0 * period)))
    angles = np.linspace(bragg_estimate - 2.0, bragg_estimate + 2.0, 161)

    reflectivity_result = simulate_reflectivity(
        ReflectivityRequest(
            angles=angles,
            energy_ev=photon_energy_ev,
            stack=stack,
            roughness_step=1.0,
        )
    )
    reflectivity = reflectivity_result.reflectivity
    peak_angle = angles[np.argmax(reflectivity)]
    offpeak_mask = np.abs(angles - peak_angle) > 1.25

    rc_result = simulate_rocking_curves(
        RockingCurveRequest(
            angles=angles,
            photon_energy_ev=photon_energy_ev,
            stack=stack,
            core_levels=core_level_requests(photon_energy_ev),
            field_step=1.0,
            roughness_step=1.0,
            offpeak_mask=offpeak_mask,
        )
    )
    curves = {core.name: core.curve.intensity for core in rc_result.core_levels}

    output_data = Path(__file__).resolve().parent / "lno_sto_c_synthetic_data.csv"
    table = np.column_stack(
        [
            angles,
            reflectivity,
            curves["La 4d"],
            curves["O 1s"],
            curves["Ti 2p"],
            curves["C 1s"],
        ]
    )
    header = "angle_deg,reflectivity,la4d_rc,o1s_rc,ti2p_rc,c1s_rc"
    np.savetxt(output_data, table, delimiter=",", header=header, comments="")

    fig, axes = plt.subplots(5, 1, figsize=(7.6, 9.6), sharex=True)
    axes[0].semilogy(angles, reflectivity, color="black", linewidth=1.3)
    axes[0].set_ylabel("Reflectivity")
    axes[0].grid(True, which="both", alpha=0.25)

    plot_specs = [
        ("La 4d", "tab:purple"),
        ("O 1s", "tab:green"),
        ("Ti 2p", "tab:orange"),
        ("C 1s", "tab:brown"),
    ]
    for ax, (name, color) in zip(axes[1:], plot_specs):
        ax.plot(angles, curves[name], color=color, linewidth=1.8, label=name)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0)
        ax.set_ylabel("Norm. RC")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    for ax in axes:
        ax.axvline(bragg_estimate, color="tab:red", linestyle="--", linewidth=1.0)
        ax.axvline(peak_angle, color="tab:blue", linestyle="-", linewidth=1.0)

    axes[-1].set_xlabel("Grazing incidence angle (deg)")
    axes[-1].set_xlim(angles.min(), angles.max())
    fig.suptitle("Synthetic C/LNO/STO reflectivity and SW-XPS rocking curves")
    fig.tight_layout()

    output_figure = Path(__file__).resolve().parent / "lno_sto_c_synthetic_data.png"
    fig.savefig(output_figure, dpi=200)

    print(f"Bragg estimate: {bragg_estimate:.4f} deg")
    print(f"Reflectivity peak: {peak_angle:.4f} deg")
    print(f"Saved {output_data}")
    print(f"Saved {output_figure}")


if __name__ == "__main__":
    main()
