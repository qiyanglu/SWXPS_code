"""Plot roughness-broadened concentration profiles for an LNO/STO stack."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swxps import (
    SimulationStack,
    StackLayer,
    imfp_from_file,
    layer_from_file,
    sample_concentration_profiles,
)


def make_lno_sto_stack(
    energy_ev: float,
    repeats: int,
    layer_thickness: float,
    roughness: float,
) -> SimulationStack:
    """Return vacuum / [LaNiO3 / SrTiO3]xN / SrTiO3 substrate."""

    lno_file = REPO_ROOT / "OPC" / "LaNiO3.dat"
    sto_file = REPO_ROOT / "OPC" / "SrTiO3.dat"

    stack_layers = [StackLayer("vacuum", thickness=0.0)]
    for _ in range(repeats):
        lno = layer_from_file(lno_file, energy_ev, thickness=layer_thickness, roughness=roughness)
        sto = layer_from_file(sto_file, energy_ev, thickness=layer_thickness, roughness=roughness)
        stack_layers.append(
            StackLayer("LNO", lno.thickness, lno.delta, lno.beta, lno.roughness)
        )
        stack_layers.append(
            StackLayer("STO", sto.thickness, sto.delta, sto.beta, sto.roughness)
        )

    substrate = layer_from_file(sto_file, energy_ev, thickness=0.0, roughness=roughness)
    stack_layers.append(
        StackLayer(
            "STO",
            substrate.thickness,
            substrate.delta,
            substrate.beta,
            substrate.roughness,
        )
    )
    return SimulationStack(tuple(stack_layers))


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


def main() -> None:
    energy_ev = 1000.0
    la4d_binding_energy_ev = 105.0
    o1s_binding_energy_ev = 530.0
    ti2p_binding_energy_ev = 460.0
    repeats = 20
    layer_thickness = 20.0
    roughness = 3.0
    max_depth = 100.0

    la4d_kinetic_energy_ev = energy_ev - la4d_binding_energy_ev
    o1s_kinetic_energy_ev = energy_ev - o1s_binding_energy_ev
    ti2p_kinetic_energy_ev = energy_ev - ti2p_binding_energy_ev
    la4d_imfp = imfp_from_file(REPO_ROOT / "IMFP" / "LNO.ANG", la4d_kinetic_energy_ev)
    ti2p_imfp = imfp_from_file(REPO_ROOT / "IMFP" / "STO.ANG", ti2p_kinetic_energy_ev)
    o1s_lno_imfp = imfp_from_file(REPO_ROOT / "IMFP" / "LNO.ANG", o1s_kinetic_energy_ev)
    o1s_sto_imfp = imfp_from_file(REPO_ROOT / "IMFP" / "STO.ANG", o1s_kinetic_energy_ev)

    stack = make_lno_sto_stack(
        energy_ev=energy_ev,
        repeats=repeats,
        layer_thickness=layer_thickness,
        roughness=roughness,
    )
    profiles = sample_concentration_profiles(
        stack,
        {
            "La": {"LNO": 1.0},
            "Ti": {"STO": 1.0},
            "O": {"LNO": 1.0, "STO": 1.0},
        },
        step=0.25,
    )

    fig, ax = plt.subplots(figsize=(5.6, 6.4))
    draw_layer_boundaries(ax, repeats, layer_thickness)
    ax.plot(profiles.profiles["La"], profiles.depth, color="tab:purple", linewidth=1.8, label="La")
    ax.plot(profiles.profiles["Ti"], profiles.depth, color="tab:orange", linewidth=1.8, label="Ti")
    ax.plot(profiles.profiles["O"], profiles.depth, color="tab:green", linewidth=1.4, label="O")
    ax.axhline(
        la4d_imfp,
        color="tab:purple",
        linestyle="--",
        linewidth=1.1,
        label=f"La 4d IMFP {la4d_imfp:.1f} A",
    )
    ax.axhline(
        ti2p_imfp,
        color="tab:orange",
        linestyle="--",
        linewidth=1.1,
        label=f"Ti 2p IMFP {ti2p_imfp:.1f} A",
    )
    ax.axhline(
        o1s_lno_imfp,
        color="tab:green",
        linestyle="--",
        linewidth=1.1,
        label=f"O 1s IMFP LNO {o1s_lno_imfp:.1f} A",
    )
    ax.axhline(
        o1s_sto_imfp,
        color="tab:green",
        linestyle=":",
        linewidth=1.4,
        label=f"O 1s IMFP STO {o1s_sto_imfp:.1f} A",
    )

    ax.set_xlabel("Relative concentration")
    ax.set_ylabel("Depth below surface (Angstrom)")
    ax.set_title("Top LNO/STO concentration profile and IMFP depths")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.0, max_depth)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    output_path = REPO_ROOT / "examples" / "lno_sto_stack_profile.png"
    fig.savefig(output_path, dpi=200)
    print(f"La 4d IMFP at {la4d_kinetic_energy_ev:.1f} eV: {la4d_imfp:.2f} Angstrom")
    print(f"Ti 2p IMFP at {ti2p_kinetic_energy_ev:.1f} eV: {ti2p_imfp:.2f} Angstrom")
    print(f"O 1s IMFP in LNO at {o1s_kinetic_energy_ev:.1f} eV: {o1s_lno_imfp:.2f} Angstrom")
    print(f"O 1s IMFP in STO at {o1s_kinetic_energy_ev:.1f} eV: {o1s_sto_imfp:.2f} Angstrom")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
