"""Simulate Sample#13 using an external BestTemplate*.par stack."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Literal

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = CASE_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

from swanx.imfp import imfp_from_file

from swanx.optical_constants import constants_from_file

from swanx.stack import (

    SimulationStack,

    StackLayer,

)

from swanx.workflows.simulate import (

    CoreLevelRequest,

    ReflectivityRequest,

    RockingCurveRequest,

    simulate_reflectivity,

    simulate_rocking_curves,

)

import fit_sample13_bo as sample13  # noqa: E402


DEFAULT_PAR = Path(
    r"C:\Users\luqy0\OneDrive - 西湖大学\SWXPS_Past_Data"
    r"\LNO-STO_Standing Wave XPS\LNO-STO-LNO sample #13"
    r"\RC_r20_total_La\results1\BestTemplate02.par"
)


@dataclass(frozen=True)
class ParsedParStack:
    """Layer data parsed from the old program's `.par` file."""

    materials: tuple[str, ...]
    opc_files: tuple[str, ...]
    thicknesses: tuple[float, ...]
    roughnesses: tuple[float, ...]


def parse_besttemplate_stack(path: Path) -> ParsedParStack:
    """Parse material, OPC, thickness, and roughness rows from a BestTemplate file."""

    rows = [_tokens(line) for line in path.read_text(encoding="utf-8").splitlines()]
    materials = tuple(rows[7])
    opc_files = tuple(rows[8])
    thickness_tokens = rows[11]
    roughnesses = tuple(float(value) for value in rows[13])

    if len(thickness_tokens) == len(materials) + 1:
        # The first thickness entry belongs to the vacuum/header in this
        # template. The final Inf entry is the semi-infinite substrate.
        thickness_tokens = thickness_tokens[1:]
    if thickness_tokens[-1].lower() == "inf":
        materials = materials[:-1]
        opc_files = opc_files[:-1]
        roughnesses = roughnesses[:-1]
        thickness_tokens = thickness_tokens[:-1]
    thicknesses = tuple(float(value) for value in thickness_tokens)
    if not (len(materials) == len(opc_files) == len(thicknesses) == len(roughnesses)):
        raise ValueError(
            f"parsed {path.name} rows have inconsistent lengths: "
            f"materials={len(materials)}, opc={len(opc_files)}, "
            f"thickness={len(thicknesses)}, roughness={len(roughnesses)}"
        )
    return ParsedParStack(
        materials=materials,
        opc_files=opc_files,
        thicknesses=thicknesses,
        roughnesses=roughnesses,
    )


def build_stack(parsed: ParsedParStack) -> SimulationStack:
    """Build a package SimulationStack from the parsed old-program stack."""

    layers = [StackLayer("vacuum", 0.0, 0.0, 0.0, 0.0)]
    for material, opc_file, thickness, roughness in zip(
        parsed.materials,
        parsed.opc_files,
        parsed.thicknesses,
        parsed.roughnesses,
    ):
        mapped_material = _material_name(material)
        opc_path = _opc_path(mapped_material, opc_file)
        delta, beta = constants_from_file(opc_path, sample13.PHOTON_ENERGY_EV)
        layers.append(
            StackLayer(
                mapped_material,
                float(thickness),
                delta,
                beta,
                float(roughness),
            )
        )

    substrate_delta, substrate_beta = constants_from_file(
        REPO_ROOT / sample13.STO_OPC_FILE,
        sample13.PHOTON_ENERGY_EV,
    )
    layers.append(StackLayer("STO", 0.0, substrate_delta, substrate_beta, 0.0))
    return SimulationStack(tuple(layers))


def core_level_requests(stack: SimulationStack, emission_mode: str) -> tuple[CoreLevelRequest, ...]:
    """Return C/Ni/La requests for either all-layer or near-surface emission."""

    imfp_by_core = {}
    for core_name, binding_energy in sample13.BINDING_ENERGIES.items():
        kinetic_energy = sample13.PHOTON_ENERGY_EV - binding_energy
        imfp_by_core[core_name] = {
            "C": imfp_from_file(REPO_ROOT / "examples" / "data" / "IMFP" / "C.ANG", kinetic_energy),
            "LNO": imfp_from_file(REPO_ROOT / "examples" / "data" / "IMFP" / "LNO.ANG", kinetic_energy),
            "STO": imfp_from_file(REPO_ROOT / "examples" / "data" / "IMFP" / "STO.ANG", kinetic_energy),
        }

    c_layers = _layer_indices(stack, "C")
    lno_layers = _layer_indices(stack, "LNO")
    if not c_layers or not lno_layers:
        raise ValueError("BestTemplate stack must contain C and LNO layers")
    if emission_mode == "selected":
        c_emit = (c_layers[0],)
        lno_emit = (lno_layers[0],)
    elif emission_mode == "all":
        c_emit = tuple(c_layers)
        lno_emit = tuple(lno_layers)
    else:
        raise ValueError("emission_mode must be 'selected' or 'all'")

    return (
        CoreLevelRequest(
            name="C 1s",
            binding_energy_ev=sample13.BINDING_ENERGIES["C 1s"],
            concentration_by_material={"C": 1.0},
            imfp_by_material={"vacuum": imfp_by_core["C 1s"]["C"], **imfp_by_core["C 1s"]},
            emitting_layer_indices=c_emit,
        ),
        CoreLevelRequest(
            name="Ni 3p",
            binding_energy_ev=sample13.BINDING_ENERGIES["Ni 3p"],
            concentration_by_material={"LNO": 1.0},
            imfp_by_material={"vacuum": imfp_by_core["Ni 3p"]["C"], **imfp_by_core["Ni 3p"]},
            emitting_layer_indices=lno_emit,
        ),
        CoreLevelRequest(
            name="La 4d",
            binding_energy_ev=sample13.BINDING_ENERGIES["La 4d"],
            concentration_by_material={"LNO": 1.0},
            imfp_by_material={"vacuum": imfp_by_core["La 4d"]["C"], **imfp_by_core["La 4d"]},
            emitting_layer_indices=lno_emit,
        ),
    )


def simulate_and_plot(
    par_path: Path,
    emission_mode: str,
    reflectivity_angle_offset: float,
    rc_angle_offset: float,
    background_percent: float,
    background_order: int,
    field_step: float,
    rc_roughness_profile: Literal["erf", "linear"],
) -> None:
    """Run reflectivity and RC simulations and save overlay plots."""

    label = _template_label(par_path)
    data = sample13.load_and_prepare_data(background_percent, background_order)
    stack = build_stack(parse_besttemplate_stack(par_path))

    reflectivity = simulate_reflectivity(
        ReflectivityRequest(
            angles=data.reflectivity_angle,
            energy_ev=sample13.PHOTON_ENERGY_EV,
            stack=stack,
            angle_offset=reflectivity_angle_offset,
            roughness_step=2.0,
            slicing=None,
        )
    )
    rocking_curves = simulate_rocking_curves(
        RockingCurveRequest(
            angles=data.rc_angle,
            photon_energy_ev=sample13.PHOTON_ENERGY_EV,
            stack=stack,
            core_levels=core_level_requests(stack, emission_mode),
            angle_offset=rc_angle_offset,
            field_step=field_step,
            roughness_step=2.0,
            roughness_profile=rc_roughness_profile,
            offpeak_mask=np.ones_like(data.rc_angle, dtype=bool),
            slicing=None,
        )
    )

    reflectivity_output = CASE_DIR / f"sample13_{label}_reflectivity.png"
    rc_output = CASE_DIR / f"sample13_{label}_rcs_{emission_mode}.png"
    plot_reflectivity(reflectivity_output, data, reflectivity, label)
    plot_rcs(rc_output, data, rocking_curves, emission_mode, label)
    print_stack_summary(stack, label)
    print(f"Saved {reflectivity_output}")
    print(f"Saved {rc_output}")


def plot_reflectivity(path: Path, data: sample13.PreparedData, reflectivity, label: str) -> None:
    """Save reflectivity overlay."""

    plt = sample13._load_pyplot()
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    ax.semilogy(data.reflectivity_angle, data.reflectivity_raw, "o", markersize=3, label="experiment")
    ax.semilogy(reflectivity.angle, reflectivity.reflectivity, color="black", linewidth=1.5, label=label)
    ax.set_xlabel("Grazing incidence angle (deg)")
    ax.set_ylabel("Reflectivity")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_rcs(path: Path, data: sample13.PreparedData, rocking_curves, emission_mode: str, label: str) -> None:
    """Save RC overlays."""

    plt = sample13._load_pyplot()
    simulated = {
        core.name: core.curve.intensity
        for core in rocking_curves.core_levels
    }
    colors = {"C 1s": "tab:green", "Ni 3p": "tab:blue", "La 4d": "tab:orange"}
    fig, axes = plt.subplots(len(sample13.RC_NAMES), 1, figsize=(7.6, 6.4), sharex=True)
    axes = np.asarray(axes).ravel()
    for ax, name in zip(axes, sample13.RC_NAMES):
        ax.plot(data.rc_angle, data.rc_normalized[name], "o", color=colors[name], markersize=3, alpha=0.55, label="experiment")
        ax.plot(data.rc_angle, simulated[name], color="black", linewidth=1.5, label=label)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Grazing incidence angle (deg)")
    fig.suptitle(f"{label} RCs ({emission_mode} emission)")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def print_stack_summary(stack: SimulationStack, label: str) -> None:
    """Print a compact summary of the parsed stack."""

    finite = stack.layers[1:-1]
    print(f"{label} stack summary:")
    print(f"  finite layers: {len(finite)}")
    print(f"  top layers:")
    for index, layer in enumerate(finite[:6], start=1):
        print(
            f"    {index}: {layer.material}, thickness={layer.thickness:.2f} A, "
            f"roughness={layer.roughness:.2f} A"
        )
    print(f"  bottom finite layer: {finite[-1].material}, thickness={finite[-1].thickness:.2f} A")


def _tokens(line: str) -> list[str]:
    return [token for token in line.split("\t") if token != ""]


def _template_label(path: Path) -> str:
    return path.stem.lower()


def _material_name(material: str) -> str:
    if material == "CO":
        return "C"
    if material in {"LNO", "STO"}:
        return material
    raise ValueError(f"unsupported BestTemplate material {material!r}")


def _opc_path(material: str, opc_file: str) -> Path:
    if material == "C":
        return REPO_ROOT / sample13.C_OPC_FILE
    if material == "LNO":
        return REPO_ROOT / sample13.LNO_OPC_FILE
    if material == "STO":
        return REPO_ROOT / sample13.STO_OPC_FILE
    raise ValueError(f"unsupported material {material!r} for OPC mapping from {opc_file!r}")


def _layer_indices(stack: SimulationStack, material: str) -> list[int]:
    return [
        index
        for index, layer in enumerate(stack.layers)
        if layer.material == material
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--par", type=Path, default=DEFAULT_PAR)
    parser.add_argument(
        "--emission-mode",
        choices=("selected", "all"),
        default="selected",
        help="Use only first C/LNO emitting layers, or all material-matching layers.",
    )
    parser.add_argument("--reflectivity-angle-offset", type=float, default=0.0)
    parser.add_argument("--rc-angle-offset", type=float, default=0.0)
    parser.add_argument("--background-percent", type=float, default=10.0)
    parser.add_argument("--background-order", type=int, default=2)
    parser.add_argument("--field-step", type=float, default=1.0)
    parser.add_argument(
        "--rc-roughness-profile",
        choices=("erf", "linear"),
        default="linear",
        help="Roughness profile for RC field/concentration sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simulate_and_plot(
        args.par,
        args.emission_mode,
        args.reflectivity_angle_offset,
        args.rc_angle_offset,
        args.background_percent,
        args.background_order,
        args.field_step,
        args.rc_roughness_profile,
    )


if __name__ == "__main__":
    main()
