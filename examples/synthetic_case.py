"""Shared synthetic C/LNO/STO case used by the examples.

The teaching examples intentionally reuse the same structure as the benchmark
case: vacuum / C / [LaNiO3 / SrTiO3]x20 / SrTiO3 substrate at 1000 eV.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swanx.fitting import FittingProblem, ReflectivityData, RockingCurveData
from swanx.imfp import imfp_from_file
from swanx.optics import energy_to_wavelength
from swanx.stack import LayerTemplate, StackTemplate, SuperlatticeTemplate
from swanx.workflows import (
    CoreLevelRequest,
    ReflectivityRequest,
    RockingCurveRequest,
    simulate_reflectivity,
    simulate_rocking_curves,
)

PHOTON_ENERGY_EV = 1000.0
SUPERLATTICE_REPEATS = 20
DATA_FILE = REPO_ROOT / "benchmarks" / "synthetic_c_lno_sto" / "lno_sto_c_synthetic_data.csv"

TRUE_VALUES = {
    "carbon_thickness": 10.0,
    "carbon_roughness_fraction": 1.0 / 3.0,
    "lno_thickness": 20.0,
    "sto_thickness": 20.0,
    "superlattice_roughness": 3.0,
    "substrate_roughness": 3.0,
    "angle_offset": 0.0,
}

RC_COLUMN_BY_NAME = {
    "La 4d": "la4d_rc",
    "O 1s": "o1s_rc",
    "Ti 2p": "ti2p_rc",
    "C 1s": "c1s_rc",
}


def carbon_roughness_from_values(values: dict[str, float]) -> float:
    """Map the fitted carbon roughness fraction to an Angstrom value."""

    max_roughness = min(5.0, values["carbon_thickness"])
    return 1.0 + values["carbon_roughness_fraction"] * (max_roughness - 1.0)


def stack_values(values: dict[str, float] | None = None) -> dict[str, float]:
    """Return stack parameters including the derived carbon roughness."""

    resolved = dict(TRUE_VALUES if values is None else values)
    resolved["carbon_roughness"] = carbon_roughness_from_values(resolved)
    return resolved


def stack_template(carbon_roughness: float | str = "carbon_roughness") -> StackTemplate:
    """Return the canonical vacuum/C/[LNO/STO]x20/STO template."""

    return StackTemplate(
        energy_ev=PHOTON_ENERGY_EV,
        base_dir=REPO_ROOT,
        parts=(
            LayerTemplate.vacuum(),
            LayerTemplate.from_file(
                "C",
                "data/OPC/C.dat",
                thickness="carbon_thickness",
                roughness=carbon_roughness,
            ),
            SuperlatticeTemplate(
                repeats=SUPERLATTICE_REPEATS,
                period=(
                    LayerTemplate.from_file(
                        "LNO",
                        "data/OPC/LaNiO3.dat",
                        thickness="lno_thickness",
                        roughness="superlattice_roughness",
                    ),
                    LayerTemplate.from_file(
                        "STO",
                        "data/OPC/SrTiO3.dat",
                        thickness="sto_thickness",
                        roughness="superlattice_roughness",
                    ),
                ),
            ),
            LayerTemplate.from_file(
                "STO",
                "data/OPC/SrTiO3.dat",
                thickness=0.0,
                roughness="substrate_roughness",
            ),
        ),
    )


def build_stack(values: dict[str, float] | None = None):
    """Build the canonical synthetic simulation stack."""

    return stack_template().build(stack_values(values))


def core_level_requests() -> tuple[CoreLevelRequest, ...]:
    """Return material-selective La, O, Ti, and C core-level requests."""

    binding_energies = {
        "La 4d": 105.0,
        "O 1s": 530.0,
        "Ti 2p": 460.0,
        "C 1s": 285.0,
    }
    imfp_files = {
        "C": REPO_ROOT / "data" / "IMFP" / "C.ANG",
        "LNO": REPO_ROOT / "data" / "IMFP" / "LNO.ANG",
        "STO": REPO_ROOT / "data" / "IMFP" / "STO.ANG",
    }
    imfp_by_core = {}
    for core_name, binding_energy in binding_energies.items():
        kinetic_energy = PHOTON_ENERGY_EV - binding_energy
        imfp_by_core[core_name] = {
            material: imfp_from_file(path, kinetic_energy)
            for material, path in imfp_files.items()
        }

    def imfp_map(core_name: str) -> dict[str, float]:
        return {
            "vacuum": imfp_by_core[core_name]["C"],
            **imfp_by_core[core_name],
        }

    return (
        CoreLevelRequest(
            name="La 4d",
            binding_energy_ev=binding_energies["La 4d"],
            concentration_by_material={"LNO": 1.0},
            imfp_by_material=imfp_map("La 4d"),
        ),
        CoreLevelRequest(
            name="O 1s",
            binding_energy_ev=binding_energies["O 1s"],
            concentration_by_material={"LNO": 1.0, "STO": 1.0},
            imfp_by_material=imfp_map("O 1s"),
        ),
        CoreLevelRequest(
            name="Ti 2p",
            binding_energy_ev=binding_energies["Ti 2p"],
            concentration_by_material={"STO": 1.0},
            imfp_by_material=imfp_map("Ti 2p"),
        ),
        CoreLevelRequest(
            name="C 1s",
            binding_energy_ev=binding_energies["C 1s"],
            concentration_by_material={"C": 1.0},
            imfp_by_material=imfp_map("C 1s"),
        ),
    )


def bragg_angle_deg(values: dict[str, float] | None = None) -> float:
    """Return the first-order Bragg estimate for the LNO/STO period."""

    resolved = stack_values(values)
    period = resolved["lno_thickness"] + resolved["sto_thickness"]
    wavelength = energy_to_wavelength(PHOTON_ENERGY_EV)
    return float(np.rad2deg(np.arcsin(wavelength / (2.0 * period))))


def angles(count: int = 161) -> np.ndarray:
    """Return the benchmark angle grid around the first Bragg peak."""

    center = bragg_angle_deg()
    return np.linspace(center - 2.0, center + 2.0, count)


def load_synthetic_data(path: Path = DATA_FILE, stride: int = 1) -> dict[str, np.ndarray]:
    """Load the benchmark synthetic CSV, optionally downsampling by stride."""

    if stride <= 0:
        raise ValueError("stride must be positive")
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {name: np.asarray(data[name][::stride], dtype=float) for name in data.dtype.names}


def simulate_case(values: dict[str, float] | None = None, angle_grid: np.ndarray | None = None):
    """Simulate reflectivity and four SW-XPS rocking curves for the case."""

    stack = build_stack(values)
    scan_angles = angles() if angle_grid is None else np.asarray(angle_grid, dtype=float)
    reflectivity = simulate_reflectivity(
        ReflectivityRequest(
            angles=scan_angles,
            energy_ev=PHOTON_ENERGY_EV,
            stack=stack,
            roughness_step=1.0,
            slicing=None,
        )
    )
    peak_angle = scan_angles[np.argmax(reflectivity.reflectivity)]
    offpeak_mask = np.abs(scan_angles - peak_angle) > 1.25
    rocking_curves = simulate_rocking_curves(
        RockingCurveRequest(
            angles=scan_angles,
            photon_energy_ev=PHOTON_ENERGY_EV,
            stack=stack,
            core_levels=core_level_requests(),
            field_step=1.0,
            roughness_step=1.0,
            offpeak_mask=offpeak_mask,
            slicing=None,
        )
    )
    return stack, reflectivity, rocking_curves


def make_data_problem(data: dict[str, np.ndarray] | None = None) -> FittingProblem:
    """Build a no-parameter fitting problem for overlaying the synthetic data."""

    loaded = load_synthetic_data() if data is None else data
    scan_angles = loaded["angle_deg"]
    peak_angle = scan_angles[np.argmax(loaded["reflectivity"])]
    offpeak_mask = np.abs(scan_angles - peak_angle) > 1.25
    return FittingProblem(
        parameters=(),
        stack_builder=lambda values: build_stack(),
        photon_energy_ev=PHOTON_ENERGY_EV,
        reflectivity=ReflectivityData(
            name="synthetic C/LNO/STO reflectivity",
            angles=scan_angles,
            reflectivity=loaded["reflectivity"],
            log_floor=1.0e-12,
        ),
        rocking_curves=tuple(
            RockingCurveData(
                name=name,
                angles=scan_angles,
                intensity=loaded[column],
            )
            for name, column in RC_COLUMN_BY_NAME.items()
        ),
        core_levels=core_level_requests(),
        field_step=1.0,
        roughness_step=1.0,
        slicing=None,
        offpeak_mask=offpeak_mask,
    )
