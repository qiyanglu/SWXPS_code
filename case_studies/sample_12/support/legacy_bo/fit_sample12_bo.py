"""Prepare and fit Sample#12 C/LNO/LNO-STO/STO experimental data with BO.

By default this script only prints the proposed fitting setup and saves
normalized preview plots. Use `--run-fit` to launch Bayesian
optimization after checking the setup.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from math import erf, sqrt
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swanx.diagnostics import (

    plot_best_fit,

    plot_fit_convergence,

    plot_stack_schematic,

    plot_surrogate_slices,

    save_fit_history_csv,

)

from swanx.fitting import (

    BayesianOptimizationSettings,

    FitParameter,

    FitSimulation,

    FittingProblem,

    JointObjective,

    ReflectivityData,

    RockingCurveData,

    evaluation_from_contributions,

    run_bayesian_optimization,

)

from swanx.imfp import imfp_from_file

from swanx.preprocessing import subtract_edge_polynomial_background

from swanx.stack import (

    LayerTemplate,

    StackTemplate,

    SuperlatticeTemplate,

)

from swanx.workflows.simulate import CoreLevelRequest


PHOTON_ENERGY_EV = 815.0
SUPERLATTICE_REPEATS = 40
CASE_DIR = Path(__file__).resolve().parents[2]
REFLECTIVITY_FILE = CASE_DIR / "Reflectivity_Exp.dat"
RC_FILE = CASE_DIR / "ExpRCs.dat"
REFLECTIVITY_BO_HISTORY = CASE_DIR / "sample12_reflectivity_bo_history.csv"
LNO_OPC_FILE = "data/OPC/LaNiO3_800-900eV.dat"
STO_OPC_FILE = "data/OPC/SrTiO3_800-900eV.dat"
C_OPC_FILE = "data/OPC/C.dat"

REFLECTIVITY_START_DEG = 10.0
REFLECTIVITY_STOP_DEG = 15.45
RC_START_DEG = 11.72
RC_STOP_DEG = 14.08
RC_NAMES = ("C 1s", "Ni 3p", "La 4d")
BINDING_ENERGIES = {
    "C 1s": 285.0,
    "Ni 3p": 70.0,
    "La 4d": 105.0,
}

INITIAL_VALUES = {
    "carbon_thickness": 10.0,
    "carbon_roughness": 2.0,
    "top_lno_thickness": 200.0,
    "top_lno_signal_thickness": 50.0,
    "top_lno_buried_thickness": 150.0,
    "sto_thickness": 15.6,
    "lno_thickness": 17.2,
    "sto_thickness_start": 15.0,
    "lno_thickness_start": 16.5,
    "sto_thickness_delta": 1.0,
    "lno_thickness_delta": 1.0,
    "thickness_transition_repeat": 20.0,
    "thickness_transition_width": 8.0,
    "cap_roughness": 3.0,
    "sto_roughness": 3.0,
    "lno_roughness": 3.0,
    "sto_roughness_first": 3.0,
    "sto_roughness_last": 3.0,
    "lno_roughness_first": 3.0,
    "lno_roughness_last": 3.0,
    "substrate_roughness": 3.0,
    "reflectivity_angle_offset": 0.0,
    "rc_angle_offset": 0.0,
}

PARAMETERS = (
    FitParameter("carbon_thickness", 5.0, 15.0, "Angstrom", initial=10.0),
    FitParameter("carbon_roughness", 1.0, 3.0, "Angstrom", initial=2.0),
    FitParameter("top_lno_thickness", 150.0, 250.0, "Angstrom", initial=200.0),
    FitParameter("top_lno_signal_thickness", 30.0, 80.0, "Angstrom", initial=50.0),
    FitParameter("top_lno_buried_thickness", 100.0, 200.0, "Angstrom", initial=150.0),
    FitParameter("sto_thickness", 14.0, 17.5, "Angstrom", initial=15.6),
    FitParameter("lno_thickness", 15.0, 19.5, "Angstrom", initial=17.2),
    FitParameter("sto_thickness_start", 13.0, 18.5, "Angstrom", initial=15.0),
    FitParameter("lno_thickness_start", 13.0, 18.5, "Angstrom", initial=16.5),
    FitParameter("sto_thickness_delta", 0.0, 3.0, "Angstrom", initial=1.0),
    FitParameter("lno_thickness_delta", 0.0, 3.0, "Angstrom", initial=1.0),
    FitParameter("thickness_transition_repeat", 0.0, 39.0, "repeat", initial=20.0),
    FitParameter("thickness_transition_width", 1.0, 20.0, "repeat", initial=8.0),
    FitParameter("cap_roughness", 1.0, 5.0, "Angstrom", initial=3.0),
    FitParameter("sto_roughness", 1.0, 5.0, "Angstrom", initial=3.0),
    FitParameter("lno_roughness", 1.0, 5.0, "Angstrom", initial=3.0),
    FitParameter("sto_roughness_first", 2.0, 5.0, "Angstrom", initial=3.0),
    FitParameter("sto_roughness_last", 2.0, 5.0, "Angstrom", initial=3.0),
    FitParameter("lno_roughness_first", 2.0, 5.0, "Angstrom", initial=3.0),
    FitParameter("lno_roughness_last", 2.0, 5.0, "Angstrom", initial=3.0),
    FitParameter("substrate_roughness", 1.0, 5.0, "Angstrom", initial=3.0),
    FitParameter("reflectivity_angle_offset", -0.30, 0.30, "deg", initial=0.0),
    FitParameter("rc_angle_offset", -0.30, 0.30, "deg", initial=0.0),
)
PARAMETER_BY_NAME = {parameter.name: parameter for parameter in PARAMETERS}
REFLECTIVITY_PARAMETERS = tuple(
    PARAMETER_BY_NAME[name]
    for name in (
        "top_lno_thickness",
        "sto_thickness_start",
        "lno_thickness_start",
        "sto_thickness_delta",
        "lno_thickness_delta",
        "thickness_transition_repeat",
        "thickness_transition_width",
        "cap_roughness",
        "sto_roughness_first",
        "sto_roughness_last",
        "lno_roughness_first",
        "lno_roughness_last",
        "substrate_roughness",
        "reflectivity_angle_offset",
    )
)
RC_PARAMETERS = tuple(
    PARAMETER_BY_NAME[name]
    for name in (
        "carbon_thickness",
        "carbon_roughness",
        "sto_thickness_start",
        "lno_thickness_start",
        "sto_thickness_delta",
        "lno_thickness_delta",
        "thickness_transition_repeat",
        "thickness_transition_width",
        "cap_roughness",
        "sto_roughness_first",
        "sto_roughness_last",
        "lno_roughness_first",
        "lno_roughness_last",
        "substrate_roughness",
        "rc_angle_offset",
    )
)

DATASET_WEIGHTS = {
    "reflectivity": 1.0,
    "C 1s": 0.5,
    "Ni 3p": 3.0,
    "La 4d": 3.0,
}


@dataclass(frozen=True)
class PreparedData:
    """Experimental data after RC background subtraction and normalization."""

    reflectivity_angle: np.ndarray
    reflectivity_raw: np.ndarray
    rc_angle: np.ndarray
    rc_raw: dict[str, np.ndarray]
    rc_background: dict[str, np.ndarray]
    rc_corrected: dict[str, np.ndarray]
    rc_normalized: dict[str, np.ndarray]


@dataclass(frozen=True)
class CombinedSampleProblem:
    """Joint reflectivity/RC problem with independent angle offsets."""

    parameters: tuple[FitParameter, ...]
    reflectivity_problem: FittingProblem
    rc_problem: FittingProblem

    @property
    def reflectivity(self):
        return self.reflectivity_problem.reflectivity

    @property
    def rocking_curves(self):
        return self.rc_problem.rocking_curves

    def objective(self) -> JointObjective:
        return JointObjective(self.parameters, self.evaluate)

    def evaluate(self, values: dict[str, float]):
        reflectivity_evaluation = self.reflectivity_problem.evaluate(values)
        rc_evaluation = self.rc_problem.evaluate(values)
        timings = _combine_timings(
            reflectivity_evaluation.timings,
            rc_evaluation.timings,
        )
        return evaluation_from_contributions(
            values,
            (*reflectivity_evaluation.contributions, *rc_evaluation.contributions),
            timings=timings,
        )

    def simulate(self, values: dict[str, float]) -> FitSimulation:
        reflectivity_simulation = self.reflectivity_problem.simulate(values)
        rc_simulation = self.rc_problem.simulate(values)
        return FitSimulation(
            parameters=dict(values),
            stack=rc_simulation.stack,
            reflectivity=reflectivity_simulation.reflectivity,
            rocking_curves=rc_simulation.rocking_curves,
        )


def sample12_superlattice_template() -> SuperlatticeTemplate:
    """Return the corrected Sample#12 STO/LNO superlattice template."""

    return SuperlatticeTemplate(
        repeats=SUPERLATTICE_REPEATS,
        period=(
            LayerTemplate.from_file(
                "STO",
                STO_OPC_FILE,
                "sto_thickness",
                "sto_roughness",
            ),
            LayerTemplate.from_file(
                "LNO",
                LNO_OPC_FILE,
                "lno_thickness",
                "lno_roughness",
            ),
        ),
    )


def build_reflectivity_stack(values: dict[str, float]):
    """Build the Sample#12 reflectivity stack from fit parameters."""

    parts = (
        LayerTemplate.vacuum(),
        LayerTemplate.from_file(
            "LNO",
            LNO_OPC_FILE,
            "top_lno_thickness",
            "cap_roughness",
        ),
        *sample12_graded_superlattice_templates(values),
        LayerTemplate.from_file(
            "STO",
            STO_OPC_FILE,
            0.0,
            "substrate_roughness",
        ),
    )
    return StackTemplate(
        energy_ev=PHOTON_ENERGY_EV,
        base_dir=REPO_ROOT,
        parts=parts,
    ).build(values)


def build_rc_stack(values: dict[str, float]):
    """Build the Sample#12 RC stack from fit parameters."""

    top_lno_signal_thickness = values["top_lno_signal_thickness"]
    top_lno_buried_thickness = values["top_lno_thickness"] - top_lno_signal_thickness
    if top_lno_buried_thickness < 0.0:
        raise ValueError("top LNO signal slab cannot exceed total cap thickness")
    values = {
        **values,
        "top_lno_buried_thickness": top_lno_buried_thickness,
    }
    parts = (
        LayerTemplate.vacuum(),
        LayerTemplate.from_file(
            "C",
            C_OPC_FILE,
            "carbon_thickness",
            "carbon_roughness",
        ),
        LayerTemplate.from_file(
            "LNO",
            LNO_OPC_FILE,
            top_lno_signal_thickness,
            "cap_roughness",
        ),
        LayerTemplate.from_file(
            "LNO",
            LNO_OPC_FILE,
            top_lno_buried_thickness,
            0.0,
        ),
        *sample12_graded_superlattice_templates(values),
        LayerTemplate.from_file(
            "STO",
            STO_OPC_FILE,
            0.0,
            "substrate_roughness",
        ),
    )
    return StackTemplate(
        energy_ev=PHOTON_ENERGY_EV,
        base_dir=REPO_ROOT,
        parts=parts,
    ).build(values)


def load_best_reflectivity_values(
    path: Path = REFLECTIVITY_BO_HISTORY,
) -> dict[str, float]:
    """Return default values updated with the best reflectivity BO history row."""

    values = dict(INITIAL_VALUES)
    if not path.exists():
        return values
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return values
    best = min(rows, key=lambda row: float(row["objective"]))
    for name in values:
        if name in best and best[name] != "":
            values[name] = float(best[name])
    values["top_lno_signal_thickness"] = 50.0
    values["top_lno_buried_thickness"] = (
        values["top_lno_thickness"] - values["top_lno_signal_thickness"]
    )
    values["carbon_thickness"] = 10.0
    values["carbon_roughness"] = 2.0
    values["rc_angle_offset"] = 0.0
    return values


def initial_values_for_mode(fit_mode: str) -> dict[str, float]:
    """Return the initial/fixed values used to create a fitting problem."""

    if fit_mode in {"reflectivity", "rc"}:
        return load_best_reflectivity_values()
    return dict(INITIAL_VALUES)


def parameters_with_initial_values(
    parameters: tuple[FitParameter, ...],
    values: dict[str, float],
) -> tuple[FitParameter, ...]:
    """Return parameter definitions with initials replaced from a value dict."""

    return tuple(
        replace(parameter, initial=values.get(parameter.name, parameter.initial))
        for parameter in parameters
    )


def sample12_graded_superlattice_templates(values: dict[str, float]) -> tuple[LayerTemplate, ...]:
    """Return explicit STO/LNO layers with repeat-dependent thickness and roughness."""

    sto_thickness_start = values["sto_thickness_start"]
    lno_thickness_start = values["lno_thickness_start"]
    sto_thickness_end = sto_thickness_start + values["sto_thickness_delta"]
    lno_thickness_end = lno_thickness_start + values["lno_thickness_delta"]
    parts: list[LayerTemplate] = []
    for repeat_index in range(SUPERLATTICE_REPEATS):
        sto_thickness = _transitioned_layer_value(
            repeat_index,
            sto_thickness_start,
            sto_thickness_end,
            values["thickness_transition_repeat"],
            values["thickness_transition_width"],
        )
        lno_thickness = _transitioned_layer_value(
            repeat_index,
            lno_thickness_start,
            lno_thickness_end,
            values["thickness_transition_repeat"],
            values["thickness_transition_width"],
        )
        sto_roughness = _interpolated_roughness(
            repeat_index,
            values["sto_roughness_first"],
            values["sto_roughness_last"],
        )
        lno_roughness = _interpolated_roughness(
            repeat_index,
            values["lno_roughness_first"],
            values["lno_roughness_last"],
        )
        parts.extend(
            (
                LayerTemplate.from_file(
                    "STO",
                    STO_OPC_FILE,
                    sto_thickness,
                    sto_roughness,
                ),
                LayerTemplate.from_file(
                    "LNO",
                    LNO_OPC_FILE,
                    lno_thickness,
                    lno_roughness,
                ),
            )
        )
    return tuple(parts)


def _transitioned_layer_value(
    repeat_index: int,
    start_value: float,
    end_value: float,
    transition_repeat: float,
    transition_width: float,
) -> float:
    fraction = 0.5 * (
        1.0
        + erf((repeat_index - transition_repeat) / (sqrt(2.0) * transition_width))
    )
    return (1.0 - fraction) * start_value + fraction * end_value


def _interpolated_roughness(
    repeat_index: int,
    first_roughness: float,
    last_roughness: float,
) -> float:
    if SUPERLATTICE_REPEATS == 1:
        return first_roughness
    fraction = repeat_index / (SUPERLATTICE_REPEATS - 1)
    return (1.0 - fraction) * first_roughness + fraction * last_roughness


def core_level_requests() -> tuple[CoreLevelRequest, ...]:
    """Return Sample#12 C 1s, Ni 3p, and La 4d core-level requests."""

    imfp_files = {
        "C": REPO_ROOT / "data" / "IMFP" / "C.ANG",
        "LNO": REPO_ROOT / "data" / "IMFP" / "LNO.ANG",
        "STO": REPO_ROOT / "data" / "IMFP" / "STO.ANG",
    }
    imfp_by_core = {}
    for core_name, binding_energy in BINDING_ENERGIES.items():
        kinetic_energy = PHOTON_ENERGY_EV - binding_energy
        imfp_by_core[core_name] = {
            material: imfp_from_file(path, kinetic_energy)
            for material, path in imfp_files.items()
        }

    return (
        CoreLevelRequest(
            name="C 1s",
            binding_energy_ev=BINDING_ENERGIES["C 1s"],
            concentration_by_material={"C": 1.0},
            imfp_by_material={"vacuum": imfp_by_core["C 1s"]["C"], **imfp_by_core["C 1s"]},
            emitting_layer_indices=(1,),
        ),
        CoreLevelRequest(
            name="Ni 3p",
            binding_energy_ev=BINDING_ENERGIES["Ni 3p"],
            concentration_by_material={"LNO": 1.0},
            imfp_by_material={"vacuum": imfp_by_core["Ni 3p"]["C"], **imfp_by_core["Ni 3p"]},
            emitting_layer_indices=(2,),
        ),
        CoreLevelRequest(
            name="La 4d",
            binding_energy_ev=BINDING_ENERGIES["La 4d"],
            concentration_by_material={"LNO": 1.0},
            imfp_by_material={"vacuum": imfp_by_core["La 4d"]["C"], **imfp_by_core["La 4d"]},
            emitting_layer_indices=(2,),
        ),
    )


def load_and_prepare_data(background_percent: float, background_order: int) -> PreparedData:
    """Load raw experimental data and build preview/fitting curves."""

    reflectivity_raw = np.loadtxt(REFLECTIVITY_FILE)
    rc_raw_array = np.loadtxt(RC_FILE)

    reflectivity_angle = _angle_grid(
        REFLECTIVITY_START_DEG,
        REFLECTIVITY_STOP_DEG,
        len(reflectivity_raw),
    )
    rc_angle = _angle_grid(RC_START_DEG, RC_STOP_DEG, len(rc_raw_array))

    rc_raw = {
        name: np.asarray(rc_raw_array[:, index], dtype=float)
        for index, name in enumerate(RC_NAMES)
    }
    corrections = {
        name: subtract_edge_polynomial_background(
            rc_angle,
            values,
            edge_fraction=background_percent,
            order=background_order,
        )
        for name, values in rc_raw.items()
    }
    rc_background = {name: correction.background for name, correction in corrections.items()}
    rc_corrected = {name: correction.corrected for name, correction in corrections.items()}
    rc_normalized = {
        name: correction.normalized
        for name, correction in corrections.items()
    }
    return PreparedData(
        reflectivity_angle=reflectivity_angle,
        reflectivity_raw=reflectivity_raw,
        rc_angle=rc_angle,
        rc_raw=rc_raw,
        rc_background=rc_background,
        rc_corrected=rc_corrected,
        rc_normalized=rc_normalized,
    )


def apply_reflectivity_window(
    data: PreparedData,
    min_angle: float | None,
    max_angle: float | None,
) -> PreparedData:
    """Return prepared data with reflectivity restricted to an angle window."""

    if min_angle is None and max_angle is None:
        return data
    lower = -np.inf if min_angle is None else float(min_angle)
    upper = np.inf if max_angle is None else float(max_angle)
    if lower > upper:
        raise ValueError("reflectivity minimum angle cannot exceed maximum angle")
    mask = (data.reflectivity_angle >= lower) & (data.reflectivity_angle <= upper)
    if not np.any(mask):
        raise ValueError("reflectivity angle window does not include any data points")
    return PreparedData(
        reflectivity_angle=data.reflectivity_angle[mask],
        reflectivity_raw=data.reflectivity_raw[mask],
        rc_angle=data.rc_angle,
        rc_raw=data.rc_raw,
        rc_background=data.rc_background,
        rc_corrected=data.rc_corrected,
        rc_normalized=data.rc_normalized,
    )


def make_fit_problem(
    data: PreparedData,
    fit_mode: str = "joint",
    initial_values: dict[str, float] | None = None,
) -> FittingProblem | CombinedSampleProblem:
    """Create the Sample#12 fitting problem from prepared curves."""

    if fit_mode not in {"joint", "reflectivity", "rc"}:
        raise ValueError("fit_mode must be 'joint', 'reflectivity', or 'rc'")
    initial_values = (
        initial_values_for_mode(fit_mode)
        if initial_values is None
        else dict(initial_values)
    )

    rc_data = tuple(
        RockingCurveData(
            name,
            data.rc_angle,
            data.rc_normalized[name],
            weight=DATASET_WEIGHTS[name],
        )
        for name in RC_NAMES
    )
    rc_problem = FittingProblem(
        parameters=parameters_with_initial_values(RC_PARAMETERS, initial_values),
        stack_builder=build_rc_stack,
        photon_energy_ev=PHOTON_ENERGY_EV,
        rocking_curves=rc_data,
        core_levels=core_level_requests(),
        angle_offset_parameter="rc_angle_offset",
        field_step=5.0,
        roughness_step=2.0,
        slicing=None,
        offpeak_mask=np.ones_like(data.rc_angle, dtype=bool),
        fixed_values=initial_values,
    )
    if fit_mode == "rc":
        return rc_problem
    reflectivity_problem = FittingProblem(
        parameters=parameters_with_initial_values(REFLECTIVITY_PARAMETERS, initial_values),
        stack_builder=build_reflectivity_stack,
        photon_energy_ev=PHOTON_ENERGY_EV,
        reflectivity=ReflectivityData(
            name="reflectivity",
            angles=data.reflectivity_angle,
            reflectivity=data.reflectivity_raw,
            weight=DATASET_WEIGHTS["reflectivity"],
            log_floor=1.0e-12,
        ),
        angle_offset_parameter="reflectivity_angle_offset",
        field_step=5.0,
        roughness_step=2.0,
        slicing=None,
        fixed_values=initial_values,
    )
    if fit_mode == "reflectivity":
        return reflectivity_problem
    return CombinedSampleProblem(
        parameters=parameters_with_initial_values(PARAMETERS, initial_values),
        reflectivity_problem=reflectivity_problem,
        rc_problem=rc_problem,
    )


def print_setup(
    background_percent: float,
    background_order: int,
    fit_mode: str,
    initial_values: dict[str, float],
    reflectivity_min_angle: float | None,
    reflectivity_max_angle: float | None,
    data: PreparedData,
) -> None:
    """Print the proposed stack, preprocessing, and fitting ranges."""

    print("Sample#12 proposed fitting setup")
    print(f"Photon energy: {PHOTON_ENERGY_EV:g} eV")
    print(f"RC background edge percentage: {background_percent:g}% at each edge")
    print(f"RC background polynomial order: {background_order}")
    print(f"Fit mode: {fit_mode}")
    if reflectivity_min_angle is not None or reflectivity_max_angle is not None:
        min_text = "-inf" if reflectivity_min_angle is None else f"{reflectivity_min_angle:g}"
        max_text = "inf" if reflectivity_max_angle is None else f"{reflectivity_max_angle:g}"
        print(f"Reflectivity fitting window: {min_text} to {max_text} deg")
    if fit_mode == "rc" and REFLECTIVITY_BO_HISTORY.exists():
        print(f"RC initial values seeded from {REFLECTIVITY_BO_HISTORY.name}")
    print()
    print("Stack model, top to bottom:")
    if fit_mode == "reflectivity":
        print_reflectivity_stack_model()
    elif fit_mode == "rc":
        print_rc_stack_model()
    else:
        print("  Reflectivity stack:")
        print_reflectivity_stack_model(indent="    ")
        print("  RC stack:")
        print_rc_stack_model(indent="    ")
    print()
    print("Parameter ranges:")
    for parameter in parameters_with_initial_values(
        parameters_for_mode(fit_mode),
        initial_values,
    ):
        unit = f" {parameter.unit}" if parameter.unit else ""
        print(
            f"  {parameter.name}: "
            f"{parameter.lower:g} to {parameter.upper:g}{unit}, "
            f"initial={parameter.initial:g}{unit}"
        )
    print()
    print("Experimental data:")
    print(
        f"  Reflectivity used for fitting: {data.reflectivity_angle[0]:g} to "
        f"{data.reflectivity_angle[-1]:g} deg, {len(data.reflectivity_angle)} points"
    )
    print(
        f"  RCs: {RC_START_DEG:g} to {RC_STOP_DEG:g} deg, "
        f"{len(np.loadtxt(RC_FILE))} points, columns={', '.join(RC_NAMES)}"
    )
    print()
    print("Preprocessing:")
    print("  Reflectivity: raw measured values are used directly; no background subtraction.")
    print("  RCs: divide by polynomial background fitted to both curve edges.")
    print("  Angle offsets: reflectivity and RCs are fitted with independent offset parameters.")
    print("  RC emission layers: C 1s from C layer only; Ni 3p/La 4d from the top LNO slab only.")


def print_reflectivity_stack_model(indent: str = "  ") -> None:
    """Print the reflectivity stack model."""

    print(f"{indent}vacuum")
    print(f"{indent}LNO cap: thickness=top_lno_thickness, roughness=cap_roughness")
    print(f"{indent}[STO/LNO] x {SUPERLATTICE_REPEATS}, graded:")
    print(f"{indent}  STO thickness transitions from sto_thickness_start to start + sto_thickness_delta")
    print(f"{indent}  LNO thickness transitions from lno_thickness_start to start + lno_thickness_delta")
    print(f"{indent}  erf-like transition controlled by thickness_transition_repeat and thickness_transition_width")
    print(f"{indent}  STO roughness linearly interpolates from sto_roughness_first to sto_roughness_last")
    print(f"{indent}  LNO roughness linearly interpolates from lno_roughness_first to lno_roughness_last")
    print(f"{indent}STO substrate: semi-infinite, roughness=substrate_roughness")


def print_rc_stack_model(indent: str = "  ") -> None:
    """Print the RC stack model."""

    print(f"{indent}vacuum")
    print(f"{indent}C: thickness=carbon_thickness, roughness=carbon_roughness")
    print(f"{indent}LNO cap signal slab: fixed thickness=top_lno_signal_thickness, roughness=cap_roughness")
    print(f"{indent}LNO cap buried slab: thickness=top_lno_thickness - top_lno_signal_thickness, roughness=0")
    print(f"{indent}[STO/LNO] x {SUPERLATTICE_REPEATS}, graded:")
    print(f"{indent}  STO thickness transitions from sto_thickness_start to start + sto_thickness_delta")
    print(f"{indent}  LNO thickness transitions from lno_thickness_start to start + lno_thickness_delta")
    print(f"{indent}  erf-like transition controlled by thickness_transition_repeat and thickness_transition_width")
    print(f"{indent}  STO roughness linearly interpolates from sto_roughness_first to sto_roughness_last")
    print(f"{indent}  LNO roughness linearly interpolates from lno_roughness_first to lno_roughness_last")
    print(f"{indent}STO substrate: semi-infinite, roughness=substrate_roughness")


def parameters_for_mode(fit_mode: str) -> tuple[FitParameter, ...]:
    """Return the active fitting parameters for the requested fit target."""

    if fit_mode == "joint":
        return PARAMETERS
    if fit_mode == "reflectivity":
        return REFLECTIVITY_PARAMETERS
    if fit_mode == "rc":
        return RC_PARAMETERS
    raise ValueError("fit_mode must be 'joint', 'reflectivity', or 'rc'")


def save_preview_plot(data: PreparedData, path: Path) -> None:
    """Save raw and normalized experimental curves for inspection."""

    plt = _load_pyplot()
    fig, axes = plt.subplots(4, 1, figsize=(8.0, 9.0), sharex=False)

    ax = axes[0]
    ax.semilogy(data.reflectivity_angle, data.reflectivity_raw, "o-", ms=3, label="raw")
    ax.set_ylabel("Reflectivity")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)

    for ax, name in zip(axes[1:], RC_NAMES):
        ax.plot(data.rc_angle, data.rc_raw[name], "o", ms=3, alpha=0.45, label="raw")
        ax.plot(data.rc_angle, data.rc_background[name], "-", color="0.35", label="background")
        ax2 = ax.twinx()
        ax2.plot(
            data.rc_angle,
            data.rc_normalized[name],
            "-",
            color="black",
            linewidth=1.5,
            label="normalized",
        )
        ax.set_ylabel(f"{name} raw")
        ax2.set_ylabel(f"{name} norm.")
        ax.grid(True, alpha=0.25)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)

    axes[-1].set_xlabel("Grazing incidence angle (deg)")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_superlattice_profile_plot(values: dict[str, float], path: Path) -> None:
    """Save repeat-by-repeat superlattice thickness and roughness profiles."""

    plt = _load_pyplot()
    repeat_numbers = np.arange(1, SUPERLATTICE_REPEATS + 1)
    repeat_indices = repeat_numbers - 1

    sto_thickness = np.array(
        [
            _transitioned_layer_value(
                int(index),
                values["sto_thickness_start"],
                values["sto_thickness_start"] + values["sto_thickness_delta"],
                values["thickness_transition_repeat"],
                values["thickness_transition_width"],
            )
            for index in repeat_indices
        ]
    )
    lno_thickness = np.array(
        [
            _transitioned_layer_value(
                int(index),
                values["lno_thickness_start"],
                values["lno_thickness_start"] + values["lno_thickness_delta"],
                values["thickness_transition_repeat"],
                values["thickness_transition_width"],
            )
            for index in repeat_indices
        ]
    )
    period = sto_thickness + lno_thickness
    sto_roughness = np.array(
        [
            _interpolated_roughness(
                int(index),
                values["sto_roughness_first"],
                values["sto_roughness_last"],
            )
            for index in repeat_indices
        ]
    )
    lno_roughness = np.array(
        [
            _interpolated_roughness(
                int(index),
                values["lno_roughness_first"],
                values["lno_roughness_last"],
            )
            for index in repeat_indices
        ]
    )

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.8), sharex=True)

    thickness_ax = axes[0]
    thickness_ax.plot(repeat_numbers, sto_thickness, "o-", ms=3, label="STO")
    thickness_ax.plot(repeat_numbers, lno_thickness, "s-", ms=3, label="LNO")
    thickness_ax.set_ylabel("Layer thickness (A)")
    thickness_ax.grid(True, alpha=0.25)
    thickness_ax.legend(loc="upper left")

    period_ax = thickness_ax.twinx()
    period_ax.plot(
        repeat_numbers,
        period,
        "--",
        color="0.35",
        linewidth=1.6,
        label="period",
    )
    period_ax.set_ylabel("Period thickness (A)")
    period_ax.legend(loc="upper right")

    roughness_ax = axes[1]
    roughness_ax.plot(repeat_numbers, sto_roughness, "o-", ms=3, label="STO")
    roughness_ax.plot(repeat_numbers, lno_roughness, "s-", ms=3, label="LNO")
    roughness_ax.set_xlabel("Superlattice repeat number")
    roughness_ax.set_ylabel("Roughness (A)")
    roughness_ax.grid(True, alpha=0.25)
    roughness_ax.legend(loc="best")

    fig.suptitle("Sample#12 Superlattice Profile")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_fit(
    data: PreparedData,
    fit_mode: str,
    initial_values: dict[str, float],
    n_calls: int,
    n_initial_points: int,
    random_state: int,
    show_progress: bool,
    progress_interval: int,
) -> None:
    """Run BO fitting and save standard diagnostics."""

    problem = make_fit_problem(
        data,
        fit_mode=fit_mode,
        initial_values=initial_values,
    )
    output_prefix = {
        "joint": "sample12_bo",
        "reflectivity": "sample12_reflectivity_bo",
        "rc": "sample12_rc_bo",
    }[fit_mode]
    result = run_bayesian_optimization(
        problem.objective(),
        BayesianOptimizationSettings(
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            random_state=random_state,
            show_progress=show_progress,
            progress_interval=progress_interval,
        ),
    )
    best_simulation = problem.simulate(result.best_parameters)
    save_fit_history_csv(
        CASE_DIR / f"{output_prefix}_history.csv",
        result.history,
        problem.parameters,
    )
    plot_fit_convergence(CASE_DIR / f"{output_prefix}_convergence.png", result.history)
    plot_best_fit(
        CASE_DIR / f"{output_prefix}_best_fit.png",
        problem.reflectivity,
        problem.rocking_curves,
        best_simulation,
    )
    plot_surrogate_slices(
        CASE_DIR / f"{output_prefix}_surrogate_slices.png",
        result,
        problem.parameters,
        initial_values,
    )
    plot_stack_schematic(
        CASE_DIR / f"{output_prefix}_stack_schematic.png",
        best_simulation.stack,
        title=f"Sample#12 {fit_mode.title()} BO Stack",
        top_layers=5,
        bottom_layers=3,
    )
    if fit_mode == "reflectivity":
        save_superlattice_profile_plot(
            result.best_parameters,
            CASE_DIR / f"{output_prefix}_superlattice_profile.png",
        )
    if fit_mode == "rc":
        save_superlattice_profile_plot(
            {**initial_values, **result.best_parameters},
            CASE_DIR / f"{output_prefix}_superlattice_profile.png",
        )

    print(f"Best objective: {result.best_objective:.6g}")
    print("Best parameters:")
    for parameter in problem.parameters:
        value = result.best_parameters[parameter.name]
        unit = f" {parameter.unit}" if parameter.unit else ""
        print(f"  {parameter.name}: {value:.6g}{unit}")
    print_timing_summary(result)


def print_timing_summary(result) -> None:
    """Print optimizer and forward-model timing diagnostics."""

    print("Timing summary:")
    print(f"  total optimizer wall time: {result.timing.total_seconds:.3f} s")
    print(f"  objective evaluations: {result.timing.evaluations}")
    print(f"  objective/forward time: {result.timing.objective_seconds:.3f} s")
    print(f"  BO/GP overhead time: {result.timing.optimizer_overhead_seconds:.3f} s")
    if not result.history.evaluations:
        return
    timing_totals: dict[str, float] = {}
    for evaluation in result.history.evaluations:
        for name, value in evaluation.timings.items():
            timing_totals[name] = timing_totals.get(name, 0.0) + float(value)
    if not timing_totals:
        return
    print("  accumulated objective substeps:")
    for name, value in sorted(timing_totals.items()):
        print(f"    {name}: {value:.3f} s")


def _combine_timings(*timings: dict[str, float]) -> dict[str, float]:
    combined: dict[str, float] = {}
    for timing in timings:
        for name, value in timing.items():
            combined[name] = combined.get(name, 0.0) + float(value)
    return combined


def _angle_grid(start: float, stop: float, count: int) -> np.ndarray:
    return np.linspace(start, stop, count)


def _load_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("matplotlib is required for Sample#12 preview plots") from error
    return plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--background-percent",
        type=float,
        default=10.0,
        help="Percentage of points at each RC edge used for polynomial background fitting.",
    )
    parser.add_argument(
        "--background-order",
        type=int,
        default=2,
        help="Polynomial order for RC edge-background fitting.",
    )
    parser.add_argument("--no-reflectivity", action="store_true")
    parser.add_argument(
        "--fit-mode",
        choices=("joint", "reflectivity", "rc"),
        default="joint",
        help="Which data to fit: joint RC+reflectivity, reflectivity only, or RC only.",
    )
    parser.add_argument("--run-fit", action="store_true")
    parser.add_argument("--n-calls", type=int, default=80)
    parser.add_argument("--n-initial-points", type=int, default=24)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument(
        "--reflectivity-min-angle",
        type=float,
        default=None,
        help="Minimum reflectivity angle included in fitting.",
    )
    parser.add_argument(
        "--reflectivity-max-angle",
        type=float,
        default=None,
        help="Maximum reflectivity angle included in fitting.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print one-line BO progress updates during optimization.",
    )
    parser.add_argument("--progress-interval", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fit_mode = "rc" if args.no_reflectivity else args.fit_mode
    initial_values = initial_values_for_mode(fit_mode)
    data = load_and_prepare_data(args.background_percent, args.background_order)
    data = apply_reflectivity_window(
        data,
        args.reflectivity_min_angle,
        args.reflectivity_max_angle,
    )
    print_setup(
        args.background_percent,
        args.background_order,
        fit_mode,
        initial_values,
        args.reflectivity_min_angle,
        args.reflectivity_max_angle,
        data,
    )
    preview_path = CASE_DIR / "sample12_normalized_preview.png"
    save_preview_plot(data, preview_path)
    initial_stack = (
        build_reflectivity_stack(initial_values)
        if fit_mode == "reflectivity"
        else build_rc_stack(initial_values)
    )
    stack_path = CASE_DIR / "sample12_initial_stack_schematic.png"
    plot_stack_schematic(
        stack_path,
        initial_stack,
        title="Sample#12 Initial Stack",
        top_layers=5,
        bottom_layers=3,
    )
    print(f"Saved {preview_path}")
    print(f"Saved {stack_path}")

    if not args.run_fit:
        print("BO fitting was not run. Re-run with --run-fit after checking the preview.")
        return
    run_fit(
        data,
        fit_mode=fit_mode,
        initial_values=initial_values,
        n_calls=args.n_calls,
        n_initial_points=args.n_initial_points,
        random_state=args.random_state,
        show_progress=args.progress,
        progress_interval=args.progress_interval,
    )


if __name__ == "__main__":
    main()

