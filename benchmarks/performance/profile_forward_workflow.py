"""Profile the maintained NumPy forward-model and fitting workflow.

Run from the repository root with::

    python benchmarks/performance/profile_forward_workflow.py

The stages are intentionally reported separately.  This keeps future
optimization work evidence-based without turning the benchmark into another
case-study runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from swanx.fields import (  # noqa: E402
    effective_layers_with_roughness,
    transfer_matrix_reflectivity_array,
)
from swanx.fitting import (  # noqa: E402
    FitParameter,
    FittingProblem,
    ReflectivityData,
    RockingCurveData,
)
from swanx.imfp import clear_imfp_cache, load_imfp  # noqa: E402
from swanx.optical_constants import (  # noqa: E402
    clear_optical_constants_cache,
    load_optical_constants,
)
from swanx.simulation import (  # noqa: E402
    CoreLevelRequest,
    ReflectivityRequest,
    RockingCurveRequest,
    simulate_reflectivity,
    simulate_rocking_curves,
)
from swanx.stack_builders import (  # noqa: E402
    LayerTemplate,
    StackTemplate,
    SuperlatticeTemplate,
)

ENERGY_EV = 1000.0
OPTICAL_PATHS = (
    ROOT / "examples" / "data" / "OPC" / "C.dat",
    ROOT / "examples" / "data" / "OPC" / "LaNiO3.dat",
    ROOT / "examples" / "data" / "OPC" / "SrTiO3.dat",
)
IMFP_PATHS = (
    ROOT / "examples" / "data" / "IMFP" / "C.ANG",
    ROOT / "examples" / "data" / "IMFP" / "LNO.ANG",
    ROOT / "examples" / "data" / "IMFP" / "STO.ANG",
)


def run_benchmark(repeats: int = 5, angle_count: int = 61) -> dict[str, float]:
    """Return best-of-repeat timings for one representative workflow."""

    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if angle_count < 3:
        raise ValueError("angle_count must be at least 3")

    angles = np.linspace(10.5, 16.5, angle_count)

    clear_optical_constants_cache()
    clear_imfp_cache()
    cold_table_seconds = _time_once(_load_static_tables)
    cached_table_seconds = _best_time(_load_static_tables, repeats)

    template = _stack_template()
    values = {"lno_thickness": 16.0, "sto_thickness": 16.0}
    stack_seconds = _best_time(lambda: template.build(values), repeats)
    stack = template.build(values)

    roughness_seconds = _best_time(
        lambda: effective_layers_with_roughness(stack.optical_layers, step=1.0),
        repeats,
    )
    effective_layers = effective_layers_with_roughness(stack.optical_layers, step=1.0)
    reflectivity_seconds = _best_time(
        lambda: transfer_matrix_reflectivity_array(
            angles,
            ENERGY_EV,
            effective_layers,
        ),
        repeats,
    )

    core_levels = (_la_4d_request(),)
    rocking_request = RockingCurveRequest(
        angles=angles,
        photon_energy_ev=ENERGY_EV,
        stack=stack,
        core_levels=core_levels,
        field_step=1.0,
        roughness_step=1.0,
        slicing=None,
    )
    fields_and_swanx_seconds = _best_time(
        lambda: simulate_rocking_curves(rocking_request),
        repeats,
    )

    problem = _fitting_problem(template, angles, stack, core_levels)
    objective_seconds = _best_time(
        lambda: problem.evaluate(values),
        repeats,
    )

    return {
        "table_load_cold": cold_table_seconds,
        "table_load_cached": cached_table_seconds,
        "stack_construction": stack_seconds,
        "roughness_discretization": roughness_seconds,
        "reflectivity_from_effective_stack": reflectivity_seconds,
        "fields_and_swanx_forward": fields_and_swanx_seconds,
        "full_fitting_objective": objective_seconds,
    }


def _stack_template() -> StackTemplate:
    return StackTemplate(
        energy_ev=ENERGY_EV,
        base_dir=ROOT,
        parts=(
            LayerTemplate.vacuum(),
            LayerTemplate.from_file("C", "examples/data/OPC/C.dat", 8.0, roughness=2.0),
            SuperlatticeTemplate(
                repeats=8,
                period=(
                    LayerTemplate.from_file(
                        "LNO",
                        "examples/data/OPC/LaNiO3.dat",
                        "lno_thickness",
                        roughness=3.0,
                    ),
                    LayerTemplate.from_file(
                        "STO",
                        "examples/data/OPC/SrTiO3.dat",
                        "sto_thickness",
                        roughness=3.0,
                    ),
                ),
            ),
            LayerTemplate.from_file(
                "STO",
                "examples/data/OPC/SrTiO3.dat",
                0.0,
                roughness=3.0,
            ),
        ),
    )


def _la_4d_request() -> CoreLevelRequest:
    kinetic_energy_ev = ENERGY_EV - 105.0
    imfp_by_material = {
        "vacuum": load_imfp(ROOT / "examples" / "data" / "IMFP" / "C.ANG").imfp_at(kinetic_energy_ev),
        "C": load_imfp(ROOT / "examples" / "data" / "IMFP" / "C.ANG").imfp_at(kinetic_energy_ev),
        "LNO": load_imfp(ROOT / "examples" / "data" / "IMFP" / "LNO.ANG").imfp_at(kinetic_energy_ev),
        "STO": load_imfp(ROOT / "examples" / "data" / "IMFP" / "STO.ANG").imfp_at(kinetic_energy_ev),
    }
    return CoreLevelRequest(
        name="La 4d",
        binding_energy_ev=105.0,
        concentration_by_material={"LNO": 1.0},
        imfp_by_material=imfp_by_material,
    )


def _fitting_problem(
    template: StackTemplate,
    angles: np.ndarray,
    stack,
    core_levels: tuple[CoreLevelRequest, ...],
) -> FittingProblem:
    reflectivity_target = simulate_reflectivity(
        ReflectivityRequest(
            angles=angles,
            energy_ev=ENERGY_EV,
            stack=stack,
            roughness_step=1.0,
            slicing=None,
        )
    )
    rocking_target = simulate_rocking_curves(
        RockingCurveRequest(
            angles=angles,
            photon_energy_ev=ENERGY_EV,
            stack=stack,
            core_levels=core_levels,
            field_step=1.0,
            roughness_step=1.0,
            slicing=None,
        )
    )
    return FittingProblem(
        parameters=(
            FitParameter("lno_thickness", 14.0, 18.0, unit="Angstrom", initial=16.0),
        ),
        fixed_values={"sto_thickness": 16.0},
        stack_builder=template.builder(),
        photon_energy_ev=ENERGY_EV,
        reflectivity=ReflectivityData(
            name="reflectivity",
            angles=angles,
            reflectivity=reflectivity_target.reflectivity,
        ),
        rocking_curves=(
            RockingCurveData(
                name="La 4d",
                angles=angles,
                intensity=rocking_target.core_levels[0].curve.intensity,
            ),
        ),
        core_levels=core_levels,
        angle_offset_parameter=None,
        field_step=1.0,
        roughness_step=1.0,
        slicing=None,
    )


def _load_static_tables() -> None:
    for path in OPTICAL_PATHS:
        load_optical_constants(path)
    for path in IMFP_PATHS:
        load_imfp(path)


def _time_once(callback: Callable[[], object]) -> float:
    start = perf_counter()
    callback()
    return perf_counter() - start


def _best_time(callback: Callable[[], object], repeats: int) -> float:
    return min(_time_once(callback) for _ in range(repeats))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--angles", type=int, default=61)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    timings = run_benchmark(repeats=args.repeats, angle_count=args.angles)
    if args.as_json:
        print(json.dumps(timings, indent=2, sort_keys=True))
        return

    print("Representative C/[LNO/STO]x8/STO NumPy workflow")
    print(f"Angles: {args.angles}; best of {args.repeats} timed runs")
    for stage, seconds in timings.items():
        print(f"{stage:36s} {seconds:10.6f} s")
    speedup = timings["table_load_cold"] / timings["table_load_cached"]
    print(f"{'table cache speedup':36s} {speedup:10.2f} x")


if __name__ == "__main__":
    main()
