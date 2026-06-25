"""Small physics and shape benchmark for the unified slicing strategy."""

from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from swanx.stack import (

    LayerSlicingPolicy,

    SimulationStack,

    StackLayer,

    fixed_layer_grid,

    fixed_layer_grid_plan,

)

from swanx.workflows.simulate import (

    CoreLevelRequest,

    ReflectivityRequest,

    RockingCurveRequest,

    simulate_reflectivity,

    simulate_rocking_curves,

)
from swanx.fields import transfer_matrix_reflectivity_array  # noqa: E402


def make_stack(
    surface_thickness: float = 4.0,
    film_thickness: float = 16.0,
) -> SimulationStack:
    return SimulationStack(
        (
            StackLayer("vacuum", 0.0),
            StackLayer(
                "surface",
                surface_thickness,
                delta=7.0e-6,
                beta=1.0e-7,
                roughness=1.0,
            ),
            StackLayer(
                "film",
                film_thickness,
                delta=2.5e-6,
                beta=8.0e-8,
                roughness=1.5,
            ),
            StackLayer(
                "substrate",
                0.0,
                delta=1.0e-5,
                beta=2.0e-7,
                roughness=2.0,
            ),
        )
    )


def run_small_physics_case() -> dict[str, float]:
    """Return convergence, shape, and repeated-evaluation measurements."""

    angles = np.linspace(0.6, 5.0, 61)
    policy = LayerSlicingPolicy(min_slices=10, max_slice_thickness=2.0)
    capacity = make_stack(surface_thickness=6.0)
    plan = fixed_layer_grid_plan(capacity.optical_layers, policy)
    stack = make_stack()

    start = perf_counter()
    unified = simulate_reflectivity(
        ReflectivityRequest(
            angles=angles,
            energy_ev=3000.0,
            stack=stack,
            slicing=plan,
        )
    ).reflectivity
    unified_seconds = perf_counter() - start

    start = perf_counter()
    fine = transfer_matrix_reflectivity_array(
        angles,
        3000.0,
        stack.optical_layers,
        roughness_step=0.1,
    )
    fine_seconds = perf_counter() - start

    core = CoreLevelRequest(
        name="surface",
        binding_energy_ev=100.0,
        concentration_by_material={"surface": 1.0},
        imfp_by_material={
            "vacuum": 20.0,
            "surface": 20.0,
            "film": 25.0,
            "substrate": 25.0,
        },
    )
    curve = simulate_rocking_curves(
        RockingCurveRequest(
            angles=angles,
            photon_energy_ev=3000.0,
            stack=stack,
            core_levels=(core,),
            slicing=plan,
        )
    ).core_levels[0].curve

    sweep_shapes = set()
    start = perf_counter()
    for thickness in np.linspace(2.0, 6.0, 17):
        trial = make_stack(float(thickness))
        grid = fixed_layer_grid(trial.optical_layers, plan)
        sweep_shapes.add(grid.centers.shape)
        simulate_reflectivity(
            ReflectivityRequest(
                angles=angles,
                energy_ev=3000.0,
                stack=trial,
                slicing=plan,
            )
        )
    sweep_seconds = perf_counter() - start

    thick_stack = make_stack(film_thickness=160.0)
    thick_grid = fixed_layer_grid_plan(thick_stack.optical_layers, policy)
    start = perf_counter()
    thick_result = simulate_reflectivity(
        ReflectivityRequest(
            angles=angles,
            energy_ev=3000.0,
            stack=thick_stack,
            slicing=thick_grid,
        )
    ).reflectivity
    thick_seconds = perf_counter() - start

    return {
        "cell_count": float(sum(plan.slice_counts)),
        "unique_sweep_shapes": float(len(sweep_shapes)),
        "thick_160A_cell_count": float(sum(thick_grid.slice_counts)),
        "thick_160A_reflectivity_seconds": thick_seconds,
        "thick_160A_all_finite": float(np.all(np.isfinite(thick_result))),
        "max_reflectivity_absolute_error": float(np.max(np.abs(unified - fine))),
        "max_reflectivity_relative_error": float(
            np.max(np.abs(unified - fine) / np.maximum(fine, 1.0e-12))
        ),
        "unified_reflectivity_seconds": unified_seconds,
        "fine_0p1A_reflectivity_seconds": fine_seconds,
        "thickness_sweep_seconds": sweep_seconds,
        "rc_min": float(np.min(curve.intensity)),
        "rc_max": float(np.max(curve.intensity)),
        "rc_all_finite": float(np.all(np.isfinite(curve.intensity))),
    }


def main() -> None:
    results = run_small_physics_case()
    print("Unified-grid thin-surface physics case")
    for name, value in results.items():
        print(f"{name:40s} {value:.8g}")


if __name__ == "__main__":
    main()
