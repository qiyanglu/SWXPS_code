"""Benchmark the experimental JAX Parratt backend against NumPy.

Run from the repository root with:

    python benchmarks/performance/benchmark_jax_reflectivity.py
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from swxps import (
    CoreLevelRequest,
    Layer,
    ReflectivityRequest,
    RockingCurveRequest,
    SimulationStack,
    StackLayer,
    parratt_reflectivity,
    simulate_reflectivity,
    simulate_rocking_curves,
    vacuum,
)
from swxps.fields import depth_grid, transfer_matrix_electric_field_profiles
from swxps.xps import nominal_layer_index_at_depth, normalized_rocking_curve
from swxps.reflectivity_jax import (
    jitted_electric_field_intensity,
    jitted_normalized_rocking_curve,
    jitted_parratt_reflectivity,
    jitted_value_and_grad_rocking_curve_loss,
    jitted_value_and_grad_reflectivity_loss,
    layer_arrays_from_layers,
)
from swxps.simulation_jax import simulate_reflectivity_jax, simulate_rocking_curves_jax


def main() -> None:
    energy_ev = 8000.0
    angles = np.linspace(0.2, 5.0, 512)
    layers = [vacuum()]
    for _ in range(20):
        layers.append(Layer(thickness=20.0, delta=8.0e-6, beta=1.0e-7))
        layers.append(Layer(thickness=20.0, delta=3.0e-6, beta=7.0e-8))
    layers.append(Layer(thickness=0.0, delta=7.0e-6, beta=1.5e-7))
    stack = SimulationStack(
        tuple(
            StackLayer(
                material="vacuum" if index == 0 else ("substrate" if index == len(layers) - 1 else f"L{index % 2}"),
                thickness=layer.thickness,
                delta=layer.delta,
                beta=layer.beta,
                roughness=layer.roughness,
            )
            for index, layer in enumerate(layers)
        )
    )

    thicknesses, deltas, betas, roughnesses = layer_arrays_from_layers(layers)
    target = parratt_reflectivity(angles, energy_ev, layers)
    depths, layer_indices = depth_grid(layers, step=2.0)
    sampled_layers = nominal_layer_index_at_depth(layers, depths)
    concentration_by_layer = np.asarray([0.0] + [1.0, 0.25] * 20 + [0.0])
    imfp_by_layer = np.asarray([20.0] + [20.0, 30.0] * 20 + [30.0])
    concentration = concentration_by_layer[sampled_layers]
    attenuation_length = imfp_by_layer[sampled_layers]
    offpeak_mask = np.ones(angles.shape, dtype=bool)
    rc_target = normalized_rocking_curve(
        angles=angles,
        energy_ev=energy_ev,
        layers=layers,
        concentration_by_layer=concentration_by_layer,
        imfp_by_layer=imfp_by_layer,
        field_step=2.0,
    ).intensity

    numpy_seconds = _time_once(lambda: parratt_reflectivity(angles, energy_ev, layers))
    first_jax_seconds = _time_once(
        lambda: np.asarray(
            jitted_parratt_reflectivity(
                angles,
                energy_ev,
                thicknesses,
                deltas,
                betas,
                roughnesses,
            )
        )
    )
    repeated_jax_seconds = min(
        _time_once(
            lambda: np.asarray(
                jitted_parratt_reflectivity(
                    angles,
                    energy_ev,
                    thicknesses,
                    deltas,
                    betas,
                    roughnesses,
                )
            )
        )
        for _ in range(5)
    )
    value_grad_seconds = min(
        _time_once(
            lambda: jitted_value_and_grad_reflectivity_loss(
                thicknesses,
                angles,
                energy_ev,
                deltas,
                betas,
                roughnesses,
                target,
            )
        )
        for _ in range(5)
    )
    numpy_field_seconds = _time_once(
        lambda: transfer_matrix_electric_field_profiles(
            angles,
            energy_ev,
            layers,
            step=2.0,
        )
    )
    first_jax_field_seconds = _time_once(
        lambda: np.asarray(
            jitted_electric_field_intensity(
                angles,
                energy_ev,
                thicknesses,
                deltas,
                betas,
                roughnesses,
                depths,
                layer_indices,
            )
        )
    )
    repeated_jax_field_seconds = min(
        _time_once(
            lambda: np.asarray(
                jitted_electric_field_intensity(
                    angles,
                    energy_ev,
                    thicknesses,
                    deltas,
                    betas,
                    roughnesses,
                    depths,
                    layer_indices,
                )
            )
        )
        for _ in range(5)
    )
    numpy_rc_seconds = _time_once(
        lambda: normalized_rocking_curve(
            angles=angles,
            energy_ev=energy_ev,
            layers=layers,
            concentration_by_layer=concentration_by_layer,
            imfp_by_layer=imfp_by_layer,
            field_step=2.0,
        )
    )
    first_jax_rc_seconds = _time_once(
        lambda: jitted_normalized_rocking_curve(
            angles,
            energy_ev,
            thicknesses,
            deltas,
            betas,
            roughnesses,
            depths,
            layer_indices,
            concentration,
            attenuation_length,
            0.0,
            offpeak_mask,
        )
    )
    repeated_jax_rc_seconds = min(
        _time_once(
            lambda: jitted_normalized_rocking_curve(
                angles,
                energy_ev,
                thicknesses,
                deltas,
                betas,
                roughnesses,
                depths,
                layer_indices,
                concentration,
                attenuation_length,
                0.0,
                offpeak_mask,
            )
        )
        for _ in range(5)
    )
    value_grad_rc_seconds = min(
        _time_once(
            lambda: jitted_value_and_grad_rocking_curve_loss(
                thicknesses,
                angles,
                energy_ev,
                deltas,
                betas,
                roughnesses,
                depths,
                layer_indices,
                concentration,
                attenuation_length,
                0.0,
                offpeak_mask,
                rc_target,
            )
        )
        for _ in range(5)
    )
    reflectivity_request = ReflectivityRequest(
        angles=angles,
        energy_ev=energy_ev,
        stack=stack,
    )
    core_levels = (
        CoreLevelRequest(
            name="L1",
            binding_energy_ev=100.0,
            concentration_by_material={"L1": 1.0},
            imfp_by_material={
                "vacuum": 20.0,
                "L0": 20.0,
                "L1": 30.0,
                "substrate": 30.0,
            },
        ),
        CoreLevelRequest(
            name="L0",
            binding_energy_ev=200.0,
            concentration_by_material={"L0": 1.0},
            imfp_by_material={
                "vacuum": 20.0,
                "L0": 20.0,
                "L1": 30.0,
                "substrate": 30.0,
            },
        ),
    )
    rc_request = RockingCurveRequest(
        angles=angles,
        photon_energy_ev=energy_ev,
        stack=stack,
        core_levels=core_levels,
        field_step=2.0,
    )
    numpy_high_level_reflectivity_seconds = _time_once(
        lambda: simulate_reflectivity(reflectivity_request)
    )
    first_jax_high_level_reflectivity_seconds = _time_once(
        lambda: simulate_reflectivity_jax(reflectivity_request)
    )
    repeated_jax_high_level_reflectivity_seconds = min(
        _time_once(lambda: simulate_reflectivity_jax(reflectivity_request))
        for _ in range(5)
    )
    numpy_high_level_rc_seconds = _time_once(lambda: simulate_rocking_curves(rc_request))
    first_jax_high_level_rc_seconds = _time_once(
        lambda: simulate_rocking_curves_jax(rc_request)
    )
    repeated_jax_high_level_rc_seconds = min(
        _time_once(lambda: simulate_rocking_curves_jax(rc_request))
        for _ in range(5)
    )

    jax_result = np.asarray(
        jitted_parratt_reflectivity(
            angles,
            energy_ev,
            thicknesses,
            deltas,
            betas,
            roughnesses,
        )
    )
    max_difference = float(np.max(np.abs(jax_result - target)))
    jax_rc, _, _ = jitted_normalized_rocking_curve(
        angles,
        energy_ev,
        thicknesses,
        deltas,
        betas,
        roughnesses,
        depths,
        layer_indices,
        concentration,
        attenuation_length,
        0.0,
        offpeak_mask,
    )
    max_rc_difference = float(np.max(np.abs(np.asarray(jax_rc) - rc_target)))

    print(f"NumPy Parratt: {numpy_seconds:.6f} s")
    print(f"First JAX JIT call, including compile: {first_jax_seconds:.6f} s")
    print(f"Best repeated JAX JIT call: {repeated_jax_seconds:.6f} s")
    print(f"Best JAX value-and-gradient call: {value_grad_seconds:.6f} s")
    print(f"Max |JAX - NumPy|: {max_difference:.3e}")
    print(f"NumPy field profiles: {numpy_field_seconds:.6f} s")
    print(f"First JAX field JIT call, including compile: {first_jax_field_seconds:.6f} s")
    print(f"Best repeated JAX field call: {repeated_jax_field_seconds:.6f} s")
    print(f"NumPy normalized RC: {numpy_rc_seconds:.6f} s")
    print(f"First JAX RC JIT call, including compile: {first_jax_rc_seconds:.6f} s")
    print(f"Best repeated JAX RC call: {repeated_jax_rc_seconds:.6f} s")
    print(f"Best JAX RC value-and-gradient call: {value_grad_rc_seconds:.6f} s")
    print(f"Max |JAX RC - NumPy RC|: {max_rc_difference:.3e}")
    print(f"NumPy high-level reflectivity: {numpy_high_level_reflectivity_seconds:.6f} s")
    print(
        "First JAX high-level reflectivity, including compile: "
        f"{first_jax_high_level_reflectivity_seconds:.6f} s"
    )
    print(
        "Best repeated JAX high-level reflectivity: "
        f"{repeated_jax_high_level_reflectivity_seconds:.6f} s"
    )
    print(f"NumPy high-level two-core RC: {numpy_high_level_rc_seconds:.6f} s")
    print(
        "First JAX high-level two-core RC, including compile: "
        f"{first_jax_high_level_rc_seconds:.6f} s"
    )
    print(
        "Best repeated JAX high-level two-core RC: "
        f"{repeated_jax_high_level_rc_seconds:.6f} s"
    )


def _time_once(callback) -> float:
    start = perf_counter()
    callback()
    return perf_counter() - start


if __name__ == "__main__":
    main()
