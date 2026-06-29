"""Run a small JAX least-squares fit on the synthetic C/LaNiO3/SrTiO3 stack geometry."""

from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from examples.synthetic_case import (  # noqa: E402
    PHOTON_ENERGY_EV,
    SUPERLATTICE_REPEATS,
    TRUE_VALUES,
    angles,
    build_stack,
)
from swanx.fitting import (  # noqa: E402
    FitParameter,
    ReflectivityData,
    JaxLeastSquaresOptimizerSettings,
    JaxLeastSquaresResidualSettings,
    build_jax_residual_function,
    optimize_with_jax_least_squares,
)
from swanx.reflectivity_jax import parratt_reflectivity_jax  # noqa: E402


def layer_arrays():
    """Return shared-layer Parratt arrays for the canonical synthetic stack."""

    layers = build_stack().optical_layers
    thicknesses = np.array([layer.thickness for layer in layers], dtype=float)
    deltas = np.array([layer.delta for layer in layers], dtype=float)
    betas = np.array([layer.beta for layer in layers], dtype=float)
    roughnesses = np.array([layer.roughness for layer in layers], dtype=float)
    lno_indices = np.arange(2, 2 + 2 * SUPERLATTICE_REPEATS, 2)
    sto_indices = np.arange(3, 3 + 2 * SUPERLATTICE_REPEATS, 2)
    return thicknesses, deltas, betas, roughnesses, lno_indices, sto_indices


def main() -> None:
    import jax.numpy as jnp

    scan_angles = jnp.asarray(angles(count=96))
    (
        true_thicknesses,
        deltas,
        betas,
        roughnesses,
        lno_indices,
        sto_indices,
    ) = layer_arrays()
    true_thicknesses_jax = jnp.asarray(true_thicknesses)
    deltas_jax = jnp.asarray(deltas)
    betas_jax = jnp.asarray(betas)
    roughnesses_jax = jnp.asarray(roughnesses)
    lno_indices_jax = jnp.asarray(lno_indices)
    sto_indices_jax = jnp.asarray(sto_indices)

    target = np.asarray(
        parratt_reflectivity_jax(
            scan_angles,
            PHOTON_ENERGY_EV,
            true_thicknesses_jax,
            deltas_jax,
            betas_jax,
            roughnesses_jax,
        )
    )
    data = ReflectivityData(
        name="synthetic C/LaNiO3/SrTiO3 reflectivity",
        angles=np.asarray(scan_angles),
        reflectivity=target,
        sigma=np.full(target.shape, 1.0e-5),
    )

    def simulate_curves(parameters):
        thicknesses = true_thicknesses_jax
        thicknesses = thicknesses.at[lno_indices_jax].set(parameters[0])
        thicknesses = thicknesses.at[sto_indices_jax].set(parameters[1])
        reflectivity = parratt_reflectivity_jax(
            scan_angles,
            PHOTON_ENERGY_EV,
            thicknesses,
            deltas_jax,
            betas_jax,
            roughnesses_jax,
        )
        return reflectivity, ()

    residual_function = build_jax_residual_function(
        simulate_curves,
        reflectivity=data,
        settings=JaxLeastSquaresResidualSettings(reflectivity_log=False),
    )
    parameters = (
        FitParameter("lno_thickness", 18.0, 22.0, "Angstrom", initial=19.2),
        FitParameter("sto_thickness", 18.0, 22.0, "Angstrom", initial=20.8),
    )
    initial = np.array([parameter.initial for parameter in parameters], dtype=float)
    initial_residuals = residual_function(initial)
    initial_cost = 0.5 * float(np.dot(initial_residuals, initial_residuals))

    start = perf_counter()
    result = optimize_with_jax_least_squares(
        parameters,
        residual_function,
        settings=JaxLeastSquaresOptimizerSettings(max_nfev=80),
    )
    elapsed = perf_counter() - start

    print(f"Initial cost: {initial_cost:.6e}")
    print(f"Final cost: {result.final_cost:.6e}")
    print(f"Best-fit parameters: {result.best_parameters}")
    print(
        "Synthetic truth: "
        f"LNO={TRUE_VALUES['lno_thickness']:.1f} A, "
        f"STO={TRUE_VALUES['sto_thickness']:.1f} A"
    )
    print(f"Optimizer status: {result.status} ({result.message})")
    print(f"Function/Jacobian evaluations: {result.nfev}/{result.njev}")
    print(f"Wall time: {elapsed:.3f} s")


if __name__ == "__main__":
    main()
