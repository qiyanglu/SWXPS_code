"""Run a small standalone JAX nonlinear least-squares reflectivity fit."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swxps import (  # noqa: E402
    FitParameter,
    JaxLeastSquaresOptimizerSettings,
    ReflectivityData,
    build_jax_residual_function,
    optimize_with_jax_least_squares,
)
from swxps.reflectivity_jax import parratt_reflectivity_jax  # noqa: E402


def main() -> None:
    import jax.numpy as jnp

    angles = jnp.linspace(0.5, 4.0, 96)
    energy_ev = 3000.0
    deltas = jnp.array([0.0, 5.0e-6, 1.0e-5])
    betas = jnp.array([0.0, 1.0e-7, 2.0e-7])
    roughnesses = jnp.zeros(3)
    true_thicknesses = jnp.array([0.0, 24.0, 0.0])
    target = np.asarray(
        parratt_reflectivity_jax(
            angles,
            energy_ev,
            true_thicknesses,
            deltas,
            betas,
            roughnesses,
        )
    )
    data = ReflectivityData(
        name="synthetic reflectivity",
        angles=np.asarray(angles),
        reflectivity=target,
    )

    def simulate_curves(parameters):
        thicknesses = true_thicknesses.at[1].set(parameters[0])
        reflectivity = parratt_reflectivity_jax(
            angles,
            energy_ev,
            thicknesses,
            deltas,
            betas,
            roughnesses,
        )
        return reflectivity, ()

    residual_function = build_jax_residual_function(
        simulate_curves,
        reflectivity=data,
    )
    parameters = (
        FitParameter(
            "film_thickness",
            5.0,
            50.0,
            "Angstrom",
            initial=25.0,
        ),
    )
    initial_residuals = residual_function(np.array([25.0]))
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
    print(f"Optimizer status: {result.status} ({result.message})")
    print(f"Function/Jacobian evaluations: {result.nfev}/{result.njev}")
    print(f"Wall time: {elapsed:.3f} s")


if __name__ == "__main__":
    main()
