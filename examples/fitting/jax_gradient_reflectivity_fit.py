"""Run a tiny standalone JAX-gradient reflectivity fit.

This script is intentionally small: it fits one film thickness against a
synthetic Parratt reflectivity curve and prints optimizer diagnostics.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src"))

from swanx.fitting import (  # noqa: E402
    FitParameter,
    JaxGradientOptimizerSettings,
    optimize_with_jax_gradient,
)
from swanx.reflectivity_jax import (  # noqa: E402
    jitted_parratt_reflectivity,
    jitted_value_and_grad_reflectivity_loss,
)


def main() -> None:
    angles = np.linspace(0.5, 4.0, 96)
    energy_ev = 3000.0
    deltas = np.array([0.0, 5.0e-6, 1.0e-5])
    betas = np.array([0.0, 1.0e-7, 2.0e-7])
    roughnesses = np.array([0.0, 0.0, 0.0])
    true_thicknesses = np.array([0.0, 24.0, 0.0])
    target = np.asarray(
        jitted_parratt_reflectivity(
            angles,
            energy_ev,
            true_thicknesses,
            deltas,
            betas,
            roughnesses,
        )
    )

    def value_and_grad(vector: np.ndarray) -> tuple[float, np.ndarray]:
        trial_thicknesses = true_thicknesses.copy()
        trial_thicknesses[1] = vector[0]
        loss, gradient = jitted_value_and_grad_reflectivity_loss(
            trial_thicknesses,
            angles,
            energy_ev,
            deltas,
            betas,
            roughnesses,
            target,
        )
        scale = 1.0e8
        return (
            scale * float(loss),
            scale * np.asarray([gradient[1]], dtype=float),
        )

    parameters = (FitParameter("film_thickness", 5.0, 50.0, "Angstrom", initial=35.0),)
    initial = np.array([parameters[0].initial], dtype=float)
    initial_loss, _ = value_and_grad(initial)

    start = perf_counter()
    result = optimize_with_jax_gradient(
        parameters,
        value_and_grad,
        settings=JaxGradientOptimizerSettings(maxiter=60),
    )
    elapsed = perf_counter() - start

    print(f"Initial loss: {initial_loss:.6e}")
    print(f"Final loss: {result.best_loss:.6e}")
    print(f"Best-fit parameters: {result.best_parameters}")
    print(f"Optimizer status: {result.status} ({result.message})")
    print(f"Iterations: {result.nit}, function evaluations: {result.nfev}")
    print(f"Wall time: {elapsed:.3f} s")


if __name__ == "__main__":
    main()
