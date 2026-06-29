"""Run a small JAX-gradient fit on the synthetic C/LaNiO3/SrTiO3 stack geometry."""

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
    JaxGradientOptimizerSettings,
    optimize_with_jax_gradient,
)
from swanx.reflectivity_jax import (  # noqa: E402
    jitted_parratt_reflectivity,
    jitted_value_and_grad_reflectivity_loss,
)


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
    scan_angles = angles(count=96)
    (
        true_thicknesses,
        deltas,
        betas,
        roughnesses,
        lno_indices,
        sto_indices,
    ) = layer_arrays()
    target = np.asarray(
        jitted_parratt_reflectivity(
            scan_angles,
            PHOTON_ENERGY_EV,
            true_thicknesses,
            deltas,
            betas,
            roughnesses,
        )
    )

    def value_and_grad(vector: np.ndarray) -> tuple[float, np.ndarray]:
        trial_thicknesses = true_thicknesses.copy()
        trial_thicknesses[lno_indices] = vector[0]
        trial_thicknesses[sto_indices] = vector[1]
        loss, gradient = jitted_value_and_grad_reflectivity_loss(
            trial_thicknesses,
            scan_angles,
            PHOTON_ENERGY_EV,
            deltas,
            betas,
            roughnesses,
            target,
        )
        scale = 1.0e8
        return (
            scale * float(loss),
            scale
            * np.asarray(
                [gradient[lno_indices].sum(), gradient[sto_indices].sum()],
                dtype=float,
            ),
        )

    parameters = (
        FitParameter("lno_thickness", 18.0, 22.0, "Angstrom", initial=19.2),
        FitParameter("sto_thickness", 18.0, 22.0, "Angstrom", initial=20.8),
    )
    initial = np.array([parameter.initial for parameter in parameters], dtype=float)
    initial_loss, _ = value_and_grad(initial)

    start = perf_counter()
    result = optimize_with_jax_gradient(
        parameters,
        value_and_grad,
        settings=JaxGradientOptimizerSettings(maxiter=80),
    )
    elapsed = perf_counter() - start

    print(f"Initial loss: {initial_loss:.6e}")
    print(f"Final loss: {result.best_loss:.6e}")
    print(f"Best-fit parameters: {result.best_parameters}")
    print(
        "Synthetic truth: "
        f"LNO={TRUE_VALUES['lno_thickness']:.1f} A, "
        f"STO={TRUE_VALUES['sto_thickness']:.1f} A"
    )
    print(f"Optimizer status: {result.status} ({result.message})")
    print(f"Iterations: {result.nit}, function evaluations: {result.nfev}")
    print(f"Wall time: {elapsed:.3f} s")


if __name__ == "__main__":
    main()
