"""Fixed-shape JAX residual factory for the synthetic C/LNO/STO ProjectSpec.

ProjectSpec v1.2 intentionally does not generate no-code JAX residuals. This
module is the explicit callback requested by ``settings.optimizer`` in
``project.yaml``. It reuses the existing synthetic benchmark's fixed-grid JAX
model and weighting conventions while letting the YAML file own materials,
stack layout, datasets, and report generation.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
CASE_DIR = PROJECT_DIR.parent
REPO_ROOT = CASE_DIR.parents[1]
for path in (REPO_ROOT / "src", CASE_DIR, PROJECT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fit_reflectivity_rc_bo as case  # noqa: E402
from fit_unified_jax_least_squares import (  # noqa: E402
    UnifiedSyntheticJaxModel,
    capacity_values,
)
from swanx.jax_least_squares import (  # noqa: E402
    JaxLeastSquaresResidualSettings,
    build_jax_residual_function,
)
from swanx.stack.slicing import LayerSlicingPolicy, fixed_layer_grid_plan  # noqa: E402

DATA_FILE = PROJECT_DIR / "data" / "curves" / "lno_sto_c_synthetic_data.csv"


def build_residual_function(problem):
    """Return a fixed-shape residual function for ``swanx run project.yaml``.

    The YAML ProjectSpec supplies the datasets and parameter declarations. This
    factory adds the benchmark-specific pieces that are intentionally outside
    generic ProjectSpec v1.2: fixed-grid JAX tracing, the off-peak RC
    normalization mask, and the original synthetic benchmark dataset weights.
    """

    if problem.reflectivity is None:
        raise ValueError("synthetic ProjectSpec requires a reflectivity dataset")
    if len(problem.rocking_curves) != 4:
        raise ValueError("synthetic ProjectSpec requires La 4d, O 1s, Ti 2p, and C 1s datasets")

    data = case.load_data(DATA_FILE)
    reflectivity_weight, _diagnostics = case.resolve_reflectivity_weight(data, "auto")
    reflectivity = replace(problem.reflectivity, weight=reflectivity_weight, log_floor=1.0e-12)
    rocking_curves = tuple(
        replace(curve, weight=case.RC_WEIGHTS.get(curve.name, 1.0))
        for curve in problem.rocking_curves
    )

    capacity_stack = case.build_stack(capacity_values())
    plan = fixed_layer_grid_plan(
        capacity_stack.optical_layers,
        LayerSlicingPolicy(min_slices=3, max_slice_thickness=1.0),
    )
    angles = np.asarray(problem.reflectivity.angles, dtype=float)
    peak_angle = angles[np.argmax(problem.reflectivity.reflectivity)]
    offpeak_mask = np.abs(angles - peak_angle) > 1.25
    model = UnifiedSyntheticJaxModel(
        angles,
        offpeak_mask,
        problem.core_levels,
        plan,
        capacity_stack,
    )
    return build_jax_residual_function(
        model.simulate_curves,
        reflectivity=reflectivity,
        rocking_curves=rocking_curves,
        settings=JaxLeastSquaresResidualSettings(
            reflectivity_log=True,
            rocking_curve_normalization="mean_absolute",
        ),
    )
