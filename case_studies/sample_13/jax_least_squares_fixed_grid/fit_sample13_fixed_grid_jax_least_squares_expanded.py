"""Run the Sample 13 fixed-grid JAX/TRF fit with expanded pressed bounds."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import fit_sample13_fixed_grid_jax_least_squares as runner


EXPANDED_BOUND_OVERRIDES = {
    "carbon_thickness": (1.0, 15.0),
    "lno_thickness_start": (12.5, 20.5),
    "sto_thickness_delta": (-1.0, 3.0),
    "lno_thickness_delta": (-1.0, 3.0),
    "thickness_transition_repeat": (-5.0, 39.0),
    "thickness_transition_width": (1.0, 30.0),
    "sto_roughness_first": (1.0, 6.0),
    "sto_roughness_last": (1.0, 6.0),
    "lno_roughness_first": (1.0, 6.0),
    "lno_roughness_last": (1.0, 6.0),
    "substrate_roughness": (1.0, 6.0),
    "reflectivity_angle_offset": (-0.50, 0.50),
    "rc_angle_offset": (-0.50, 0.50),
}


PARAMETERS = tuple(
    replace(
        parameter,
        lower=EXPANDED_BOUND_OVERRIDES[parameter.name][0],
        upper=EXPANDED_BOUND_OVERRIDES[parameter.name][1],
    )
    if parameter.name in EXPANDED_BOUND_OVERRIDES
    else parameter
    for parameter in runner.PARAMETERS
)

# The base runner deliberately reads these globals at execution time. Updating
# both modules keeps its model, problem definitions, exports, and plots on the
# same expanded parameter declarations without duplicating the fitting code.
runner.PARAMETERS = PARAMETERS
runner.PARAMETER_INDEX = {
    item.name: index for index, item in enumerate(PARAMETERS)
}
runner.legacy_trf.PARAMETERS = PARAMETERS
runner.DEFAULT_OUTPUT_DIR = (
    runner.REPO_ROOT
    / "runs"
    / "sample_13"
    / "jax_least_squares_fixed_grid"
    / "expanded_bounds_60nfev"
)


if __name__ == "__main__":
    runner.main()
