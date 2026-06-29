"""Load the synthetic C/LaNiO3/SrTiO3 data into SWANX fitting data objects."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from examples.synthetic_case import (  # noqa: E402
    DATA_FILE,
    RC_COLUMN_BY_NAME,
    make_data_problem,
)
from swanx.io import read_reflectivity_data, read_rocking_curve_data  # noqa: E402


def main() -> None:
    reflectivity = read_reflectivity_data(
        DATA_FILE,
        name="synthetic C/LaNiO3/SrTiO3 reflectivity",
        angle_column="angle_deg",
        intensity_column="reflectivity",
    )
    rocking_curves = [
        read_rocking_curve_data(
            DATA_FILE,
            name=name,
            angle_column="angle_deg",
            intensity_column=column,
            normalization_mode="mean",
        )
        for name, column in RC_COLUMN_BY_NAME.items()
    ]

    problem = make_data_problem()
    simulation = problem.simulate({})
    evaluation = problem.evaluate({})

    print(f"data file: {DATA_FILE}")
    print(f"reflectivity shape: {reflectivity.angles.shape}")
    print(f"first reflectivity point: {reflectivity.angles[0]:.3f} deg, {reflectivity.reflectivity[0]:.3g}")
    for curve in rocking_curves:
        print(f"{curve.name}: {curve.angles.size} points, mean-normalized mean {curve.intensity.mean():.3f}")
    print(
        "FittingProblem datasets: "
        f"reflectivity={problem.reflectivity.name}, "
        f"rocking_curves={len(problem.rocking_curves)}"
    )
    print(f"simulated stack layers: {len(simulation.stack.optical_layers)}")
    print(f"overlay objective at synthetic truth: {evaluation.objective:.3e}")


if __name__ == "__main__":
    main()
