"""Draw a schematic of the fitted synthetic C/LNO/STO stack."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swxps import plot_stack_schematic

from fit_lno_sto_c_synthetic_bo import PARAMETERS, TRUE_VALUES, build_stack


def best_parameters_from_history(path: Path) -> dict[str, float]:
    """Read the best BO parameters from a saved fitting-history CSV."""

    data = np.genfromtxt(path, delimiter=",", names=True)
    best_index = int(np.argmin(data["objective"]))
    return {
        parameter.name: float(data[parameter.name][best_index])
        for parameter in PARAMETERS
    }


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    history_path = output_dir / "lno_sto_c_bo_history.csv"
    values = best_parameters_from_history(history_path)
    stack = build_stack(values)
    output_path = output_dir / "lno_sto_c_bo_stack_schematic.png"
    plot_stack_schematic(
        output_path,
        stack,
        title="Fitted C/LNO/STO Stack",
        top_layers=5,
        bottom_layers=3,
    )

    print("Best-fit values used for schematic:")
    for parameter in PARAMETERS:
        value = values[parameter.name]
        true_value = TRUE_VALUES[parameter.name]
        unit = f" {parameter.unit}" if parameter.unit else ""
        print(f"  {parameter.name}: {value:.6g}{unit} (true {true_value:g}{unit})")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
