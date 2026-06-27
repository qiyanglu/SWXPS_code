"""Beginner-friendly YAML project initializer."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from .spec import ProjectValidationError, load_project_spec


RUN_PROJECT_TEXT = '''from pathlib import Path
from swanx.project import run_project

output = run_project(Path(__file__).with_name("project.yaml"))
print(f"SWANX results written to: {output}")
'''

_REQUIRED_EXAMPLE_FILES = (
    ("OPC", "LaNiO3.dat"),
    ("OPC", "SrTiO3.dat"),
    ("IMFP", "LNO.ANG"),
    ("IMFP", "STO.ANG"),
)
_OPTIONAL_EXAMPLE_FILES = (
    ("curves", "lno_sto_reflectivity.csv"),
    ("curves", "la4d_rocking_curve.csv"),
)


def init_project(
    path: str | Path,
    *,
    copy_example_data: bool = False,
    data_root: str | Path | None = None,
) -> Path:
    """Create a small editable YAML project folder."""

    project_dir = Path(path)
    if project_dir.exists() and any(project_dir.iterdir()):
        raise FileExistsError(f"project directory is not empty: {project_dir}")
    source_data_root = _source_data_root(data_root)
    _require_example_data(source_data_root)

    project_dir.mkdir(parents=True, exist_ok=True)
    project_name = _project_name(project_dir.name)
    yaml_data_root = source_data_root
    if copy_example_data:
        yaml_data_root = project_dir / "data"
        _copy_example_data(source_data_root, yaml_data_root)

    (project_dir / "project.yaml").write_text(
        _project_yaml(project_dir, project_name, yaml_data_root),
        encoding="utf-8",
    )
    (project_dir / "run_project.py").write_text(RUN_PROJECT_TEXT, encoding="utf-8")
    (project_dir / "README.md").write_text(
        _readme(project_name, copied_data=copy_example_data, data_root=yaml_data_root),
        encoding="utf-8",
    )
    load_project_spec(project_dir / "project.yaml")
    return project_dir


def _source_data_root(data_root: str | Path | None) -> Path:
    if data_root is not None:
        return Path(data_root).expanduser().resolve()
    return (Path.cwd() / "data").resolve()


def _require_example_data(data_root: Path) -> None:
    missing = [data_root / Path(*parts) for parts in _REQUIRED_EXAMPLE_FILES if not (data_root / Path(*parts)).exists()]
    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise ProjectValidationError(
            "swanx init requires the minimal tutorial data files. "
            "Run from the SWANX repository, pass --data-root, or use --copy-example-data with a valid data root. "
            f"Missing:\n{formatted}"
        )


def _copy_example_data(source_data_root: Path, target_data_root: Path) -> None:
    for parts in (*_REQUIRED_EXAMPLE_FILES, *_OPTIONAL_EXAMPLE_FILES):
        source = source_data_root / Path(*parts)
        if not source.exists():
            continue
        target = target_data_root / Path(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def _project_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    return cleaned or "swanx_project"


def _relative_path(project_dir: Path, path: Path) -> str:
    try:
        return Path(os.path.relpath(path, project_dir)).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _data_file(project_dir: Path, data_root: Path, *parts: str) -> str:
    return _relative_path(project_dir, data_root / Path(*parts))


def _project_yaml(project_dir: Path, project_name: str, data_root: Path) -> str:
    lno_opc = _data_file(project_dir, data_root, "OPC", "LaNiO3.dat")
    sto_opc = _data_file(project_dir, data_root, "OPC", "SrTiO3.dat")
    lno_imfp = _data_file(project_dir, data_root, "IMFP", "LNO.ANG")
    sto_imfp = _data_file(project_dir, data_root, "IMFP", "STO.ANG")
    return f'''project:
  name: "{project_name}"

settings:
  photon_energy_ev: 900.0
  angle_start_deg: 5.0
  angle_stop_deg: 15.0
  angle_count: 41
  polarization: "p"
  normalization: "mean"
  fit_method: "simulate_only"

materials:
  LNO:
    opc_file: "{lno_opc}"
    imfp_file: "{lno_imfp}"
  STO:
    opc_file: "{sto_opc}"
    imfp_file: "{sto_imfp}"

parameters:
  lno_thickness:
    initial: 40.0
    lower: 30.0
    upper: 50.0
    vary: true
  interface_roughness:
    initial: 3.0
    lower: 0.0
    upper: 8.0
    vary: true

stack:
  - id: "vacuum"
    material: "vacuum"
    thickness_A: 0.0
    roughness_A: 0.0

  - id: "lno_1"
    material: "LNO"
    tags: ["lno_layers"]
    thickness_A: "$lno_thickness"
    roughness_A: "$interface_roughness"

  - id: "sto_substrate"
    material: "STO"
    thickness_A: 0.0
    roughness_A: 0.0

core_levels:
  - name: "La 4d"
    binding_energy_ev: 105.0
    emit_from:
      tags: ["lno_layers"]
    concentration: 1.0
    emission_angle_deg: 0.0

report:
  save_plots: true
'''


def _readme(project_name: str, *, copied_data: bool, data_root: Path) -> str:
    if copied_data:
        data_note = "This project contains a local `data/` copy of the minimal tutorial OPC, IMFP, and curve files."
    else:
        data_note = (
            f"This project references tutorial data at `{data_root}` using paths relative to `project.yaml`. "
            "Keep that data folder available, or rerun `swanx init --copy-example-data` for a self-contained copy."
        )
    return f'''# {project_name}

Edit `project.yaml`, then run:

```bash
python run_project.py
```

Outputs are written under `runs/` inside this project folder unless `project.output_dir` is set.

{data_note}

Optional automation commands:

```bash
swanx validate project.yaml
swanx run project.yaml
```

Notes:

- `thickness_A` and `roughness_A` are in Angstrom.
- `roughness_A` means upper-interface roughness/interdiffusion for that layer.
- `repeat_index` is 1-based inside repeat blocks.
- JAX least-squares is the recommended fitting path for differentiable fixed-shape workflows.
- Bayesian optimization is available as an optional global black-box baseline.
- This starter is simulation-only. JAX fitting ProjectSpecs still require factory callbacks in v1.1.
'''
