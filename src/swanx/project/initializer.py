"""Beginner-friendly YAML project initializer."""

from __future__ import annotations

import os
import re
from pathlib import Path


RUN_PROJECT_TEXT = '''from pathlib import Path
from swanx.project import run_project

output = run_project(Path(__file__).with_name("project.yaml"))
print(f"SWANX results written to: {output}")
'''


def init_project(path: str | Path) -> Path:
    """Create a small editable YAML project folder."""

    project_dir = Path(path)
    if project_dir.exists() and any(project_dir.iterdir()):
        raise FileExistsError(f"project directory is not empty: {project_dir}")
    project_dir.mkdir(parents=True, exist_ok=True)
    project_name = _project_name(project_dir.name)
    (project_dir / "project.yaml").write_text(_project_yaml(project_dir, project_name), encoding="utf-8")
    (project_dir / "run_project.py").write_text(RUN_PROJECT_TEXT, encoding="utf-8")
    (project_dir / "README.md").write_text(_readme(project_name), encoding="utf-8")
    return project_dir


def _project_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    return cleaned or "swanx_project"


def _relative_data_path(project_dir: Path, *parts: str) -> str:
    data_path = Path.cwd() / "data" / Path(*parts)
    return Path(os.path.relpath(data_path, project_dir)).as_posix()


def _project_yaml(project_dir: Path, project_name: str) -> str:
    lno_opc = _relative_data_path(project_dir, "OPC", "LaNiO3.dat")
    sto_opc = _relative_data_path(project_dir, "OPC", "SrTiO3.dat")
    lno_imfp = _relative_data_path(project_dir, "IMFP", "LNO.ANG")
    sto_imfp = _relative_data_path(project_dir, "IMFP", "STO.ANG")
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


def _readme(project_name: str) -> str:
    return f'''# {project_name}

Edit `project.yaml`, then run:

```bash
python run_project.py
```

Optional automation commands:

```bash
swanx validate project.yaml
swanx run project.yaml
```

This starter is simulation-only. Add experimental datasets before changing `settings.fit_method` to `jax_least_squares`, the recommended fitting path. Bayesian optimization remains available as an optional global black-box baseline or robustness check.
'''
