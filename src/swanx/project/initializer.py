"""Beginner-friendly YAML project initializer."""

from __future__ import annotations

from importlib import resources
from importlib.resources.abc import Traversable
import os
import re
import shutil
from pathlib import Path
from typing import Literal

from .spec import ProjectValidationError, load_project_spec


InitTemplate = Literal["minimal", "multilayer", "fit-demo"]

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
_TEMPLATES = {"minimal", "multilayer", "fit-demo"}


def init_project(
    path: str | Path,
    *,
    copy_example_data: bool = False,
    data_root: str | Path | None = None,
    template: InitTemplate = "minimal",
) -> Path:
    """Create a small editable YAML project folder."""

    if template not in _TEMPLATES:
        raise ProjectValidationError(f"unknown init template {template!r}; choose one of {sorted(_TEMPLATES)}")

    project_dir = Path(path)
    if project_dir.exists() and any(project_dir.iterdir()):
        raise FileExistsError(f"project directory is not empty: {project_dir}")

    source_data_root = _source_data_root(data_root)
    _require_example_data(source_data_root)
    _require_template_data(source_data_root, template)

    project_dir.mkdir(parents=True, exist_ok=True)
    project_name = _project_name(project_dir.name)

    uses_packaged_default = data_root is None
    should_copy = copy_example_data or uses_packaged_default
    yaml_data_root = project_dir / "data" if should_copy else Path(str(source_data_root))
    if should_copy:
        _copy_example_data(source_data_root, yaml_data_root)

    has_curves = _has_file(source_data_root, ("curves", "lno_sto_reflectivity.csv")) and _has_file(
        source_data_root,
        ("curves", "la4d_rocking_curve.csv"),
    )
    (project_dir / "project.yaml").write_text(
        _project_yaml(project_dir, project_name, yaml_data_root, template=template, has_curves=has_curves),
        encoding="utf-8",
    )
    (project_dir / "run_project.py").write_text(RUN_PROJECT_TEXT, encoding="utf-8")
    (project_dir / "README.md").write_text(
        _readme(
            project_name,
            copied_data=should_copy,
            data_root=yaml_data_root,
            template=template,
            packaged_default=uses_packaged_default,
        ),
        encoding="utf-8",
    )
    load_project_spec(project_dir / "project.yaml")
    return project_dir


def _source_data_root(data_root: str | Path | None) -> Path | Traversable:
    if data_root is not None:
        return Path(data_root).expanduser().resolve()
    return resources.files("swanx.project") / "example_data"


def _has_file(data_root: Path | Traversable, parts: tuple[str, ...]) -> bool:
    if isinstance(data_root, Path):
        return (data_root / Path(*parts)).is_file()
    return data_root.joinpath(*parts).is_file()


def _require_example_data(data_root: Path | Traversable) -> None:
    missing = [Path(*parts).as_posix() for parts in _REQUIRED_EXAMPLE_FILES if not _has_file(data_root, parts)]
    if missing:
        formatted = "\n".join(missing)
        raise ProjectValidationError(
            "swanx init requires the minimal tutorial OPC and IMFP files. "
            "Install SWANX from a complete source tree or pass --data-root to a folder containing "
            "OPC/LaNiO3.dat, OPC/SrTiO3.dat, IMFP/LNO.ANG, and IMFP/STO.ANG. "
            f"Missing:\n{formatted}"
        )


def _require_template_data(data_root: Path | Traversable, template: str) -> None:
    if template != "fit-demo":
        return
    missing = [Path(*parts).as_posix() for parts in _OPTIONAL_EXAMPLE_FILES if not _has_file(data_root, parts)]
    if missing:
        formatted = "\n".join(missing)
        raise ProjectValidationError(
            "swanx init --template fit-demo requires tutorial curve files. "
            "Use the packaged default data, pass --copy-example-data with a data root containing curves/, "
            f"or choose --template minimal. Missing:\n{formatted}"
        )


def _copy_example_data(source_data_root: Path | Traversable, target_data_root: Path) -> None:
    for parts in (*_REQUIRED_EXAMPLE_FILES, *_OPTIONAL_EXAMPLE_FILES):
        if not _has_file(source_data_root, parts):
            continue
        target = target_data_root / Path(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(source_data_root, Path):
            shutil.copyfile(source_data_root / Path(*parts), target)
        else:
            with source_data_root.joinpath(*parts).open("rb") as source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)


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


def _dataset_comments(project_dir: Path, data_root: Path, *, has_curves: bool) -> str:
    if not has_curves:
        return ""
    refl = _data_file(project_dir, data_root, "curves", "lno_sto_reflectivity.csv")
    rc = _data_file(project_dir, data_root, "curves", "la4d_rocking_curve.csv")
    return f'''
# Example experimental datasets are available in the tutorial data.
# Uncomment this block and keep fit_method as "simulate_only" to compare
# simulated curves with data, or provide a fitting callback before switching
# to "jax_least_squares".
# datasets:
#   reflectivity:
#     path: "{refl}"
#     name: "Reflectivity"
#   rocking_curves:
#     - path: "{rc}"
#       name: "La 4d"
'''


def _project_yaml(
    project_dir: Path,
    project_name: str,
    data_root: Path,
    *,
    template: str,
    has_curves: bool,
) -> str:
    paths = _common_paths(project_dir, data_root)
    if template == "multilayer":
        return _multilayer_yaml(project_name, paths, _dataset_comments(project_dir, data_root, has_curves=has_curves))
    if template == "fit-demo":
        return _fit_demo_yaml(project_dir, project_name, data_root, paths)
    return _minimal_yaml(project_name, paths, _dataset_comments(project_dir, data_root, has_curves=has_curves))


def _common_paths(project_dir: Path, data_root: Path) -> dict[str, str]:
    return {
        "lno_opc": _data_file(project_dir, data_root, "OPC", "LaNiO3.dat"),
        "sto_opc": _data_file(project_dir, data_root, "OPC", "SrTiO3.dat"),
        "lno_imfp": _data_file(project_dir, data_root, "IMFP", "LNO.ANG"),
        "sto_imfp": _data_file(project_dir, data_root, "IMFP", "STO.ANG"),
    }


def _materials_yaml(paths: dict[str, str]) -> str:
    return f'''materials:
  LNO:
    opc_file: "{paths["lno_opc"]}"
    imfp_file: "{paths["lno_imfp"]}"
  STO:
    opc_file: "{paths["sto_opc"]}"
    imfp_file: "{paths["sto_imfp"]}"
'''


def _minimal_yaml(project_name: str, paths: dict[str, str], dataset_comments: str) -> str:
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

{_materials_yaml(paths)}
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

datasets: {{}}
{dataset_comments}
report:
  save_plots: true
'''


def _multilayer_yaml(project_name: str, paths: dict[str, str], dataset_comments: str) -> str:
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

{_materials_yaml(paths)}
parameters:
  lno_thickness:
    initial: 40.0
    lower: 30.0
    upper: 50.0
    vary: true
  sto_thickness:
    initial: 10.0
    lower: 5.0
    upper: 20.0
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

  - repeat:
      times: 4
      layers:
        - id: "lno_{{repeat_index}}"
          material: "LNO"
          tags: ["lno_layers"]
          thickness_A: "$lno_thickness"
          roughness_A: "$interface_roughness"
        - id: "sto_{{repeat_index}}"
          material: "STO"
          tags: ["sto_layers"]
          thickness_A: "$sto_thickness"
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

datasets: {{}}
{dataset_comments}
report:
  save_plots: true
'''


def _fit_demo_yaml(project_dir: Path, project_name: str, data_root: Path, paths: dict[str, str]) -> str:
    refl = _data_file(project_dir, data_root, "curves", "lno_sto_reflectivity.csv")
    rc = _data_file(project_dir, data_root, "curves", "la4d_rocking_curve.csv")
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
  # To run JAX least-squares, change fit_method to "jax_least_squares"
  # and provide a fixed-shape residual callback:
  # optimizer:
  #   residual_function_factory: "your_module:build_residual_function"
  # Bayesian optimization is available as an optional global black-box baseline,
  # but it is not the default or a fallback.

{_materials_yaml(paths)}
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
  scale_hint:
    value: 1.0
    vary: false

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

datasets:
  reflectivity:
    path: "{refl}"
    name: "Reflectivity"
  rocking_curves:
    - path: "{rc}"
      name: "La 4d"

report:
  save_plots: true
'''


def _readme(
    project_name: str,
    *,
    copied_data: bool,
    data_root: Path,
    template: str,
    packaged_default: bool,
) -> str:
    if packaged_default:
        data_note = "This project contains a local `data/` copy of the packaged SWANX tutorial OPC, IMFP, and curve files."
    elif copied_data:
        data_note = "This project contains a local `data/` copy of the minimal tutorial OPC, IMFP, and curve files."
    else:
        data_note = (
            f"This project references external tutorial data at `{data_root}` using paths relative to `project.yaml`. "
            "Keep that folder available, or rerun `swanx init --copy-example-data --data-root <path>` for a local copy."
        )
    return f'''# {project_name}

Edit `project.yaml`, then run:

```bash
python run_project.py
```

Outputs are written under `runs/` inside this project folder unless `project.output_dir` is set.

Template: `{template}`

{data_note}

Optional automation commands:

```bash
swanx inspect project.yaml
swanx validate project.yaml
swanx run project.yaml
```

Notes:

- `thickness_A` and `roughness_A` are in Angstrom.
- `roughness_A` means upper-interface roughness/interdiffusion for that layer.
- `repeat_index` is 1-based inside repeat blocks.
- JAX least-squares is the recommended fitting path for differentiable fixed-shape workflows.
- Bayesian optimization is available as an optional global black-box baseline.
- ProjectSpec v1.2 does not build automatic no-code JAX residuals; JAX fitting still requires factory callbacks.
'''
