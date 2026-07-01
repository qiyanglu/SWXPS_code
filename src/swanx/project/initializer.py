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


InitTemplate = Literal["minimal", "multilayer", "fit-demo", "fit", "simulate"]

RUN_PROJECT_TEXT = '''from pathlib import Path
from swanx.project import run_project

output = run_project(Path(__file__).with_name("project.yaml"), progress=True)
print(f"SWANX results written to: {output}")
'''

_REQUIRED_EXAMPLE_FILES = (
    ("OPC", "C.dat"),
    ("OPC", "LaNiO3.dat"),
    ("OPC", "SrTiO3.dat"),
    ("IMFP", "C.ANG"),
    ("IMFP", "LNO.ANG"),
    ("IMFP", "STO.ANG"),
)
_OPTIONAL_EXAMPLE_FILES = (
    ("curves", "lno_sto_c_synthetic_data.csv"),
)
_TEMPLATES = {"minimal", "multilayer", "fit-demo", "fit", "simulate"}


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

    canonical_template = _canonical_template(template)
    source_data_root = _source_data_root(data_root)
    _require_example_data(source_data_root)
    _require_template_data(source_data_root, canonical_template)

    project_dir.mkdir(parents=True, exist_ok=True)
    project_name = _project_name(project_dir.name)

    uses_packaged_default = data_root is None
    should_copy = copy_example_data or uses_packaged_default
    yaml_data_root = project_dir / "data" if should_copy else Path(str(source_data_root))
    if should_copy:
        _copy_example_data(source_data_root, yaml_data_root)

    has_curves = _has_file(source_data_root, ("curves", "lno_sto_c_synthetic_data.csv"))
    (project_dir / "project.yaml").write_text(
        _project_yaml(project_dir, project_name, yaml_data_root, template=canonical_template, has_curves=has_curves),
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


def _canonical_template(template: str) -> str:
    if template in {"minimal", "fit-demo", "fit"}:
        return "fit-demo"
    if template in {"multilayer", "simulate"}:
        return "multilayer"
    return template


def _has_file(data_root: Path | Traversable, parts: tuple[str, ...]) -> bool:
    if isinstance(data_root, Path):
        return (data_root / Path(*parts)).is_file()
    return data_root.joinpath(*parts).is_file()


def _require_example_data(data_root: Path | Traversable) -> None:
    missing = [Path(*parts).as_posix() for parts in _REQUIRED_EXAMPLE_FILES if not _has_file(data_root, parts)]
    if missing:
        formatted = "\n".join(missing)
        raise ProjectValidationError(
            "swanx init requires the C/LaNiO3/SrTiO3 starter OPC and IMFP files. "
            "Install SWANX from a complete source tree or pass --data-root to a folder containing "
            "OPC/C.dat, OPC/LaNiO3.dat, OPC/SrTiO3.dat, IMFP/C.ANG, IMFP/LNO.ANG, and IMFP/STO.ANG. "
            f"Missing:\n{formatted}"
        )


def _require_template_data(data_root: Path | Traversable, template: str) -> None:
    if template == "multilayer":
        return
    missing = [Path(*parts).as_posix() for parts in _OPTIONAL_EXAMPLE_FILES if not _has_file(data_root, parts)]
    if missing:
        formatted = "\n".join(missing)
        raise ProjectValidationError(
            "swanx init fitting templates require tutorial curve files. "
            "Use the packaged default data, pass --copy-example-data with a data root containing curves/, "
            f"or choose --template simulate for a simulation-only project. Missing:\n{formatted}"
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
    data = _data_file(project_dir, data_root, "curves", "lno_sto_c_synthetic_data.csv")
    return f'''
# Example synthetic datasets are available in the starter data.
# Uncomment this block and keep run.mode as "simulate_only" to compare
# simulated curves with data, or switch run.mode to "jax_least_squares"
# with run.optimizer.residual: "auto_fixed_grid" for a fixed-shape fit.
# datasets:
#   reflectivity:
#     path: "{data}"
#     name: "Reflectivity"
#     angle_column: "angle_deg"
#     intensity_column: "reflectivity"
#   rocking_curves:
#     - path: "{data}"
#       name: "La 4d"
#       angle_column: "angle_deg"
#       intensity_column: "la4d_rc"
#     - path: "{data}"
#       name: "O 1s"
#       angle_column: "angle_deg"
#       intensity_column: "o1s_rc"
#     - path: "{data}"
#       name: "Ti 2p"
#       angle_column: "angle_deg"
#       intensity_column: "ti2p_rc"
#     - path: "{data}"
#       name: "C 1s"
#       angle_column: "angle_deg"
#       intensity_column: "c1s_rc"
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
    return _fit_demo_yaml(project_dir, project_name, data_root, paths)


def _common_paths(project_dir: Path, data_root: Path) -> dict[str, str]:
    return {
        "c_opc": _data_file(project_dir, data_root, "OPC", "C.dat"),
        "lno_opc": _data_file(project_dir, data_root, "OPC", "LaNiO3.dat"),
        "sto_opc": _data_file(project_dir, data_root, "OPC", "SrTiO3.dat"),
        "c_imfp": _data_file(project_dir, data_root, "IMFP", "C.ANG"),
        "lno_imfp": _data_file(project_dir, data_root, "IMFP", "LNO.ANG"),
        "sto_imfp": _data_file(project_dir, data_root, "IMFP", "STO.ANG"),
    }


def _materials_yaml(paths: dict[str, str]) -> str:
    return f'''materials:
  C:
    opc_file: "{paths["c_opc"]}"
    imfp_file: "{paths["c_imfp"]}"
  LNO:
    opc_file: "{paths["lno_opc"]}"
    imfp_file: "{paths["lno_imfp"]}"
  STO:
    opc_file: "{paths["sto_opc"]}"
    imfp_file: "{paths["sto_imfp"]}"
'''


def _synthetic_parameters_yaml(*, fit_ready: bool) -> str:
    if fit_ready:
        return '''parameters:
  carbon_thickness:
    initial: 9.6
    lower: 5.0
    upper: 16.0
    vary: true
  carbon_roughness_fraction:
    initial: 0.33
    lower: 0.0
    upper: 1.0
    vary: true
  lno_thickness:
    initial: 19.7
    lower: 18.0
    upper: 22.0
    vary: true
  sto_thickness:
    initial: 20.3
    lower: 18.0
    upper: 22.0
    vary: true
  superlattice_roughness:
    initial: 2.8
    lower: 1.0
    upper: 5.0
    vary: true
  substrate_roughness:
    initial: 3.2
    lower: 1.0
    upper: 5.0
    vary: true
  angle_offset:
    initial: 0.03
    lower: -0.25
    upper: 0.25
    vary: true
'''
    return '''parameters:
  carbon_thickness:
    value: 10.0
    vary: false
  carbon_roughness:
    value: 2.333333333333333
    vary: false
  lno_thickness:
    value: 20.0
    vary: false
  sto_thickness:
    value: 20.0
    vary: false
  superlattice_roughness:
    value: 3.0
    vary: false
  substrate_roughness:
    value: 3.0
    vary: false
'''


def _synthetic_stack_and_core_levels_yaml(*, fit_ready: bool) -> str:
    carbon_roughness = (
        '"1.0 + carbon_roughness_fraction * 4.0"'
        if fit_ready
        else '"$carbon_roughness"'
    )
    return f'''stack:
  - id: "vacuum"
    material: "vacuum"
    thickness_A: 0.0
    roughness_A: 0.0

  - id: "carbon_cap"
    material: "C"
    tags: ["carbon_cap", "surface"]
    thickness_A: "$carbon_thickness"
    roughness_A: {carbon_roughness}

  - repeat:
      times: 20
      layers:
        - id: "lno_{{repeat_index}}"
          material: "LNO"
          tags: ["lno_layers", "oxide_layers", "superlattice"]
          thickness_A: "$lno_thickness"
          roughness_A: "$superlattice_roughness"
        - id: "sto_{{repeat_index}}"
          material: "STO"
          tags: ["sto_layers", "oxide_layers", "superlattice"]
          thickness_A: "$sto_thickness"
          roughness_A: "$superlattice_roughness"

  - id: "sto_substrate"
    material: "STO"
    tags: ["substrate", "sto_layers"]
    thickness_A: 0.0
    roughness_A: "$substrate_roughness"

core_levels:
  - name: "La 4d"
    binding_energy_ev: 105.0
    emit_from:
      tags: ["lno_layers"]
    concentration: 1.0
    emission_angle_deg: 0.0

  - name: "O 1s"
    binding_energy_ev: 530.0
    emit_from:
      tags: ["oxide_layers", "substrate"]
    concentration: 1.0
    emission_angle_deg: 0.0

  - name: "Ti 2p"
    binding_energy_ev: 460.0
    emit_from:
      tags: ["sto_layers"]
    concentration: 1.0
    emission_angle_deg: 0.0

  - name: "C 1s"
    binding_energy_ev: 285.0
    emit_from:
      tags: ["carbon_cap"]
    concentration: 1.0
    emission_angle_deg: 0.0
'''


def _base_simulation_yaml(project_name: str, paths: dict[str, str], dataset_comments: str, *, fit_ready: bool) -> str:
    return f'''project:
  name: "{project_name}"

run:
  mode: "simulate_only"
  outputs:
    plots: true

settings:
  photon_energy_ev: 1000.0
  angle_start_deg: 6.9
  angle_stop_deg: 10.9
  angle_count: 161
  polarization: "s"
  normalization: "edge_polynomial"
  normalization_edge_fraction: 0.10
  normalization_polynomial_order: 2

{_materials_yaml(paths)}
{_synthetic_parameters_yaml(fit_ready=fit_ready)}
{_synthetic_stack_and_core_levels_yaml(fit_ready=fit_ready)}
datasets: {{}}
{dataset_comments}
report: {{}}
'''


def _minimal_yaml(project_name: str, paths: dict[str, str], dataset_comments: str) -> str:
    del dataset_comments
    return _fit_demo_yaml_from_paths(project_name, paths, "data/curves/lno_sto_c_synthetic_data.csv")


def _multilayer_yaml(project_name: str, paths: dict[str, str], dataset_comments: str) -> str:
    return _base_simulation_yaml(project_name, paths, dataset_comments, fit_ready=False)


def _fit_demo_yaml(project_dir: Path, project_name: str, data_root: Path, paths: dict[str, str]) -> str:
    data = _data_file(project_dir, data_root, "curves", "lno_sto_c_synthetic_data.csv")
    return _fit_demo_yaml_from_paths(project_name, paths, data)


def _fit_demo_yaml_from_paths(project_name: str, paths: dict[str, str], data: str) -> str:
    return f'''project:
  name: "{project_name}"

run:
  mode: "jax_least_squares"
  optimizer:
    residual: "auto_fixed_grid"
    max_nfev: 40
    ftol: 1.0e-9
    xtol: 1.0e-9
    gtol: 1.0e-8
    estimate_covariance: true
  outputs:
    plots: true
    identifiability: true

settings:
  photon_energy_ev: 1000.0
  angle_start_deg: 6.9
  angle_stop_deg: 10.9
  angle_count: 161
  polarization: "s"
  normalization: "edge_polynomial"
  normalization_edge_fraction: 0.10
  normalization_polynomial_order: 2
  slicing:
    mode: "fixed_grid"
    min_slices: 10
    max_slice_thickness_A: 2.0
    reference_values:
      carbon_thickness: 16.0
      lno_thickness: 22.0
      sto_thickness: 22.0
  # Bayesian optimization is available as an optional global black-box baseline,
  # but it is not the default or a fallback.

{_materials_yaml(paths)}
{_synthetic_parameters_yaml(fit_ready=True)}
{_synthetic_stack_and_core_levels_yaml(fit_ready=True)}
datasets:
  reflectivity:
    path: "{data}"
    name: "Reflectivity"
    angle_column: "angle_deg"
    intensity_column: "reflectivity"
    log_floor: 1.0e-12
    weight: 0.2709975524929604
  rocking_curves:
    - path: "{data}"
      name: "La 4d"
      angle_column: "angle_deg"
      intensity_column: "la4d_rc"
      weight: 5.0
    - path: "{data}"
      name: "O 1s"
      angle_column: "angle_deg"
      intensity_column: "o1s_rc"
      weight: 5.0
    - path: "{data}"
      name: "Ti 2p"
      angle_column: "angle_deg"
      intensity_column: "ti2p_rc"
      weight: 5.0
    - path: "{data}"
      name: "C 1s"
      angle_column: "angle_deg"
      intensity_column: "c1s_rc"
      weight: 5.0

report: {{}}
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
        data_note = "This project contains a local `data/` copy of the packaged SWANX C/LaNiO3/SrTiO3 starter OPC, IMFP, and curve files."
    elif copied_data:
        data_note = "This project contains a local `data/` copy of the C/LaNiO3/SrTiO3 starter OPC, IMFP, and curve files."
    else:
        data_note = (
            f"This project references external C/LaNiO3/SrTiO3 starter data at `{data_root}` using paths relative to `project.yaml`. "
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

The sample stack is a carbon cap on a LaNiO3/SrTiO3 superlattice on a SrTiO3
substrate. In `project.yaml`, `LNO` means LaNiO3 and `STO` means SrTiO3.
The default project fits the packaged synthetic reflectivity and SW-XPS
rocking curves with JAX least-squares. The fixed-grid JAX residual is built
directly from the stack, parameters, datasets, and slicing settings in
`project.yaml`.

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
- `repeat_index0` is available for formulas that read better with zero-based
  repeat coordinates.
- JAX least-squares is the active fitting path in this starter project.
- Bayesian optimization is available as an optional global black-box baseline.
- Advanced ProjectSpec JAX fitting can still use explicit factory callbacks for
  custom residuals.
'''
