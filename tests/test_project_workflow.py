from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path

import numpy as np
import pytest

from swanx.fitting import FitContribution, JointObjective, FitParameter, evaluation_from_contributions
from swanx.project.builder import build_project, project_polarization
from swanx.project.reports import write_method_outputs
from swanx.project.runner import run_project, validate_project
from swanx.project.spec import ProjectValidationError, load_project_spec
from swanx.project.yaml_io import YAML_INSTALL_MESSAGE, read_yaml
from swanx.cli import main as cli_main


def _write_opc(path: Path, delta: float = 0.1) -> None:
    path.write_text(
        "M Density=1\n"
        "Energy(eV), Delta, Beta\n"
        f"800 {delta} 0.01\n"
        f"1000 {2 * delta} 0.02\n",
        encoding="utf-8",
    )


def _write_imfp(path: Path) -> None:
    path.write_text(
        "energy imfp\n"
        "700 7.0\n"
        "900 9.0\n",
        encoding="utf-8",
    )


def _write_curve(path: Path, column: str = "reflectivity") -> None:
    path.write_text(
        f"angle_deg,{column}\n"
        "5.0,1.0\n"
        "6.0,1.1\n",
        encoding="utf-8",
    )


def _project_yaml(tmp_path: Path, *, extra_stack: str = "", datasets: str = "{}", output_dir: str | None = None) -> Path:
    _write_opc(tmp_path / "LNO.dat", 0.1)
    _write_opc(tmp_path / "STO.dat", 0.2)
    _write_imfp(tmp_path / "LNO.ANG")
    _write_imfp(tmp_path / "STO.ANG")
    output = "" if output_dir is None else f"  output_dir: \"{output_dir}\"\n"
    text = f"""
project:
  name: "yaml_test"
{output}
settings:
  photon_energy_ev: 900.0
  angle_start_deg: 5.0
  angle_stop_deg: 6.0
  angle_count: 2
  polarization: "unpolarized"
  normalization: "mean"
  fit_method: "simulate_only"
materials:
  LNO:
    opc_file: "LNO.dat"
    imfp_file: "LNO.ANG"
  STO:
    opc_file: "STO.dat"
    imfp_file: "STO.ANG"
parameters:
  lno_thickness:
    initial: 40.0
    lower: 30.0
    upper: 50.0
  sto_thickness:
    initial: 10.0
    lower: 5.0
    upper: 20.0
  interface_roughness:
    initial: 3.0
    lower: 0.0
    upper: 8.0
stack:
  - id: "vacuum"
    material: "vacuum"
    thickness_A: 0.0
    roughness_A: 0.0
{extra_stack if extra_stack else '''  - repeat:
      times: 2
      layers:
        - id: "lno_{repeat_index}"
          material: "LNO"
          tags: ["lno_layers"]
          thickness_A: "$lno_thickness"
          roughness_A: "$interface_roughness"
        - id: "sto_{repeat_index}"
          material: "STO"
          tags: ["sto_layers"]
          thickness_A: "sto_thickness + repeat_index"
          roughness_A: "interface_roughness / 2"
'''}  - id: "sto_substrate"
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
datasets: {datasets}
report:
  save_plots: false
"""
    path = tmp_path / "project.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_lazy_pyyaml_import_missing_message(monkeypatch, tmp_path):
    original = importlib.import_module

    def fake_import(name, package=None):
        if name == "yaml":
            raise ImportError("missing")
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(ImportError, match="YAML project support requires PyYAML") as error:
        read_yaml(tmp_path / "project.yaml")
    assert YAML_INSTALL_MESSAGE in str(error.value)


pytest.importorskip("yaml")


def test_yaml_parsing_validation_repeat_and_expressions(tmp_path):
    path = _project_yaml(tmp_path)
    spec = load_project_spec(path)

    assert spec.expanded_layer_ids() == (
        "vacuum",
        "lno_1",
        "sto_1",
        "lno_2",
        "sto_2",
        "sto_substrate",
    )
    layers = spec.layer_specs_for_values(spec.default_parameter_values())
    assert layers[1]["thickness"] == pytest.approx(40.0)
    assert layers[2]["thickness"] == pytest.approx(11.0)
    assert layers[4]["thickness"] == pytest.approx(12.0)
    assert layers[2]["roughness"] == pytest.approx(1.5)


def test_template_project_minimal_validates():
    spec = validate_project(Path("templates/project_minimal.yaml"))
    assert spec.name == "minimal_yaml_project"


def test_duplicate_layer_id_error(tmp_path):
    extra = '''  - id: "film"
    material: "LNO"
    tags: ["lno_layers"]
    thickness_A: 10.0
    roughness_A: 1.0
  - id: "film"
    material: "LNO"
    thickness_A: 10.0
    roughness_A: 1.0
'''
    path = _project_yaml(tmp_path, extra_stack=extra)
    with pytest.raises(ProjectValidationError, match="duplicate layer id"):
        load_project_spec(path)


def test_unknown_parameter_layer_tag_and_missing_file_errors(tmp_path):
    path = _project_yaml(
        tmp_path,
        extra_stack='''  - id: "film"
    material: "LNO"
    thickness_A: "$missing_parameter"
    roughness_A: 1.0
''',
    )
    with pytest.raises(ProjectValidationError, match="unknown parameter"):
        load_project_spec(path)

    path = _project_yaml(tmp_path)
    text = path.read_text(encoding="utf-8").replace('emit_from:\n      tags: ["lno_layers"]', 'emit_from:\n      tags: ["missing_tag"]')
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ProjectValidationError, match="unknown tag"):
        load_project_spec(path)

    path = _project_yaml(tmp_path)
    (tmp_path / "LNO.dat").unlink()
    with pytest.raises(ProjectValidationError, match="missing data file"):
        load_project_spec(path)


def test_core_level_layer_tag_resolution_and_polarization(tmp_path):
    path = _project_yaml(tmp_path)
    built = build_project(load_project_spec(path))

    assert built.core_levels[0].emitting_layer_indices == (1, 3)
    assert project_polarization("s") == "s"
    assert project_polarization("p") == "p"
    assert project_polarization("unpolarized") == {"s": 0.5, "p": 0.5}


def test_validate_run_cli_and_simulate_only_outputs(tmp_path):
    output_dir = (tmp_path / "out").as_posix()
    path = _project_yaml(tmp_path, output_dir=output_dir)

    assert validate_project(path).name == "yaml_test"
    assert cli_main(["validate", str(path)]) == 0
    output = run_project(path)
    assert output == tmp_path / "out"
    assert cli_main(["run", str(path)]) == 0

    expected = [
        "input/project_original.yaml",
        "input/project_resolved.yaml",
        "input/run_metadata.json",
        "resolved/stack_resolved.csv",
        "resolved/materials_resolved.csv",
        "resolved/core_levels_resolved.csv",
        "resolved/parameters_resolved.csv",
        "resolved/datasets_resolved.csv",
        "simulation/reflectivity_simulated.csv",
        "simulation/rocking_curves_simulated.csv",
        "fit/fit_summary.json",
        "fit/best_parameters.csv",
    ]
    for relative in expected:
        assert (output / relative).exists()
    assert not any((output / "optimizer").rglob("*.csv"))


def test_run_project_writes_experimental_data_and_residuals(tmp_path):
    _write_curve(tmp_path / "reflectivity.csv", "reflectivity")
    _write_curve(tmp_path / "la4d.csv", "intensity")
    datasets = '''
  reflectivity:
    path: "reflectivity.csv"
    name: "R"
  rocking_curves:
    - path: "la4d.csv"
      name: "La 4d"
'''
    path = _project_yaml(tmp_path, datasets=datasets, output_dir=(tmp_path / "out_data").as_posix())

    output = run_project(path)

    assert (output / "data" / "reflectivity_experimental.csv").exists()
    assert (output / "data" / "rocking_curves_experimental.csv").exists()
    assert (output / "fit" / "residuals.csv").exists()


@dataclass(frozen=True)
class _Record:
    iteration: int
    cost: float = 1.0
    loss: float = 1.0
    gradient_norm: float = 0.1
    parameters: dict[str, float] | None = None


class _History:
    def __init__(self, evaluations):
        self.evaluations = tuple(evaluations)


class _Result:
    status = 1
    message = "ok"
    success = True
    best_parameters = {"x": 1.0}
    final_cost = 0.5
    best_loss = 0.4
    best_objective = 0.3
    final_residuals = np.array([1.0, 2.0])
    final_jacobian = np.eye(2)
    covariance = np.eye(2)
    final_gradient = np.array([0.1, 0.2])
    history = (_Record(1, parameters={"x": 1.0}),)


def test_method_specific_report_writers(tmp_path):
    write_method_outputs(tmp_path, "jax_least_squares", _Result())
    assert (tmp_path / "optimizer" / "least_squares" / "covariance.csv").exists()
    assert (tmp_path / "optimizer" / "least_squares" / "correlation.csv").exists()

    write_method_outputs(tmp_path, "jax_gradient", _Result())
    gradient_dir = tmp_path / "optimizer" / "gradient"
    assert (gradient_dir / "objective_history.csv").exists()
    assert (gradient_dir / "final_gradient.csv").exists()
    assert not (gradient_dir / "covariance.csv").exists()

    evaluation = evaluation_from_contributions(
        {"x": 1.0},
        (FitContribution("synthetic", raw=1.0, weight=1.0),),
    )
    objective = JointObjective((FitParameter("x", 0.0, 2.0),), lambda _: evaluation)
    bo_result = type(
        "BOResult",
        (),
        {
            "best_parameters": {"x": 1.0},
            "best_objective": 1.0,
            "history": _History((objective.history.append(evaluation).evaluations[0],)),
        },
    )()
    write_method_outputs(tmp_path, "bayesian_optimization", bo_result)
    bayes_dir = tmp_path / "optimizer" / "bayesian"
    assert (bayes_dir / "evaluations.csv").exists()
    assert (bayes_dir / "best_so_far.csv").exists()
    assert (bayes_dir / "parameter_samples.csv").exists()
    assert not (bayes_dir / "correlation.csv").exists()
