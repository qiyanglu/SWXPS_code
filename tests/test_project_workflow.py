from __future__ import annotations

import builtins
import csv
from dataclasses import dataclass
import importlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from swanx.fitting import FitContribution, JointObjective, FitParameter, evaluation_from_contributions
from swanx.project import init_project, inspect_project
from swanx.project.builder import build_project, project_polarization
from swanx.project.reports import write_fit_files, write_method_outputs
from swanx.project.runner import _load_callable, run_project, validate_project
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




def _write_example_data_root(path: Path) -> Path:
    (path / "OPC").mkdir(parents=True, exist_ok=True)
    (path / "IMFP").mkdir(parents=True, exist_ok=True)
    (path / "curves").mkdir(parents=True, exist_ok=True)
    _write_opc(path / "OPC" / "LaNiO3.dat", 0.1)
    _write_opc(path / "OPC" / "SrTiO3.dat", 0.2)
    _write_imfp(path / "IMFP" / "LNO.ANG")
    _write_imfp(path / "IMFP" / "STO.ANG")
    _write_curve(path / "curves" / "lno_sto_reflectivity.csv", "reflectivity")
    _write_curve(path / "curves" / "la4d_rocking_curve.csv", "intensity")
    return path


def _project_yaml(
    tmp_path: Path,
    *,
    extra_stack: str = "",
    datasets: str = "{}",
    output_dir: str | None = None,
    fit_method: str = "simulate_only",
    report: str = "  save_plots: false\n",
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
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
  fit_method: "{fit_method}"
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
  repeat_center:
    value: 20.0
    vary: false
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
          thickness_A: "sto_thickness + repeat_index - repeat_center / 20"
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
{report}"""
    path = tmp_path / "project.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _read_csv(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


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
    assert layers[2]["thickness"] == pytest.approx(10.0)
    assert layers[4]["thickness"] == pytest.approx(11.0)
    assert layers[2]["roughness"] == pytest.approx(1.5)


def test_optional_sections_default_correctly(tmp_path):
    _write_opc(tmp_path / "LNO.dat")
    _write_imfp(tmp_path / "LNO.ANG")
    path = tmp_path / "minimal_optional.yaml"
    path.write_text(
        """
project:
  name: "optional_defaults"
settings:
  photon_energy_ev: 900.0
  angles_deg: [5.0, 6.0]
  polarization: "s"
  fit_method: "simulate_only"
materials:
  LNO:
    opc_file: "LNO.dat"
    imfp_file: "LNO.ANG"
stack:
  - id: "vacuum"
    material: "vacuum"
  - id: "film"
    material: "LNO"
    tags: ["film"]
    thickness_A: 10.0
    roughness_A: 1.0
core_levels:
  - name: "La 4d"
    binding_energy_ev: 105.0
    emit_from:
      tags: ["film"]
""",
        encoding="utf-8",
    )

    spec = load_project_spec(path)

    assert spec.parameters == {}
    assert spec.datasets == {}
    assert spec.report == {}


def test_template_project_minimal_validates():
    spec = validate_project(Path("templates/project_minimal.yaml"))
    assert spec.name == "minimal_yaml_project"


def test_swanx_init_generated_project_validates_and_runs_from_different_cwd(monkeypatch, tmp_path):
    start_cwd = tmp_path / "start_without_data"
    start_cwd.mkdir()
    monkeypatch.chdir(start_cwd)
    project_dir = tmp_path / "my_project"
    assert cli_main(["init", str(project_dir)]) == 0

    assert (project_dir / "project.yaml").exists()
    assert (project_dir / "run_project.py").exists()
    assert (project_dir / "README.md").exists()
    assert (project_dir / "data" / "OPC" / "LaNiO3.dat").exists()
    assert (project_dir / "data" / "IMFP" / "LNO.ANG").exists()
    starter_yaml = (project_dir / "project.yaml").read_text(encoding="utf-8")
    assert 'opc_file: "data/OPC/LaNiO3.dat"' in starter_yaml
    assert validate_project(project_dir / "project.yaml").name == "my_project"

    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    completed = subprocess.run(
        [sys.executable, str(project_dir / "run_project.py")],
        cwd=other_cwd,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "[swanx] Reading ProjectSpec:" in completed.stdout
    assert "[swanx] Simulating final curves" in completed.stdout
    assert "SWANX results written to:" in completed.stdout
    outputs = list((project_dir / "runs").glob("my_project_*"))
    assert outputs
    assert (outputs[-1] / "report.md").exists()


def test_swanx_init_copy_example_data_and_data_root(tmp_path):
    data_root = _write_example_data_root(tmp_path / "custom_data")

    copied_project = tmp_path / "copied_project"
    assert cli_main(["init", str(copied_project), "--copy-example-data", "--data-root", str(data_root)]) == 0
    assert (copied_project / "data" / "OPC" / "LaNiO3.dat").exists()
    assert (copied_project / "data" / "IMFP" / "LNO.ANG").exists()
    assert (copied_project / "data" / "curves" / "lno_sto_reflectivity.csv").exists()
    copied_yaml = (copied_project / "project.yaml").read_text(encoding="utf-8")
    assert 'opc_file: "data/OPC/LaNiO3.dat"' in copied_yaml
    assert validate_project(copied_project / "project.yaml").name == "copied_project"

    rooted_project = tmp_path / "rooted_project"
    assert cli_main(["init", str(rooted_project), "--data-root", str(data_root)]) == 0
    rooted_yaml = (rooted_project / "project.yaml").read_text(encoding="utf-8")
    assert "../custom_data/OPC/LaNiO3.dat" in rooted_yaml
    assert validate_project(rooted_project / "project.yaml").name == "rooted_project"


def test_swanx_init_templates_validate_and_minimal_runs(tmp_path):
    for template in ("minimal", "multilayer", "fit-demo"):
        project_dir = tmp_path / f"project_{template.replace('-', '_')}"
        assert cli_main(["init", str(project_dir), "--template", template]) == 0
        spec = validate_project(project_dir / "project.yaml")
        assert spec.name == project_dir.name
    output = run_project(tmp_path / "project_minimal" / "project.yaml")
    assert output.parent == tmp_path / "project_minimal" / "runs"


def test_swanx_inspect_prints_expected_sections(tmp_path, capsys):
    project_dir = tmp_path / "inspect_project"
    assert cli_main(["init", str(project_dir), "--template", "multilayer"]) == 0

    assert cli_main(["inspect", str(project_dir / "project.yaml")]) == 0
    captured = capsys.readouterr().out
    assert "[Project]" in captured
    assert "[Materials]" in captured
    assert "[Stack]" in captured
    assert "layer_count:" in captured
    assert "[Core Levels]" in captured
    assert "[Datasets]" in captured
    assert "[Varying Parameters]" in captured
    assert "[Optional Dependencies]" in captured
    assert "[Fitting Callbacks]" in captured
    assert "callback_required: no" in captured

    direct = inspect_project(project_dir / "project.yaml")
    assert "fit_method: simulate_only" in direct


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
    tags: ["lno_layers"]
    thickness_A: "$missing_parameter"
    roughness_A: 1.0
''',
    )
    with pytest.raises(ProjectValidationError, match="unknown parameter"):
        load_project_spec(path)

    path = _project_yaml(tmp_path)
    text = path.read_text(encoding="utf-8").replace('tags: ["lno_layers"]', 'tags: ["missing_tag"]', 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ProjectValidationError, match="unknown tag"):
        load_project_spec(path)

    path = _project_yaml(tmp_path)
    (tmp_path / "LNO.dat").unlink()
    with pytest.raises(ProjectValidationError, match="missing data file"):
        load_project_spec(path)


def test_emit_from_required_unless_all_true(tmp_path):
    path = _project_yaml(tmp_path)
    text = path.read_text(encoding="utf-8").replace('    emit_from:\n      tags: ["lno_layers"]\n', "")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ProjectValidationError, match="requires emit_from"):
        load_project_spec(path)

    path = _project_yaml(tmp_path)
    text = path.read_text(encoding="utf-8").replace('tags: ["lno_layers"]', 'all: true')
    path.write_text(text, encoding="utf-8")
    built = build_project(load_project_spec(path))
    assert built.core_levels[0].emitting_layer_indices is None


def test_emitting_material_missing_imfp_and_stack_material_missing_opc_fail(tmp_path):
    path = _project_yaml(tmp_path)
    text = path.read_text(encoding="utf-8").replace('    imfp_file: "LNO.ANG"\n', "", 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ProjectValidationError, match="emitting material 'LNO'.*imfp_file"):
        load_project_spec(path)

    path = _project_yaml(tmp_path)
    text = path.read_text(encoding="utf-8").replace('    opc_file: "LNO.dat"\n', "", 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ProjectValidationError, match="non-vacuum stack material 'LNO'.*opc_file"):
        load_project_spec(path)


def test_vary_false_parameters_are_constants_not_fit_parameters(tmp_path):
    _write_curve(tmp_path / "reflectivity.csv", "reflectivity")
    datasets = '''
  reflectivity:
    path: "reflectivity.csv"
    name: "R"
'''
    path = _project_yaml(tmp_path, datasets=datasets, fit_method="bayesian_optimization")
    built = build_project(load_project_spec(path))

    assert built.values["repeat_center"] == 20.0
    assert built.fitting_problem is not None
    assert "repeat_center" not in [parameter.name for parameter in built.fitting_problem.parameters]


def test_core_level_layer_tag_resolution_and_polarization(tmp_path):
    path = _project_yaml(tmp_path)
    built = build_project(load_project_spec(path))

    assert built.core_levels[0].emitting_layer_indices == (1, 3)
    assert project_polarization("s") == "s"
    assert project_polarization("p") == "p"
    assert project_polarization("unpolarized") == {"s": 0.5, "p": 0.5}


def test_optimizer_factory_imports_relative_to_project_yaml(tmp_path, monkeypatch):
    factory = tmp_path / "local_factory.py"
    factory.write_text(
        "def make(problem):\n"
        "    return {'problem': problem}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path.parent)

    callback = _load_callable("local_factory:make", tmp_path)

    assert callback("ok") == {"problem": "ok"}


def test_jax_least_squares_requires_factory_without_bo_fallback(tmp_path):
    _write_curve(tmp_path / "reflectivity.csv", "reflectivity")
    datasets = '''
  reflectivity:
    path: "reflectivity.csv"
    name: "R"
'''
    path = _project_yaml(tmp_path, datasets=datasets, fit_method="jax_least_squares")

    with pytest.raises(ProjectValidationError, match="residual_function_factory.*Bayesian optimization is not used as a fallback"):
        load_project_spec(path)


def test_jax_gradient_requires_factory_without_bo_fallback(tmp_path):
    _write_curve(tmp_path / "reflectivity.csv", "reflectivity")
    datasets = '''
  reflectivity:
    path: "reflectivity.csv"
    name: "R"
'''
    path = _project_yaml(tmp_path, datasets=datasets, fit_method="jax_gradient")

    with pytest.raises(ProjectValidationError, match="value_and_grad_factory.*Bayesian optimization is not used as a fallback"):
        load_project_spec(path)


def test_validate_run_cli_and_simulate_only_outputs(tmp_path, capsys):
    output_dir = (tmp_path / "out").as_posix()
    path = _project_yaml(tmp_path, output_dir=output_dir)

    assert validate_project(path).name == "yaml_test"
    assert cli_main(["validate", str(path)]) == 0
    output = run_project(path)
    assert output == tmp_path / "out"
    assert cli_main(["run", str(path)]) == 0
    captured = capsys.readouterr().out
    assert "Project is valid:" in captured
    assert "[swanx] Reading ProjectSpec:" in captured
    assert "[swanx] Fit method is simulate_only; skipping optimizer" in captured
    assert "[swanx] Writing simulation, data, fit, optimizer, and plot reports" in captured
    assert "SWANX results written to:" in captured

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
        "report.md",
    ]
    for relative in expected:
        assert (output / relative).exists()
    assert not (output / "fit" / "best_parameters.csv").exists()
    assert not (output / "fit" / "residuals.csv").exists()
    assert not any((output / "optimizer").rglob("*.csv")) if (output / "optimizer").exists() else True


def test_default_output_goes_under_project_folder_and_report_contains_fields(tmp_path):
    project_dir = tmp_path / "project_dir"
    path = _project_yaml(project_dir)

    output = run_project(path)

    assert output.parent == project_dir / "runs"
    report = output / "report.md"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "# SWANX Project Report: yaml_test" in text
    assert "Fit method: simulate_only" in text
    assert "Photon energy: 900.0 eV" in text
    assert "No fitting was performed" in text
    assert "Used parameter values:" in text
    assert "- lno_thickness: 40.0" in text
    assert "simulation/reflectivity_simulated.csv" in text


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
    assert not (output / "fit" / "best_parameters.csv").exists()


def test_matplotlib_missing_plot_skip_is_recorded(monkeypatch, tmp_path):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib.pyplot":
            raise ImportError("matplotlib unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    path = _project_yaml(
        tmp_path,
        output_dir=(tmp_path / "out_no_matplotlib").as_posix(),
        report="  save_plots: true\n",
    )

    output = run_project(path)

    assert not (output / "plots").exists()
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "plots/fit_overview.png skipped because matplotlib is not installed" in report
    assert "plots/reflectivity_fit.png skipped because matplotlib is not installed" in report
    assert "plots/rocking_curves_fit.png skipped because matplotlib is not installed" in report
    assert "plots/stack_schematic.png skipped because matplotlib is not installed" in report


def test_plots_overlay_experimental_data_when_matplotlib_exists(tmp_path):
    pytest.importorskip("matplotlib")
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
    path = _project_yaml(
        tmp_path,
        datasets=datasets,
        output_dir=(tmp_path / "out_plots").as_posix(),
        report="  save_plots: true\n",
    )

    output = run_project(path)

    assert (output / "plots" / "fit_overview.png").exists()
    assert (output / "plots" / "reflectivity_fit.png").exists()
    assert (output / "plots" / "rocking_curves_fit.png").exists()
    assert (output / "plots" / "stack_schematic.png").exists()
    assert not (output / "plots" / "residuals.png").exists()
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "plots/fit_overview.png written with experimental overlays: reflectivity, La 4d" in report
    assert "plots/reflectivity_fit.png written with experimental overlay" in report
    assert "plots/rocking_curves_fit.png written with experimental overlays: La 4d" in report
    assert "plots/stack_schematic.png written from the final stack" in report


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
    best_parameters = {"lno_thickness": 41.0, "sto_thickness": 11.0, "interface_roughness": 2.5}
    final_cost = 0.5
    best_loss = 0.4
    best_objective = 0.3
    final_residuals = np.array([1.0, 2.0, 3.0, 4.0])
    final_jacobian = np.array([[1.0, 0.0, 0.2], [0.0, 1.0, 0.1], [0.3, 0.0, 1.0], [0.0, 0.4, 1.0]])
    covariance = np.eye(3)
    final_gradient = np.array([0.1, 0.2, 0.3])
    history = (_Record(1, parameters={"lno_thickness": 41.0}),)


def test_fitting_report_writes_best_parameters_with_bounds(tmp_path):
    _write_curve(tmp_path / "reflectivity.csv", "reflectivity")
    datasets = '''
  reflectivity:
    path: "reflectivity.csv"
    name: "R"
'''
    path = _project_yaml(tmp_path, datasets=datasets, fit_method="bayesian_optimization")
    built = build_project(load_project_spec(path))
    assert built.fitting_problem is not None
    simulation = built.fitting_problem.simulate(built.values)
    evaluation = built.fitting_problem.evaluate(built.values)

    write_fit_files(tmp_path, built, simulation, evaluation, _Result())

    rows = _read_csv(tmp_path / "fit" / "best_parameters.csv")
    assert rows[0] == ["name", "initial", "lower", "upper", "best_value"]
    assert ["lno_thickness", "40.0", "30.0", "50.0", "41.0"] in rows
    assert not any(row and row[0] == "repeat_center" for row in rows[1:])


def test_method_specific_report_writers(tmp_path):
    built = build_project(load_project_spec(_project_yaml(tmp_path / "project", output_dir=(tmp_path / "unused").as_posix())))
    write_method_outputs(tmp_path, "jax_least_squares", _Result(), built)
    ls_dir = tmp_path / "optimizer" / "least_squares"
    assert (ls_dir / "covariance.csv").exists()
    assert (ls_dir / "correlation.csv").exists()
    uncertainty = _read_csv(ls_dir / "parameter_uncertainty.csv")
    assert uncertainty[0] == ["parameter", "best_value", "stderr", "ci95_low", "ci95_high", "lower", "upper"]
    assert uncertainty[1][0] == "lno_thickness"

    if importlib.util.find_spec("matplotlib") is not None:
        plot_project = tmp_path / "plot_project"
        plot_project.mkdir()
        _write_curve(plot_project / "reflectivity.csv", "reflectivity")
        datasets = '''
  reflectivity:
    path: "reflectivity.csv"
    name: "R"
'''
        plot_built = build_project(load_project_spec(_project_yaml(plot_project, datasets=datasets)))
        write_method_outputs(tmp_path / "plot_outputs", "jax_least_squares", _Result(), plot_built)
        assert (tmp_path / "plot_outputs" / "plots" / "parameter_uncertainty.png").exists()
        assert (tmp_path / "plot_outputs" / "plots" / "parameter_correlation.png").exists()
        assert (tmp_path / "plot_outputs" / "plots" / "convergence.png").exists()

    write_method_outputs(tmp_path, "jax_gradient", _Result())
    gradient_dir = tmp_path / "optimizer" / "gradient"
    assert (gradient_dir / "objective_history.csv").exists()
    assert (gradient_dir / "final_gradient.csv").exists()
    assert not (gradient_dir / "covariance.csv").exists()

    evaluation1 = evaluation_from_contributions(
        {"x": 2.0},
        (FitContribution("synthetic", raw=2.0, weight=1.0),),
    )
    evaluation2 = evaluation_from_contributions(
        {"x": 1.0},
        (FitContribution("synthetic", raw=1.0, weight=1.0),),
    )
    objective = JointObjective((FitParameter("x", 0.0, 2.0),), lambda _: evaluation1)
    history = objective.history.append(evaluation1).append(evaluation2)
    bo_result = type(
        "BOResult",
        (),
        {
            "best_parameters": {"x": 1.0},
            "best_objective": 1.0,
            "history": _History(history.evaluations),
            "predict_objective": lambda self, vectors: (
                np.sum(np.asarray(vectors, dtype=float) ** 2, axis=1),
                np.full(np.asarray(vectors).shape[0], 0.1),
            ),
        },
    )()
    write_method_outputs(tmp_path, "bayesian_optimization", bo_result)
    bayes_dir = tmp_path / "optimizer" / "bayesian"
    assert _read_csv(bayes_dir / "evaluations.csv")[0] == ["evaluation", "objective", "parameters_json"]
    assert _read_csv(bayes_dir / "best_so_far.csv")[0] == ["evaluation", "best_objective", "best_parameters_json"]
    assert (bayes_dir / "parameter_samples.csv").exists()
    if importlib.util.find_spec("matplotlib") is not None:
        assert (tmp_path / "plots" / "convergence.png").exists()
    assert not (bayes_dir / "covariance.csv").exists()
    assert not (bayes_dir / "correlation.csv").exists()


def test_readme_and_project_state_docs_are_current():
    readme = Path("README.md").read_text(encoding="utf-8")
    user_guide = Path("docs/user_guide.md").read_text(encoding="utf-8")
    reference = Path("docs/projectspec_reference.md").read_text(encoding="utf-8")
    project_state = Path("docs/PROJECT_STATE.md").read_text(encoding="utf-8")

    assert readme.index("## What problem does SWANX solve?") < readme.index("## Quickstart")
    assert "## What can I do with SWANX?" in readme
    assert "## ProjectSpec overview" in readme
    assert "## Outputs" in readme
    assert "## Fitting" in readme
    assert "## Installation options" in readme
    assert "The generated project is self-contained" in readme
    assert "copies packaged tutorial OPC, IMFP, and curve files" in readme
    assert "thickness_A` and `roughness_A` are in Angstrom" in readme
    assert "repeat_index` is 1-based" in readme
    assert "JAX least-squares" in readme
    assert "optional global black-box baseline" in readme
    assert "BO is not the default fitting method and is not used as a fallback" in readme
    assert "Most of them use a compact LNO/STO tutorial system" in readme
    assert "C/LNO/STO synthetic case" in readme
    assert "docs/user_guide.md" in readme
    assert "docs/projectspec_reference.md" in readme
    assert "examples/README.md" in readme
    assert "swanx init my_project" in readme
    assert "swanx inspect" in readme
    assert "repository-level `data/`" not in readme

    for heading in (
        "## Overview",
        "## Core Concepts",
        "## Quickstart: Simulate-Only Project",
        "## Add Experimental Data And Overlay Points",
        "## Fit Workflow",
        "## How To Inspect And Validate",
        "## How To Read Outputs",
        "## Advanced Python API",
        "## Troubleshooting",
    ):
        assert heading in user_guide

    for section in (
        "project",
        "settings",
        "materials",
        "parameters",
        "stack",
        "core_levels",
        "datasets",
        "report",
    ):
        assert f"{section}:" in reference
    assert "BO is an optional global black-box baseline" in reference

    assert "C:\\Users" not in project_state
    assert "240 passed" not in project_state


def test_projectspec_example_yaml_files_validate():
    examples = Path("examples/01_quickstart_projectspec")
    expected = {
        "minimal_simulate_only.yaml",
        "multilayer_repeat.yaml",
        "compare_with_data.yaml",
        "fit_jax_least_squares_placeholder.yaml",
        "bo_optional_baseline.yaml",
    }
    assert expected <= {path.name for path in examples.glob("*.yaml")}

    for name in expected:
        validate_project(examples / name)
