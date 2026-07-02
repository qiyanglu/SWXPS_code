from __future__ import annotations

import builtins
import csv
from dataclasses import dataclass
import importlib
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

from swanx.fitting import FitContribution, JointObjective, FitParameter, evaluation_from_contributions
from swanx.project import init_project, inspect_project
from swanx.project.builder import build_project, project_polarization
from swanx.project.reports import (
    write_fit_files,
    write_identifiability_outputs,
    write_markdown_report,
    write_method_outputs,
    write_next_project_outputs,
)
from swanx.project.jax_fixed_grid import build_projectspec_jax_residual_function
from swanx.project.runner import _load_callable, run_project, validate_project
from swanx.project.spec import ProjectValidationError, load_project_spec
from swanx.project.yaml_io import YAML_INSTALL_MESSAGE, read_yaml
from swanx.preprocessing import normalize_rocking_curve
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
        "400 4.0\n"
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


def _write_synthetic_case_curve(path: Path) -> None:
    rows = ["angle_deg,reflectivity,la4d_rc,o1s_rc,ti2p_rc,c1s_rc"]
    for index in range(21):
        angle = 5.5 + 0.1 * index
        rows.append(
            f"{angle:.1f},"
            f"{0.0008 + 0.00002 * index:.8f},"
            f"{1.00 + 0.01 * np.sin(index / 3):.8f},"
            f"{1.01 + 0.01 * np.cos(index / 4):.8f},"
            f"{0.99 + 0.01 * np.sin(index / 5):.8f},"
            f"{1.02 + 0.01 * np.cos(index / 6):.8f}"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_reflectivity_and_la4d_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(5.0, 7.0, 21)
    reflectivity_rows = ["angle_deg,reflectivity"]
    rocking_rows = ["angle_deg,intensity"]
    background = 1.2 + 0.04 * angles + 0.003 * angles**2
    raw_rocking = background * (1.0 + 0.04 * np.sin(np.linspace(0.0, 2.0 * np.pi, angles.size)))
    for index, angle in enumerate(angles):
        reflectivity_rows.append(f"{angle:.3f},{1.0e-3 + index * 1.0e-5:.8f}")
        rocking_rows.append(f"{angle:.3f},{raw_rocking[index]:.10f}")
    (path / "reflectivity.csv").write_text("\n".join(reflectivity_rows) + "\n", encoding="utf-8")
    (path / "la4d.csv").write_text("\n".join(rocking_rows) + "\n", encoding="utf-8")
    return angles, raw_rocking


def _fixed_grid_jax_project(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    angles, raw_rocking = _write_reflectivity_and_la4d_data(tmp_path)
    datasets = '''
  reflectivity:
    path: "reflectivity.csv"
    name: "R"
  rocking_curves:
    - path: "la4d.csv"
      name: "La 4d"
'''
    path = _project_yaml(tmp_path, datasets=datasets, fit_method="jax_least_squares")
    text = path.read_text(encoding="utf-8").replace(
        '  normalization: "mean"\n',
        '  normalization: "edge_polynomial"\n'
        '  normalization_edge_fraction: 0.10\n'
        '  normalization_polynomial_order: 2\n'
        '  slicing:\n'
        '    mode: "fixed_grid"\n'
        '    min_slices: 2\n'
        '    max_slice_thickness_A: 5.0\n',
    )
    path.write_text(text, encoding="utf-8")
    return path, angles, raw_rocking




def _write_example_data_root(path: Path) -> Path:
    (path / "OPC").mkdir(parents=True, exist_ok=True)
    (path / "IMFP").mkdir(parents=True, exist_ok=True)
    (path / "curves").mkdir(parents=True, exist_ok=True)
    _write_opc(path / "OPC" / "C.dat", 0.05)
    _write_opc(path / "OPC" / "LaNiO3.dat", 0.1)
    _write_opc(path / "OPC" / "SrTiO3.dat", 0.2)
    _write_imfp(path / "IMFP" / "C.ANG")
    _write_imfp(path / "IMFP" / "LNO.ANG")
    _write_imfp(path / "IMFP" / "STO.ANG")
    _write_synthetic_case_curve(path / "curves" / "lno_sto_c_synthetic_data.csv")
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


def test_run_section_controls_mode_optimizer_and_outputs(tmp_path):
    path = _project_yaml(tmp_path)
    text = path.read_text(encoding="utf-8")
    text = text.replace('  fit_method: "simulate_only"\n', "")
    text = text.replace("report:\n  save_plots: false\n", "report: {}\n")
    text = text.replace(
        "settings:\n",
        "run:\n"
        '  mode: "jax_least_squares"\n'
        "  optimizer:\n"
        '    residual_function_factory: "factory:build_residual"\n'
        "  outputs:\n"
        "    plots: true\n"
        "    identifiability:\n"
        "      enabled: true\n"
        "      weak_modes: 3\n"
        "    next_project:\n"
        "      best_start: true\n"
        "      reduced: true\n"
        "      low_sensitivity_threshold: 0.05\n"
        "settings:\n",
    )
    path.write_text(text, encoding="utf-8")

    spec = load_project_spec(path)

    assert spec.fit_method == "jax_least_squares"
    assert spec.optimizer_settings["residual_function_factory"] == "factory:build_residual"
    assert spec.save_plots is True
    assert spec.identifiability_options["enabled"] is True
    assert spec.identifiability_options["weak_modes"] == 3
    assert spec.next_project_options["enabled"] is True
    assert spec.next_project_options["best_start"] is True
    assert spec.next_project_options["reduced"] is True
    assert spec.next_project_options["low_sensitivity_threshold"] == 0.05


def test_run_section_conflicting_legacy_mode_fails(tmp_path):
    path = _project_yaml(tmp_path)
    text = path.read_text(encoding="utf-8").replace(
        "settings:\n",
        "run:\n  mode: \"jax_gradient\"\nsettings:\n",
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ProjectValidationError, match="run.mode conflicts"):
        load_project_spec(path)


def test_projectspec_safe_expression_functions_and_repeat_index0(tmp_path):
    extra = '''  - repeat:
      times: 2
      layers:
        - id: "lno_{repeat_index}"
          material: "LNO"
          tags: ["lno_layers"]
          thickness_A: "transition_erf(repeat_index0, lno_thickness, lno_thickness + 4, 0, 1)"
          roughness_A: "linear_map(repeat_index0, 0, 1, min(interface_roughness, 4), max(interface_roughness / 2, 2))"
        - id: "sto_{repeat_index}"
          material: "STO"
          tags: ["sto_layers"]
          thickness_A: "sto_thickness + repeat_index0 + sqrt(4) + erf(0)"
          roughness_A: "$interface_roughness"
'''
    path = _project_yaml(tmp_path, extra_stack=extra)
    spec = load_project_spec(path)

    layers = spec.layer_specs_for_values(spec.default_parameter_values())
    expected_transition_0 = 0.5 * (1.0 + math.erf(0.0 / math.sqrt(2.0)))
    expected_transition_1 = 0.5 * (1.0 + math.erf(1.0 / math.sqrt(2.0)))
    assert layers[1]["thickness"] == pytest.approx(40.0 + 4.0 * expected_transition_0)
    assert layers[3]["thickness"] == pytest.approx(40.0 + 4.0 * expected_transition_1)
    assert layers[1]["roughness"] == pytest.approx(3.0)
    assert layers[3]["roughness"] == pytest.approx(2.0)
    assert layers[2]["thickness"] == pytest.approx(12.0)
    assert layers[4]["thickness"] == pytest.approx(13.0)


def test_projectspec_rejects_unknown_or_unsafe_expression_functions(tmp_path):
    path = _project_yaml(
        tmp_path,
        extra_stack='''  - id: "film"
    material: "LNO"
    tags: ["lno_layers"]
    thickness_A: "unknown_function(1)"
    roughness_A: 1.0
''',
    )
    with pytest.raises(ProjectValidationError, match="unknown function"):
        load_project_spec(path)

    path = _project_yaml(
        tmp_path,
        extra_stack='''  - id: "film"
    material: "LNO"
    tags: ["lno_layers"]
    thickness_A: "__import__('os').system('echo unsafe')"
    roughness_A: 1.0
''',
    )
    with pytest.raises(ProjectValidationError, match="unknown function|may contain only"):
        load_project_spec(path)


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
    assert spec.run == {}




def test_swanx_init_generated_project_validates_and_runs_from_different_cwd(monkeypatch, tmp_path):
    start_cwd = tmp_path / "start_without_data"
    start_cwd.mkdir()
    monkeypatch.chdir(start_cwd)
    project_dir = tmp_path / "my_project"
    assert cli_main(["init", str(project_dir)]) == 0

    assert (project_dir / "project.yaml").exists()
    assert (project_dir / "run_project.py").exists()
    assert (project_dir / "README.md").exists()
    assert not (project_dir / "synthetic_residual_factory.py").exists()
    assert (project_dir / "data" / "OPC" / "C.dat").exists()
    assert (project_dir / "data" / "OPC" / "LaNiO3.dat").exists()
    assert (project_dir / "data" / "IMFP" / "C.ANG").exists()
    assert (project_dir / "data" / "IMFP" / "LNO.ANG").exists()
    assert (project_dir / "data" / "curves" / "lno_sto_c_synthetic_data.csv").exists()
    starter_yaml = (project_dir / "project.yaml").read_text(encoding="utf-8")
    assert 'mode: "jax_least_squares"' in starter_yaml
    assert 'residual: "auto_fixed_grid"' in starter_yaml
    assert 'normalization: "edge_polynomial"' in starter_yaml
    assert "normalization_edge_fraction: 0.10" in starter_yaml
    assert 'opc_file: "data/OPC/C.dat"' in starter_yaml
    assert 'opc_file: "data/OPC/LaNiO3.dat"' in starter_yaml
    assert 'times: 20' in starter_yaml
    assert 'id: "carbon_cap"' in starter_yaml
    assert 'name: "C 1s"' in starter_yaml
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
    assert "[swanx] Building ProjectSpec auto fixed-grid JAX residual" in completed.stdout
    assert "[swanx] Running jax_least_squares" in completed.stdout
    assert "[swanx] Simulating final curves" in completed.stdout
    assert "SWANX results written to:" in completed.stdout
    outputs = list((project_dir / "runs").glob("my_project_*"))
    assert outputs
    assert (outputs[-1] / "report.md").exists()
    assert (outputs[-1] / "fit" / "best_parameters.csv").exists()
    assert (outputs[-1] / "plots" / "fit_overview.png").exists()


def test_swanx_init_copy_example_data_and_data_root(tmp_path):
    data_root = _write_example_data_root(tmp_path / "custom_data")

    copied_project = tmp_path / "copied_project"
    assert cli_main(["init", str(copied_project), "--copy-example-data", "--data-root", str(data_root)]) == 0
    assert (copied_project / "data" / "OPC" / "C.dat").exists()
    assert (copied_project / "data" / "OPC" / "LaNiO3.dat").exists()
    assert (copied_project / "data" / "IMFP" / "C.ANG").exists()
    assert (copied_project / "data" / "IMFP" / "LNO.ANG").exists()
    assert (copied_project / "data" / "curves" / "lno_sto_c_synthetic_data.csv").exists()
    copied_yaml = (copied_project / "project.yaml").read_text(encoding="utf-8")
    assert not (copied_project / "synthetic_residual_factory.py").exists()
    assert 'mode: "jax_least_squares"' in copied_yaml
    assert 'residual: "auto_fixed_grid"' in copied_yaml
    assert 'normalization: "edge_polynomial"' in copied_yaml
    assert "normalization_edge_fraction: 0.10" in copied_yaml
    assert 'opc_file: "data/OPC/C.dat"' in copied_yaml
    assert 'opc_file: "data/OPC/LaNiO3.dat"' in copied_yaml
    assert 'times: 20' in copied_yaml
    assert validate_project(copied_project / "project.yaml").name == "copied_project"

    rooted_project = tmp_path / "rooted_project"
    assert cli_main(["init", str(rooted_project), "--data-root", str(data_root)]) == 0
    rooted_yaml = (rooted_project / "project.yaml").read_text(encoding="utf-8")
    assert "../custom_data/OPC/C.dat" in rooted_yaml
    assert "../custom_data/OPC/LaNiO3.dat" in rooted_yaml
    assert "../custom_data/curves/lno_sto_c_synthetic_data.csv" in rooted_yaml
    assert validate_project(rooted_project / "project.yaml").name == "rooted_project"


def test_swanx_init_templates_validate_and_minimal_runs(tmp_path):
    for template in ("minimal", "multilayer", "fit-demo", "fit", "simulate"):
        project_dir = tmp_path / f"project_{template.replace('-', '_')}"
        assert cli_main(["init", str(project_dir), "--template", template]) == 0
        spec = validate_project(project_dir / "project.yaml")
        assert spec.name == project_dir.name
        if template in {"multilayer", "simulate"}:
            assert spec.fit_method == "simulate_only"
        else:
            assert spec.fit_method == "jax_least_squares"
    output = run_project(tmp_path / "project_multilayer" / "project.yaml")
    assert output.parent == tmp_path / "project_multilayer" / "runs"


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
    assert "[Doctor]" in captured
    assert "material files: OK" in captured
    assert "dataset files: OK" in captured
    assert "matplotlib:" in captured
    assert "plot consequence:" in captured
    assert "jax_least_squares deps:" in captured
    assert "bayesian_optimization deps: scikit-optimize=" in captured
    assert "auto_fixed_grid readiness:" in captured
    assert "mode is jax_least_squares: no" in captured
    assert "settings.slicing.mode is fixed_grid: no" in captured

    direct = inspect_project(project_dir / "project.yaml")
    assert "fit_method: simulate_only" in direct

    fit_project = tmp_path / "inspect_fit_project"
    assert cli_main(["init", str(fit_project), "--template", "fit"]) == 0
    fit_direct = inspect_project(fit_project / "project.yaml")
    assert "mode is jax_least_squares: yes" in fit_direct
    assert "residual is auto_fixed_grid: yes" in fit_direct
    assert "datasets exist: yes" in fit_direct
    assert "settings.slicing.mode is fixed_grid: yes" in fit_direct

    (fit_project / "data" / "OPC" / "C.dat").unlink()
    missing_direct = inspect_project(fit_project / "project.yaml")
    assert "[Doctor]" in missing_direct
    assert "validation_error:" in missing_direct
    assert "material files: MISSING" in missing_direct
    assert "MISSING materials.C.opc_file:" in missing_direct


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


def test_projectspec_advanced_fitting_settings_flow_to_problem(tmp_path):
    _write_curve(tmp_path / "reflectivity.csv", "reflectivity")
    (tmp_path / "la4d.csv").write_text(
        "angle_deg,intensity\n"
        "5.0,1.0\n"
        "6.0,1.1\n"
        "7.0,1.0\n",
        encoding="utf-8",
    )
    datasets = '''
  reflectivity:
    path: "reflectivity.csv"
    name: "reflectivity"
  rocking_curves:
    - path: "la4d.csv"
      name: "La 4d"
      normalization: "edge_polynomial"
'''
    path = _project_yaml(tmp_path, datasets=datasets, fit_method="bayesian_optimization")
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        '  fit_method: "bayesian_optimization"\n',
        '  fit_method: "bayesian_optimization"\n'
        '  angle_offset_parameter:\n'
        '  reflectivity_angle_offset_parameter: "reflectivity_angle_offset"\n'
        '  rocking_curve_angle_offset_parameter: "rc_angle_offset"\n'
        '  normalization_edge_fraction: 0.2\n'
        '  normalization_polynomial_order: 1\n',
    )
    text = text.replace(
        '  repeat_center:\n    value: 20.0\n    vary: false\n',
        '  repeat_center:\n    value: 20.0\n    vary: false\n'
        '  reflectivity_angle_offset:\n    initial: 0.01\n    lower: -0.1\n    upper: 0.1\n'
        '  rc_angle_offset:\n    initial: -0.02\n    lower: -0.1\n    upper: 0.1\n',
    )
    text = text.replace(
        '    binding_energy_ev: 105.0\n',
        '    binding_energy_ev: 105.0\n    vacuum_imfp_from_material: "LNO"\n',
    )
    path.write_text(text, encoding="utf-8")

    built = build_project(load_project_spec(path))

    assert built.fitting_problem is not None
    assert built.fitting_problem.angle_offset_parameter is None
    assert built.fitting_problem.reflectivity_angle_offset_parameter == "reflectivity_angle_offset"
    assert built.fitting_problem.rocking_curve_angle_offset_parameter == "rc_angle_offset"
    assert built.fitting_problem.normalization_edge_fraction == pytest.approx(0.2)
    assert built.fitting_problem.normalization_polynomial_order == 1
    assert built.core_levels[0].imfp_by_material["vacuum"] == pytest.approx(
        built.core_levels[0].imfp_by_material["LNO"]
    )


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


def test_jax_least_squares_auto_residual_requires_fixed_grid(tmp_path):
    _write_curve(tmp_path / "reflectivity.csv", "reflectivity")
    datasets = '''
  reflectivity:
    path: "reflectivity.csv"
    name: "R"
'''
    path = _project_yaml(tmp_path, datasets=datasets, fit_method="jax_least_squares")

    with pytest.raises(ProjectValidationError, match="auto_fixed_grid.*settings.slicing.mode"):
        load_project_spec(path)


def test_jax_least_squares_auto_residual_accepts_edge_polynomial_normalization(tmp_path):
    pytest.importorskip("jax")
    path, _angles, _raw_rocking = _fixed_grid_jax_project(tmp_path)
    built = build_project(load_project_spec(path))
    assert built.fitting_problem is not None

    residual = build_projectspec_jax_residual_function(built)
    vector = [parameter.initial for parameter in built.fitting_problem.parameters]
    values = residual(vector)

    assert values.shape == (42,)
    assert np.all(np.isfinite(values))


def test_auto_fixed_grid_jax_model_matches_numpy_simulation(tmp_path):
    pytest.importorskip("jax")
    from swanx.project.jax_fixed_grid import _ProjectSpecFixedGridJaxModel

    path, _angles, _raw_rocking = _fixed_grid_jax_project(tmp_path)
    built = build_project(load_project_spec(path))
    assert built.fitting_problem is not None
    problem = built.fitting_problem
    vector = np.asarray([parameter.initial for parameter in problem.parameters], dtype=float)
    values = {parameter.name: value for parameter, value in zip(problem.parameters, vector)}
    numpy_simulation = problem.simulate(values)
    model = _ProjectSpecFixedGridJaxModel(
        built=built,
        plan=problem.slicing,
        angles=problem.reflectivity.angles,
        offpeak_mask=np.ones(problem.reflectivity.angles.shape, dtype=bool),
        normalization_mode=problem.rocking_curve_normalization,
        normalization_edge_fraction=problem.normalization_edge_fraction,
        normalization_polynomial_order=problem.normalization_polynomial_order,
    )

    jax_reflectivity, jax_curves = model.simulate_curves(vector)

    np.testing.assert_allclose(
        np.asarray(jax_reflectivity),
        numpy_simulation.reflectivity.reflectivity,
        rtol=5.0e-6,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.asarray(jax_curves[0]),
        numpy_simulation.rocking_curves.core_levels[0].curve.intensity,
        rtol=5.0e-6,
        atol=1.0e-8,
    )


def test_auto_fixed_grid_jacobian_matches_finite_difference(tmp_path):
    pytest.importorskip("jax")
    path, _angles, _raw_rocking = _fixed_grid_jax_project(tmp_path)
    built = build_project(load_project_spec(path))
    assert built.fitting_problem is not None
    problem = built.fitting_problem
    residual = build_projectspec_jax_residual_function(built)
    vector = np.asarray([parameter.initial for parameter in problem.parameters], dtype=float)
    jacobian = residual.jacobian(vector)

    for parameter_index in (0, 1):
        step = 1.0e-5 * (
            problem.parameters[parameter_index].upper
            - problem.parameters[parameter_index].lower
        )
        plus = vector.copy()
        minus = vector.copy()
        plus[parameter_index] += step
        minus[parameter_index] -= step
        finite_difference = (residual(plus) - residual(minus)) / (2.0 * step)
        np.testing.assert_allclose(
            jacobian[:, parameter_index],
            finite_difference,
            rtol=2.0e-3,
            atol=2.0e-5,
        )


def test_edge_polynomial_normalization_parity_across_projectspec_paths(tmp_path):
    pytest.importorskip("jax")
    from swanx.project.jax_fixed_grid import _ProjectSpecFixedGridJaxModel

    path, angles, raw_rocking = _fixed_grid_jax_project(tmp_path)
    built = build_project(load_project_spec(path))
    assert built.fitting_problem is not None
    problem = built.fitting_problem
    expected_data, _ = normalize_rocking_curve(
        angles,
        raw_rocking,
        mode="edge_polynomial",
        edge_fraction=0.10,
        polynomial_order=2,
    )
    np.testing.assert_allclose(built.rocking_curve_data[0].intensity, expected_data)

    vector = np.asarray([parameter.initial for parameter in problem.parameters], dtype=float)
    values = {parameter.name: value for parameter, value in zip(problem.parameters, vector)}
    generic_simulation = problem.simulate(values)
    model = _ProjectSpecFixedGridJaxModel(
        built=built,
        plan=problem.slicing,
        angles=problem.reflectivity.angles,
        offpeak_mask=np.ones(problem.reflectivity.angles.shape, dtype=bool),
        normalization_mode=problem.rocking_curve_normalization,
        normalization_edge_fraction=problem.normalization_edge_fraction,
        normalization_polynomial_order=problem.normalization_polynomial_order,
    )
    _jax_reflectivity, jax_curves = model.simulate_curves(vector)
    np.testing.assert_allclose(
        np.asarray(jax_curves[0]),
        generic_simulation.rocking_curves.core_levels[0].curve.intensity,
        rtol=5.0e-6,
        atol=1.0e-8,
    )

    simulate_only = _project_yaml(tmp_path / "simulate_only")
    text = simulate_only.read_text(encoding="utf-8").replace(
        '  normalization: "mean"\n',
        '  normalization: "edge_polynomial"\n'
        '  normalization_edge_fraction: 0.20\n'
        '  normalization_polynomial_order: 2\n',
    )
    text = text.replace("  angle_count: 2\n", "  angle_count: 21\n")
    simulate_only.write_text(text, encoding="utf-8")
    output = run_project(simulate_only)
    rows = [
        row
        for row in csv.DictReader((output / "simulation" / "rocking_curves_simulated.csv").open(encoding="utf-8"))
        if row["core_level"] == "La 4d"
    ]
    output_values = np.asarray([float(row["intensity"]) for row in rows], dtype=float)
    raw_values = np.asarray([float(row["raw_intensity"]) for row in rows], dtype=float)
    output_angles = np.asarray([float(row["angle_deg"]) for row in rows], dtype=float)
    expected_output, _ = normalize_rocking_curve(
        output_angles,
        raw_values,
        mode="edge_polynomial",
        edge_fraction=0.20,
        polynomial_order=2,
    )
    np.testing.assert_allclose(output_values, expected_output)


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


def test_projectspec_offpeak_mask_normalizes_experimental_rocking_data(tmp_path):
    (tmp_path / "reflectivity.csv").write_text(
        "angle_deg,reflectivity\n"
        "5.0,1.0\n"
        "6.0,10.0\n"
        "7.0,1.0\n",
        encoding="utf-8",
    )
    (tmp_path / "la4d.csv").write_text(
        "angle_deg,intensity\n"
        "5.0,2.0\n"
        "6.0,100.0\n"
        "7.0,4.0\n",
        encoding="utf-8",
    )
    datasets = '''
  reflectivity:
    path: "reflectivity.csv"
    name: "R"
  rocking_curves:
    - path: "la4d.csv"
      name: "La 4d"
      normalization: "mean"
'''
    path = _project_yaml(tmp_path, datasets=datasets)
    text = path.read_text(encoding="utf-8").replace(
        '  fit_method: "simulate_only"\n',
        '  fit_method: "simulate_only"\n'
        '  rocking_curve_offpeak_mask:\n'
        '    mode: "exclude_reflectivity_peak"\n'
        '    half_width_deg: 0.1\n',
    )
    path.write_text(text, encoding="utf-8")

    built = build_project(load_project_spec(path))

    assert built.fitting_problem is not None
    np.testing.assert_array_equal(built.fitting_problem.offpeak_mask, [True, False, True])
    np.testing.assert_allclose(built.rocking_curve_data[0].intensity, [2.0 / 3.0, 100.0 / 3.0, 4.0 / 3.0])

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
    assert "plots/simulation_overview.png skipped because matplotlib is not installed" in report
    assert "plots/reflectivity_simulation.png skipped because matplotlib is not installed" in report
    assert "plots/rocking_curves_simulation.png skipped because matplotlib is not installed" in report
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

    assert (output / "plots" / "simulation_overview.png").exists()
    assert (output / "plots" / "reflectivity_simulation.png").exists()
    assert (output / "plots" / "rocking_curves_simulation.png").exists()
    assert (output / "plots" / "stack_schematic.png").exists()
    assert not (output / "plots" / "residuals.png").exists()
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "plots/simulation_overview.png written with experimental overlays: reflectivity, La 4d" in report
    assert "plots/reflectivity_simulation.png written with experimental overlay" in report
    assert "plots/rocking_curves_simulation.png written with experimental overlays: La 4d" in report
    assert "plots/stack_schematic.png written from the final stack" in report


def test_simulation_only_overview_colors_rocking_curves_without_data(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib.axes

    plotted_colors = []
    original_plot = matplotlib.axes.Axes.plot

    def recording_plot(self, *args, **kwargs):
        if kwargs.get("label") == "simulation":
            plotted_colors.append(kwargs.get("color"))
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", recording_plot)
    path = _project_yaml(
        tmp_path,
        output_dir=(tmp_path / "out_simulation_colors").as_posix(),
        report="  save_plots: true\n",
    )

    output = run_project(path)

    assert (output / "plots" / "simulation_overview.png").exists()
    assert "tab:purple" in plotted_colors
    assert "black" not in plotted_colors


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


def test_identifiability_report_writer_uses_run_outputs_switch(tmp_path):
    project_dir = tmp_path / "project"
    path = _project_yaml(project_dir, fit_method="jax_least_squares")
    text = path.read_text(encoding="utf-8").replace(
        "report:\n  save_plots: false\n",
        "run:\n"
        "  outputs:\n"
        "    identifiability: true\n"
        "report:\n"
        "  save_plots: false\n",
    )
    path.write_text(text, encoding="utf-8")
    built = build_project(load_project_spec(path))
    output = tmp_path / "output"
    (output / "fit").mkdir(parents=True)
    (output / "fit" / "residuals.csv").write_text(
        "dataset,angle_deg,experimental,simulated,residual\n"
        "reflectivity,5.0,1.0,0.9,0.1\n"
        "reflectivity,6.0,1.1,1.0,0.1\n"
        "La 4d,5.0,1.0,1.1,-0.1\n"
        "La 4d,6.0,1.1,1.0,0.1\n",
        encoding="utf-8",
    )

    notes = write_identifiability_outputs(output, _Result(), built)

    ident_dir = output / "identifiability_analysis"
    assert "identifiability_analysis/summary.md written from scaled least-squares Jacobian" in notes
    assert (ident_dir / "summary.md").exists()
    assert (ident_dir / "parameter_identifiability.csv").exists()
    assert (ident_dir / "dataset_sensitivity.csv").exists()
    summary = (ident_dir / "summary.md").read_text(encoding="utf-8")
    assert "Dataset Sensitivity Caveat" in summary
    rows = _read_csv(ident_dir / "parameter_identifiability.csv")
    assert rows[0][0] == "parameter"
    assert rows[1][0] == "lno_thickness"

    write_markdown_report(output, built, timestamp="2026-07-01T00:00:00", result=_Result())

    report = (output / "report.md").read_text(encoding="utf-8")
    assert "## Fit Interpretation" in report
    assert "Final least-squares cost: 0.5" in report
    assert "Identifiability analysis: see `identifiability_analysis/summary.md`" in report
    assert "Weakly identifiable parameters:" in report
    assert "Highest weak-mode participation:" in report
    assert "Dataset sensitivity caveat:" in report
    assert "Recommended next checks:" in report
    assert "Review near-bound parameters" in report
    assert "Inspect `identifiability_analysis/summary.md` when present" in report
    assert "Review strong correlations" in report
    assert "Review dataset sensitivity as a weighting/scaling audit signal" in report


def test_next_project_outputs_write_best_start_and_reduced_yaml(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    path, _angles, _raw = _fixed_grid_jax_project(project_dir)
    text = path.read_text(encoding="utf-8").replace(
        "report:\n  save_plots: false\n",
        "run:\n"
        "  outputs:\n"
        "    next_project:\n"
        "      best_start: true\n"
        "      reduced: true\n"
        "      low_sensitivity_threshold: 0.02\n"
        "report:\n"
        "  save_plots: false\n",
    )
    path.write_text(text, encoding="utf-8")
    built = build_project(load_project_spec(path))
    output = tmp_path / "output"
    ident_dir = output / "identifiability_analysis"
    ident_dir.mkdir(parents=True)
    (ident_dir / "parameter_identifiability.csv").write_text(
        "parameter,relative_sensitivity,suggestion\n"
        "lno_thickness,0.01,consider fixing this weak parameter\n"
        "sto_thickness,0.50,keep\n"
        "interface_roughness,0.80,keep\n",
        encoding="utf-8",
    )

    notes = write_next_project_outputs(output, _Result(), built)

    next_dir = output / "next_project"
    assert "next_project/project_best_start.yaml written with best-fit initial values" in notes
    assert "next_project/project_reduced.yaml written with low-sensitivity parameters fixed" in notes
    assert (next_dir / "project_best_start.yaml").exists()
    assert (next_dir / "project_reduced.yaml").exists()
    assert (next_dir / "reduction_notes.md").exists()

    best = read_yaml(next_dir / "project_best_start.yaml")
    assert best["project"]["name"] == "yaml_test_best_start"
    assert "output_dir" not in best["project"]
    assert best["run"]["outputs"]["next_project"] is False
    assert best["parameters"]["lno_thickness"]["initial"] == pytest.approx(41.0)
    assert best["parameters"]["sto_thickness"]["initial"] == pytest.approx(11.0)
    assert best["materials"]["LNO"]["opc_file"].startswith("../")
    assert best["datasets"]["reflectivity"]["path"].startswith("../")
    validate_project(next_dir / "project_best_start.yaml")

    reduced = read_yaml(next_dir / "project_reduced.yaml")
    assert reduced["project"]["name"] == "yaml_test_reduced"
    assert reduced["parameters"]["lno_thickness"] == {"value": 41.0, "vary": False}
    assert reduced["parameters"]["sto_thickness"]["vary"] is True
    validate_project(next_dir / "project_reduced.yaml")
    reduction_notes = (next_dir / "reduction_notes.md").read_text(encoding="utf-8")
    assert "lno_thickness" in reduction_notes
    assert "Low-sensitivity threshold" in reduction_notes


def test_readme_and_project_state_docs_are_current():
    readme = Path("README.md").read_text(encoding="utf-8")
    user_guide = Path("docs/user_guide.md").read_text(encoding="utf-8")
    reference = Path("docs/projectspec_reference.md").read_text(encoding="utf-8")
    project_state = Path("docs/PROJECT_STATE.md").read_text(encoding="utf-8")
    roadmap = Path("docs/roadmap.md").read_text(encoding="utf-8")
    architecture = Path("docs/architecture.md").read_text(encoding="utf-8")
    examples_readme = Path("examples/README.md").read_text(encoding="utf-8")
    fitting_readme = Path("examples/04_fitting/README.md").read_text(encoding="utf-8")
    projectspec_fit_readme = Path(
        "examples/04_fitting/projectspec_jax_least_squares/README.md"
    ).read_text(encoding="utf-8")

    assert readme.index("## Why SWANX?") < readme.index("## Quickstart")
    assert "## Features" in readme
    assert '<p align="center">' in readme
    assert '<img src="swanx_logo.png" alt="SWANX logo" height="320">' in readme
    assert "## ProjectSpec In One Minute" in readme
    assert "## Outputs" in readme
    assert "## Fitting" in readme
    assert "## Installation Options" in readme
    assert "The default init project is self-contained" in readme
    assert "copies" in readme
    assert "C capping layer on a" in readme
    assert "20-repeat LaNiO3/SrTiO3 superlattice mirror" in readme
    assert "40 oxide layers total" in readme
    assert "--template fit`: preferred fitting starter" in readme
    assert "--template minimal`: legacy alias" in readme
    assert "--template fit-demo`: explicit fitting starter alias" in readme
    assert "--template multilayer` / `--template simulate`: simulation-only repeated" in readme
    assert "JAX least-squares" in readme
    assert "simulate X-ray reflectivity" in readme
    assert "the X-ray electric field" in readme
    assert 'residual: "auto_fixed_grid"' in readme
    assert "auto_fixed_grid` is the default YAML residual path" in readme
    assert "next_project/" in readme
    assert "next_project:" in readme
    assert "best_start: true" in readme
    assert "reduced: true" in readme
    assert "optional global black-box baseline" in readme
    assert "BO is not the default fitting method and is not used as a fallback" in readme
    assert "synthetic_residual_factory.py" not in readme
    assert "docs/user_guide.md" in readme
    assert "docs/projectspec_reference.md" in readme
    assert "examples/README.md" in readme
    assert "numbered example folders are intended to be read as a path" in readme
    assert "swanx init my_project" in readme
    assert "swanx inspect" in readme
    assert "repository-level `data/`" not in readme

    for heading in (
        "## Overview",
        "## Core Concepts",
        "## Quickstart: Fitting Starter Project",
        "## Simulate Only And Overlay Data",
        "## Fit Workflow",
        "## How To Inspect And Validate",
        "## How To Read Outputs",
        "## Advanced Python API",
        "## Troubleshooting",
    ):
        assert heading in user_guide
    assert "examples/04_fitting/projectspec_jax_least_squares" in user_guide
    assert "run.outputs.next_project" in user_guide
    assert "project_best_start.yaml" in user_guide
    assert "project_reduced.yaml" in user_guide

    for section in (
        "project",
        "run",
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
    assert "auto_fixed_grid" in reference
    assert "edge_polynomial" in reference
    assert "identifiability" in reference
    assert "outputs.next_project" in reference
    assert "project_best_start.yaml" in reference
    assert "project_reduced.yaml" in reference
    assert "transition_erf" in reference
    assert "linear_map" in reference
    assert "repeat_index0" in reference
    assert "examples/04_fitting/projectspec_jax_least_squares" in reference

    for text_block in (
        project_state,
        roadmap,
        architecture,
        examples_readme,
        fitting_readme,
        projectspec_fit_readme,
    ):
        assert "examples/04_fitting/projectspec_jax_least_squares" in text_block
    assert "run.outputs.next_project" in project_state
    assert "next_project/" in examples_readme
    assert "run.outputs.next_project" in fitting_readme
    assert "project_best_start.yaml" in projectspec_fit_readme
    assert "project_reduced.yaml" in projectspec_fit_readme
    assert "`fit` as the preferred fitting starter" in roadmap
    assert "`simulate` as the preferred" in roadmap
    assert "`minimal` and `fit-demo` remain fitting aliases" in roadmap
    assert "`multilayer` remains a simulation-only alias" in roadmap
    assert "four numbered example folders" in examples_readme

    assert "C:\\Users" not in project_state
    assert not re.search(r"\b\d+\s+passed\b", project_state)
    assert not re.search(r"\b\d+\s+xfailed\b", project_state)
    assert not re.search(r"\b\d+\s+warning", project_state)
    assert "synthetic_residual_factory.py" not in readme
    assert "synthetic_residual_factory.py" not in user_guide
    assert "synthetic_residual_factory.py" not in reference


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


def test_projectspec_fitting_example_validates():
    spec = validate_project(
        Path("examples/04_fitting/projectspec_jax_least_squares/project.yaml")
    )
    assert spec.next_project_options["enabled"] is True
    assert spec.next_project_options["best_start"] is True
    assert spec.next_project_options["reduced"] is True
    assert spec.next_project_options["low_sensitivity_threshold"] == 0.02
