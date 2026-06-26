from pathlib import Path

from swanx.project import run_project


run_project(Path(__file__).with_name("project_minimal.yaml"))
