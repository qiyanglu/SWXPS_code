from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from swanx.project import inspect_project, run_project, validate_project

project_yaml = PROJECT_DIR / "project.yaml"
print(inspect_project(project_yaml))
validate_project(project_yaml)
output = run_project(project_yaml, progress=True)
print(f"SWANX results written to: {output}")
