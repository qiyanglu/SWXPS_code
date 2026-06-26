from pathlib import Path
from swanx.project import run_project

output = run_project(Path(__file__).with_name("project_minimal.yaml"))
print(f"SWANX results written to: {output}")
