from pathlib import Path

from swanx.project import run_project


output = run_project(Path(__file__).with_name("project.yaml"), progress=True)
print(f"SWANX example results written to: {output}")
