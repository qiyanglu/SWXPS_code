from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from swanx.project import run_project

output = run_project(PROJECT_DIR / "project.yaml", progress=True)
print(f"SWANX results written to: {output}")
