"""Output directory helpers for YAML project reports."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def prepare_output_dir(spec) -> Path:
    if spec.project.get("output_dir"):
        output = Path(str(spec.project["output_dir"]))
        if not output.is_absolute():
            output = spec.root_dir / output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = spec.root_dir / "runs" / f"{spec.name}_{timestamp}"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("input", "resolved", "simulation", "fit"):
        (output / name).mkdir(exist_ok=True)
    return output
