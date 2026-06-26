"""Lazy YAML loading for optional project workflow support."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


YAML_INSTALL_MESSAGE = (
    "YAML project support requires PyYAML. Install with:\n"
    'python -m pip install -e ".[project]"'
)


def _load_yaml_module():
    try:
        return importlib.import_module("yaml")
    except ImportError as error:
        raise ImportError(YAML_INSTALL_MESSAGE) from error


def read_yaml(path: str | Path) -> Any:
    yaml = _load_yaml_module()
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: str | Path, data: Any) -> None:
    yaml = _load_yaml_module()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
