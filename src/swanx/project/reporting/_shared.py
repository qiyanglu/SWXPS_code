"""Shared low-level report serialization helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _status_dict(result: Any, *, objective_attr: str) -> dict[str, Any]:
    return {
        "status": getattr(result, "status", None),
        "message": getattr(result, "message", None),
        "success": getattr(result, "success", None),
        "objective": getattr(result, objective_attr, None),
    }

def _write_array(path: Path, value: Any, column_name: str) -> None:
    if value is None:
        return
    array = np.asarray(value)
    if array.ndim == 1:
        _write_csv(path, [["index", column_name], *[[index, item] for index, item in enumerate(array)]])
    elif array.ndim == 2:
        _write_csv(path, [["row", "column", column_name], *[
            [row, column, array[row, column]]
            for row in range(array.shape[0])
            for column in range(array.shape[1])
        ]])

def _sigma_or_empty(sigma, count: int) -> list[Any]:
    return [""] * count if sigma is None else list(sigma)

def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=_json_default)

def _write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
