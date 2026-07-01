"""Experimental reflectivity and rocking-curve readers."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import numpy as np

from swanx.fitting import ReflectivityData, RockingCurveData
from swanx.preprocessing import normalize_rocking_curve


def read_reflectivity_data(
    path: str | Path,
    *,
    name: str | None = None,
    angle_column: str | int = "angle_deg",
    intensity_column: str | int = "reflectivity",
    sigma_column: str | int | None = None,
    delimiter: str | None = None,
    comments: str = "#",
) -> ReflectivityData:
    """Read an experimental reflectivity curve.

    CSV, whitespace-separated tables, and headerless files with explicit column
    indices are supported. Rows are sorted by angle before the fitting data
    object is returned.
    """

    table = _read_curve_table(
        path,
        angle_column=angle_column,
        intensity_column=intensity_column,
        sigma_column=sigma_column,
        angle_names=_ANGLE_NAMES,
        intensity_names={"reflectivity", "r", "intensity"},
        intensity_label="reflectivity",
        delimiter=delimiter,
        comments=comments,
    )
    return ReflectivityData(
        name=name or Path(path).stem,
        angles=table.angles,
        reflectivity=table.values,
        sigma=table.sigma,
    )


def read_rocking_curve_data(
    path: str | Path,
    *,
    name: str | None = None,
    angle_column: str | int = "angle_deg",
    intensity_column: str | int = "intensity",
    sigma_column: str | int | None = None,
    normalization_mode: str | None = None,
    offpeak_mask: np.ndarray | None = None,
    edge_fraction: float = 0.10,
    polynomial_order: int = 2,
    delimiter: str | None = None,
    comments: str = "#",
) -> RockingCurveData:
    """Read an experimental SW-XPS rocking curve.

    If ``normalization_mode`` is provided, normalization is delegated to
    :func:`swanx.preprocessing.normalize_rocking_curve`.
    """

    table = _read_curve_table(
        path,
        angle_column=angle_column,
        intensity_column=intensity_column,
        sigma_column=sigma_column,
        angle_names=_ANGLE_NAMES,
        intensity_names={"intensity", "i", "counts", "signal"},
        intensity_label="intensity",
        delimiter=delimiter,
        comments=comments,
    )
    values = table.values
    if normalization_mode is not None:
        values, _ = normalize_rocking_curve(
            table.angles,
            values,
            mode=normalization_mode,  # type: ignore[arg-type]
            offpeak_mask=offpeak_mask,
            edge_fraction=edge_fraction,
            polynomial_order=polynomial_order,
        )
    return RockingCurveData(
        name=name or Path(path).stem,
        angles=table.angles,
        intensity=values,
        sigma=table.sigma,
    )


class _CurveTable:
    def __init__(
        self,
        angles: np.ndarray,
        values: np.ndarray,
        sigma: np.ndarray | None,
    ) -> None:
        self.angles = angles
        self.values = values
        self.sigma = sigma


_ANGLE_NAMES = {"angle", "angledeg", "angle_deg", "theta", "theta_deg", "grazing_angle"}


def _read_curve_table(
    path: str | Path,
    *,
    angle_column: str | int,
    intensity_column: str | int,
    sigma_column: str | int | None,
    angle_names: set[str],
    intensity_names: set[str],
    intensity_label: str,
    delimiter: str | None,
    comments: str,
) -> _CurveTable:
    path = Path(path)
    lines = list(_iter_content_lines(path, comments=comments))
    if not lines:
        raise ValueError(f"no curve rows found in {path}")

    first_tokens = _split_line(lines[0][1].strip(), delimiter)
    has_header = not _tokens_are_numeric(first_tokens)
    if has_header:
        header = [_normalize_column_name(token) for token in first_tokens]
        data_lines = lines[1:]
    else:
        header = []
        data_lines = lines

    columns = (
        _resolve_column(angle_column, header, angle_names, "angle", path),
        _resolve_column(
            intensity_column,
            header,
            intensity_names,
            intensity_label,
            path,
        ),
        _resolve_optional_column(sigma_column, header, {"sigma", "uncertainty", "error", "err"}, "sigma", path),
    )

    rows: list[tuple[float, float, float | None]] = []
    for line_number, raw_line in data_lines:
        parts = _split_line(raw_line.strip(), delimiter)
        try:
            angle = float(parts[columns[0]])
            value = float(parts[columns[1]])
            sigma = None if columns[2] is None else float(parts[columns[2]])
        except (IndexError, ValueError) as error:
            raise ValueError(f"malformed curve row in {path} line {line_number}") from error
        rows.append((angle, value, sigma))

    return _validated_curve_rows(rows, path=path)


def _iter_content_lines(path: Path, *, comments: str) -> Iterable[tuple[int, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or (comments and stripped.startswith(comments)):
                continue
            yield line_number, line


def _split_line(line: str, delimiter: str | None) -> list[str]:
    if delimiter is not None:
        return [part.strip() for part in line.split(delimiter) if part.strip()]
    if "," in line:
        return [part.strip() for part in line.split(",") if part.strip()]
    return line.split()


def _tokens_are_numeric(tokens: list[str]) -> bool:
    if not tokens:
        return False
    try:
        [float(token) for token in tokens]
    except ValueError:
        return False
    return True


def _normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", name.strip().lower())


def _resolve_column(
    requested: str | int,
    header: list[str],
    common_names: set[str],
    label: str,
    path: Path,
) -> int:
    if isinstance(requested, int):
        return requested
    normalized = _normalize_column_name(requested)
    if not header:
        raise ValueError(f"{label}_column={requested!r} requires a header in {path}")
    if normalized in header:
        return header.index(normalized)
    matches = [index for index, name in enumerate(header) if name in common_names]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"column {requested!r} was not found in {path}")


def _resolve_optional_column(
    requested: str | int | None,
    header: list[str],
    common_names: set[str],
    label: str,
    path: Path,
) -> int | None:
    if requested is None:
        return None
    return _resolve_column(requested, header, common_names, label, path)


def _validated_curve_rows(
    rows: list[tuple[float, float, float | None]],
    *,
    path: Path,
) -> _CurveTable:
    if not rows:
        raise ValueError(f"no curve rows found in {path}")
    has_sigma = rows[0][2] is not None
    if any((row[2] is None) != (not has_sigma) for row in rows):
        raise ValueError(f"{path} mixes missing and present sigma values")

    angles = np.asarray([row[0] for row in rows], dtype=float)
    values = np.asarray([row[1] for row in rows], dtype=float)
    sigma = None
    if has_sigma:
        sigma = np.asarray([row[2] for row in rows], dtype=float)

    if not np.all(np.isfinite(angles)):
        raise ValueError(f"{path} contains NaN or infinite angle values")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains NaN or infinite intensity values")
    if sigma is not None:
        if not np.all(np.isfinite(sigma)):
            raise ValueError(f"{path} contains NaN or infinite sigma values")
        if np.any(sigma < 0.0):
            raise ValueError(f"{path} contains negative sigma values")

    order = np.argsort(angles)
    sorted_angles = angles[order]
    if np.any(np.diff(sorted_angles) == 0.0):
        raise ValueError(f"duplicate angle values found in {path}")
    sorted_values = values[order]
    sorted_sigma = None if sigma is None else sigma[order]
    return _CurveTable(sorted_angles, sorted_values, sorted_sigma)


__all__ = ["read_reflectivity_data", "read_rocking_curve_data"]
