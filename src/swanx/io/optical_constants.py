"""Optical-constant table readers and interpolation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class OpticalConstantTable:
    """Tabulated x-ray optical constants.

    Values follow the SWANX refractive-index convention
    ``n = 1 - delta + i beta``.  Interpolation returns ``(delta, beta)``
    because :class:`swanx.StackLayer` accepts those fields in that order.
    """

    energy_ev: np.ndarray
    delta: np.ndarray
    beta: np.ndarray
    material: str | None = None
    density: float | None = None
    source: str | None = None

    def at_energy(self, energy_ev: float) -> tuple[float, float]:
        """Return linearly interpolated ``(delta, beta)`` at photon energy."""

        energy = float(energy_ev)
        if energy < self.energy_ev[0] or energy > self.energy_ev[-1]:
            raise ValueError(
                f"energy_ev={energy} is outside the optical-constant table "
                f"range {self.energy_ev[0]} to {self.energy_ev[-1]} eV"
            )
        delta = np.interp(energy, self.energy_ev, self.delta)
        beta = np.interp(energy, self.energy_ev, self.beta)
        return float(delta), float(beta)


def read_optical_constants(
    path: str | Path,
    *,
    format: str = "cxro",
    delimiter: str | None = None,
    comments: str = "#",
    energy_column: str | int | None = None,
    delta_column: str | int | None = None,
    beta_column: str | int | None = None,
) -> OpticalConstantTable:
    """Read an optical-constant table.

    The default ``format="cxro"`` expects CXRO-style files with the fixed
    column meaning ``Energy(eV), Delta, Beta``.  ``format="table"`` supports
    user tables when column names or integer column indices make the order
    explicit.
    """

    path = Path(path)
    normalized_format = format.lower()
    if normalized_format == "cxro":
        table = _read_cxro_optical_constants(
            path,
            delimiter=delimiter,
            comments=comments,
        )
    elif normalized_format == "table":
        table = _read_table_optical_constants(
            path,
            delimiter=delimiter,
            comments=comments,
            energy_column=energy_column,
            delta_column=delta_column,
            beta_column=beta_column,
        )
    else:
        raise ValueError("format must be 'cxro' or 'table'")

    return table


def _read_cxro_optical_constants(
    path: Path,
    *,
    delimiter: str | None,
    comments: str,
) -> OpticalConstantTable:
    metadata_lines: list[str] = []
    header_seen = False
    rows: list[tuple[float, float, float]] = []

    for line_number, raw_line in _iter_content_lines(path, comments=comments):
        stripped = raw_line.strip()
        if not header_seen:
            tokens = [_normalize_column_name(token) for token in _split_line(stripped, delimiter)]
            if len(tokens) >= 3 and tokens[0] in {"energy", "energyev", "energy_ev"} and tokens[1] == "delta" and tokens[2] == "beta":
                header_seen = True
                continue
            metadata_lines.append(stripped)
            continue

        parts = _split_line(stripped, delimiter)
        if len(parts) < 3:
            raise ValueError(f"expected at least 3 columns in {path} line {line_number}")
        try:
            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError as error:
            raise ValueError(f"non-numeric optical constants in {path} line {line_number}") from error

    if not header_seen:
        raise ValueError(f"could not find CXRO header 'Energy(eV), Delta, Beta' in {path}")

    material, density = _parse_material_density(metadata_lines, fallback=path.stem)
    energy_ev, delta, beta = _validated_columns(
        rows,
        x_name="energy",
        y_names=("delta", "beta"),
        path=path,
        positive_x=True,
        positive_y=False,
    )
    return OpticalConstantTable(
        energy_ev=energy_ev,
        delta=delta,
        beta=beta,
        material=material,
        density=density,
        source=str(path),
    )


def _read_table_optical_constants(
    path: Path,
    *,
    delimiter: str | None,
    comments: str,
    energy_column: str | int | None,
    delta_column: str | int | None,
    beta_column: str | int | None,
) -> OpticalConstantTable:
    lines = list(_iter_content_lines(path, comments=comments))
    if not lines:
        raise ValueError(f"no optical-constant rows found in {path}")

    first_tokens = _split_line(lines[0][1].strip(), delimiter)
    has_header = not _tokens_are_numeric(first_tokens)
    if has_header:
        header = [_normalize_column_name(token) for token in first_tokens]
        data_lines = lines[1:]
    else:
        header = []
        data_lines = lines

    columns = (
        _resolve_column(energy_column, header, _ENERGY_NAMES, "energy", path),
        _resolve_column(delta_column, header, {"delta"}, "delta", path),
        _resolve_column(beta_column, header, {"beta"}, "beta", path),
    )

    rows: list[tuple[float, float, float]] = []
    for line_number, raw_line in data_lines:
        parts = _split_line(raw_line.strip(), delimiter)
        try:
            rows.append(tuple(float(parts[index]) for index in columns))
        except (IndexError, ValueError) as error:
            raise ValueError(f"invalid optical-constant row in {path} line {line_number}") from error

    energy_ev, delta, beta = _validated_columns(
        rows,
        x_name="energy",
        y_names=("delta", "beta"),
        path=path,
        positive_x=True,
        positive_y=False,
    )
    return OpticalConstantTable(
        energy_ev=energy_ev,
        delta=delta,
        beta=beta,
        material=path.stem,
        source=str(path),
    )


_ENERGY_NAMES = {"energy", "energyev", "energy_ev", "photonenergy", "photon_energy_ev"}


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
    requested: str | int | None,
    header: list[str],
    common_names: set[str],
    label: str,
    path: Path,
) -> int:
    if isinstance(requested, int):
        return requested
    if isinstance(requested, str):
        normalized = _normalize_column_name(requested)
        if not header:
            raise ValueError(f"{label}_column={requested!r} requires a header in {path}")
        if normalized not in header:
            raise ValueError(f"column {requested!r} was not found in {path}")
        return header.index(normalized)
    if not header:
        raise ValueError(f"{label}_column must be provided for headerless table {path}")
    matches = [index for index, name in enumerate(header) if name in common_names]
    if len(matches) != 1:
        raise ValueError(f"could not infer unique {label} column in {path}")
    return matches[0]


def _parse_material_density(
    metadata_lines: list[str],
    *,
    fallback: str,
) -> tuple[str | None, float | None]:
    material: str | None = fallback
    density: float | None = None
    density_pattern = re.compile(r"Density\s*=\s*([-+0-9.eE]+)")
    for line in metadata_lines:
        if not line:
            continue
        material = line.split()[0]
        match = density_pattern.search(line)
        if match:
            density = float(match.group(1))
            break
    return material, density


def _validated_columns(
    rows: list[tuple[float, float, float]],
    *,
    x_name: str,
    y_names: tuple[str, str],
    path: Path,
    positive_x: bool,
    positive_y: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError(f"no optical-constant rows found in {path}")
    data = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path} contains NaN or infinite optical-constant values")
    x = data[:, 0]
    if positive_x and np.any(x <= 0.0):
        raise ValueError(f"{x_name} values in {path} must be positive")
    if positive_y and np.any(data[:, 1:] <= 0.0):
        raise ValueError(f"{y_names[0]} and {y_names[1]} values in {path} must be positive")
    order = np.argsort(x)
    sorted_data = data[order]
    if np.any(np.diff(sorted_data[:, 0]) == 0.0):
        raise ValueError(f"duplicate {x_name} values found in {path}")
    return sorted_data[:, 0], sorted_data[:, 1], sorted_data[:, 2]
