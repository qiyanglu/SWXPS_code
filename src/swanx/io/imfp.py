"""Electron inelastic mean-free-path readers and interpolation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class IMFPTable:
    """Tabulated electron IMFP values in Angstrom."""

    kinetic_energy_ev: np.ndarray
    imfp_angstrom: np.ndarray
    material: str | None = None
    source: str | None = None

    def at_kinetic_energy(self, kinetic_energy_ev: float) -> float:
        """Return linearly interpolated IMFP at photoelectron kinetic energy."""

        energy = float(kinetic_energy_ev)
        if energy < self.kinetic_energy_ev[0] or energy > self.kinetic_energy_ev[-1]:
            raise ValueError(
                f"kinetic_energy_ev={energy} is outside the IMFP table range "
                f"{self.kinetic_energy_ev[0]} to {self.kinetic_energy_ev[-1]} eV"
            )
        return float(np.interp(energy, self.kinetic_energy_ev, self.imfp_angstrom))


def read_imfp(
    path: str | Path,
    *,
    delimiter: str | None = None,
    comments: str = "#",
    energy_column: str | int | None = None,
    imfp_column: str | int | None = None,
) -> IMFPTable:
    """Read an IMFP table from CSV, whitespace text, or TPP-style ``.ANG``."""

    path = Path(path)
    lines = list(_iter_content_lines(path, comments=comments))
    if not lines:
        raise ValueError(f"no IMFP rows found in {path}")

    material = _infer_ang_material(lines, fallback=path.stem)
    first_numeric_index = _first_numeric_line_index(lines, delimiter)
    first_tokens = _split_line(lines[0][1].strip(), delimiter)
    has_header = (
        first_numeric_index != 0
        and first_tokens
        and not _tokens_are_numeric(first_tokens)
        and _looks_like_table_header(first_tokens)
    )

    if has_header:
        header = [_normalize_column_name(token) for token in first_tokens]
        data_lines = _data_lines_from_first_numeric(lines[1:], delimiter, path)
    elif energy_column is not None or imfp_column is not None:
        header = []
        data_lines = _data_lines_from_first_numeric(lines, delimiter, path)
    else:
        header, data_lines = _detect_ang_header(lines, delimiter)

    energy_index = _resolve_column(
        energy_column,
        header,
        {"energy", "energyev", "energy_ev", "kineticenergy", "kinetic_energy", "kineticenergyev", "kinetic_energy_ev"},
        "energy",
        path,
    )
    imfp_index = _resolve_column(
        imfp_column,
        header,
        {"imfp", "imfpangstrom", "imfp_angstrom", "lambda", "lambdaangstrom", "lambda_angstrom"},
        "imfp",
        path,
    )

    rows: list[tuple[float, float]] = []
    for line_number, raw_line in data_lines:
        parts = _split_line(raw_line.strip(), delimiter)
        try:
            rows.append((float(parts[energy_index]), float(parts[imfp_index])))
        except (IndexError, ValueError) as error:
            raise ValueError(f"invalid IMFP row in {path} line {line_number}") from error

    kinetic_energy_ev, imfp_angstrom = _validated_imfp(rows, path=path)
    return IMFPTable(
        kinetic_energy_ev=kinetic_energy_ev,
        imfp_angstrom=imfp_angstrom,
        material=material,
        source=str(path),
    )


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


def _looks_like_table_header(tokens: list[str]) -> bool:
    names = {_normalize_column_name(token) for token in tokens}
    energy_names = {"energy", "energyev", "energy_ev", "kineticenergy", "kinetic_energy", "kineticenergyev", "kinetic_energy_ev"}
    imfp_names = {"imfp", "imfpangstrom", "imfp_angstrom", "lambda", "lambdaangstrom", "lambda_angstrom"}
    return bool(names & energy_names) and bool(names & imfp_names)


def _first_numeric_line_index(lines: list[tuple[int, str]], delimiter: str | None) -> int | None:
    for index, (_, raw_line) in enumerate(lines):
        parts = _split_line(raw_line.strip(), delimiter)
        if len(parts) >= 2 and _tokens_are_numeric(parts[:2]):
            return index
    return None


def _data_lines_from_first_numeric(
    lines: list[tuple[int, str]],
    delimiter: str | None,
    path: Path,
) -> list[tuple[int, str]]:
    first_numeric = _first_numeric_line_index(lines, delimiter)
    if first_numeric is None:
        raise ValueError(f"could not find numeric IMFP rows in {path}")
    return lines[first_numeric:]


def _detect_ang_header(
    lines: list[tuple[int, str]],
    delimiter: str | None,
) -> tuple[list[str], list[tuple[int, str]]]:
    for index, (_, raw_line) in enumerate(lines):
        tokens = [_normalize_column_name(token) for token in _split_line(raw_line.strip(), delimiter)]
        if "energy" in tokens and "imfp" in tokens:
            first_numeric = _first_numeric_line_index(lines[index + 1 :], delimiter)
            if first_numeric is None:
                raise ValueError("could not find numeric IMFP rows")
            return tokens, lines[index + 1 + first_numeric :]
    first_numeric = _first_numeric_line_index(lines, delimiter)
    if first_numeric is None:
        raise ValueError("could not find numeric IMFP rows")
    return [], lines[first_numeric:]


def _infer_ang_material(
    lines: list[tuple[int, str]],
    *,
    fallback: str,
) -> str | None:
    for index, (_, raw_line) in enumerate(lines[:-1]):
        if raw_line.strip().startswith("COMPOUND"):
            return lines[index + 1][1].strip()
    return fallback


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
    if header:
        matches = [index for index, name in enumerate(header) if name in common_names]
        if len(matches) == 1:
            return matches[0]
    if not header:
        return 0 if label == "energy" else 1
    raise ValueError(f"could not infer unique {label} column in {path}")


def _validated_imfp(
    rows: list[tuple[float, float]],
    *,
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError(f"no IMFP rows found in {path}")
    data = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path} contains NaN or infinite IMFP values")
    if np.any(data[:, 0] <= 0.0):
        raise ValueError(f"kinetic energy values in {path} must be positive")
    if np.any(data[:, 1] <= 0.0):
        raise ValueError(f"IMFP values in {path} must be positive")
    order = np.argsort(data[:, 0])
    sorted_data = data[order]
    if np.any(np.diff(sorted_data[:, 0]) == 0.0):
        raise ValueError(f"duplicate kinetic energy values found in {path}")
    return sorted_data[:, 0], sorted_data[:, 1]
