"""Read, cache, and interpolate tabulated x-ray optical constants."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .layers import Layer


@dataclass(frozen=True)
class OpticalConstantsTable:
    """A tabulated set of energy, delta, and beta values."""

    material: str
    density: float | None
    energy_ev: np.ndarray
    delta: np.ndarray
    beta: np.ndarray

    def constants_at(self, energy_ev: float) -> tuple[float, float]:
        """Return linearly interpolated ``(delta, beta)`` at energy in eV."""

        if energy_ev < self.energy_ev[0] or energy_ev > self.energy_ev[-1]:
            raise ValueError(
                f"energy_ev={energy_ev} is outside the table range "
                f"{self.energy_ev[0]} to {self.energy_ev[-1]} eV"
            )

        delta = np.interp(energy_ev, self.energy_ev, self.delta)
        beta = np.interp(energy_ev, self.energy_ev, self.beta)
        return float(delta), float(beta)

    def layer_at(
        self,
        energy_ev: float,
        thickness: float,
        roughness: float = 0.0,
    ) -> Layer:
        """Return a Layer using constants interpolated at energy in eV."""

        delta, beta = self.constants_at(energy_ev)
        return Layer(
            thickness=thickness,
            delta=delta,
            beta=beta,
            roughness=roughness,
        )


def load_optical_constants(path: str | Path) -> OpticalConstantsTable:
    """Load and cache a Henke/LBNL optical-constants file.

    Expected file format:

    ``Material Density=value``
    ``Energy(eV), Delta, Beta``
    followed by whitespace-separated numeric rows.

    The cache key includes the resolved path, modification time, and file size,
    so replacing or rewriting a table automatically triggers a fresh parse.
    """

    resolved_path = Path(path).resolve()
    stat = resolved_path.stat()
    return _load_optical_constants_cached(
        str(resolved_path),
        stat.st_mtime_ns,
        stat.st_size,
    )


def clear_optical_constants_cache() -> None:
    """Clear all cached optical-constants tables in this process."""

    _load_optical_constants_cached.cache_clear()


@lru_cache(maxsize=32)
def _load_optical_constants_cached(
    resolved_path: str,
    modified_time_ns: int,
    file_size: int,
) -> OpticalConstantsTable:
    """Parse one file identified by its resolved path and file metadata."""

    del modified_time_ns, file_size
    path = Path(resolved_path)
    rows: list[tuple[float, float, float]] = []
    material = path.stem
    density: float | None = None

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            if line_number == 1:
                material, density = _parse_header(
                    stripped,
                    fallback_material=material,
                )
                continue
            if line_number == 2:
                continue

            parts = stripped.replace(",", " ").split()
            if len(parts) != 3:
                raise ValueError(
                    f"expected 3 columns in {path} line {line_number}"
                )
            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))

    if not rows:
        raise ValueError(f"no optical-constant rows found in {path}")

    data = np.asarray(rows, dtype=float)
    energy_ev = data[:, 0]
    if np.any(np.diff(energy_ev) <= 0):
        raise ValueError(f"energy values in {path} must be strictly increasing")

    return OpticalConstantsTable(
        material=material,
        density=density,
        energy_ev=energy_ev,
        delta=data[:, 1],
        beta=data[:, 2],
    )


def constants_from_file(
    path: str | Path,
    energy_ev: float,
) -> tuple[float, float]:
    """Return interpolated ``(delta, beta)`` from an optical-constants file."""

    return load_optical_constants(path).constants_at(energy_ev)


def layer_from_file(
    path: str | Path,
    energy_ev: float,
    thickness: float,
    roughness: float = 0.0,
) -> Layer:
    """Return a Layer using constants from an optical-constants file."""

    return load_optical_constants(path).layer_at(
        energy_ev,
        thickness=thickness,
        roughness=roughness,
    )


def optical_constants_path(
    material: str,
    directory: str | Path = "OPC",
) -> Path:
    """Return the default ``.dat`` path for a material in an OPC directory."""

    return Path(directory) / f"{material}.dat"


def _parse_header(
    line: str,
    fallback_material: str,
) -> tuple[str, float | None]:
    parts = line.split()
    material = parts[0] if parts else fallback_material
    density = None

    for part in parts[1:]:
        if part.startswith("Density="):
            density = float(part.split("=", 1)[1])
            break

    return material, density
