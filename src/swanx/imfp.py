"""Read, cache, and interpolate tabulated electron mean free paths."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class IMFPTable:
    """A tabulated set of kinetic energy and IMFP values."""

    material: str
    kinetic_energy_ev: np.ndarray
    imfp: np.ndarray

    def imfp_at(self, kinetic_energy_ev: float) -> float:
        """Return linearly interpolated IMFP in Angstrom."""

        if (
            kinetic_energy_ev < self.kinetic_energy_ev[0]
            or kinetic_energy_ev > self.kinetic_energy_ev[-1]
        ):
            raise ValueError(
                f"kinetic_energy_ev={kinetic_energy_ev} is outside the table range "
                f"{self.kinetic_energy_ev[0]} to {self.kinetic_energy_ev[-1]} eV"
            )

        return float(
            np.interp(
                kinetic_energy_ev,
                self.kinetic_energy_ev,
                self.imfp,
            )
        )


def load_imfp(path: str | Path) -> IMFPTable:
    """Load and cache an IMFP file with headers and two numeric columns.

    The cache key includes the resolved path, modification time, and file size,
    so replacing or rewriting a table automatically triggers a fresh parse.
    """

    resolved_path = Path(path).resolve()
    stat = resolved_path.stat()
    return _load_imfp_cached(
        str(resolved_path),
        stat.st_mtime_ns,
        stat.st_size,
    )


def clear_imfp_cache() -> None:
    """Clear all cached IMFP tables in this process."""

    _load_imfp_cached.cache_clear()


@lru_cache(maxsize=32)
def _load_imfp_cached(
    resolved_path: str,
    modified_time_ns: int,
    file_size: int,
) -> IMFPTable:
    """Parse one file identified by its resolved path and file metadata."""

    del modified_time_ns, file_size
    path = Path(resolved_path)
    material = path.stem
    rows: list[tuple[float, float]] = []
    expect_material = False

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("COMPOUND"):
                expect_material = True
                continue
            if expect_material:
                material = stripped
                expect_material = False
                continue

            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass

    if not rows:
        raise ValueError(f"no IMFP rows found in {path}")

    data = np.asarray(rows, dtype=float)
    kinetic_energy_ev = data[:, 0]
    if np.any(np.diff(kinetic_energy_ev) <= 0):
        raise ValueError(
            f"kinetic energy values in {path} must be strictly increasing"
        )

    return IMFPTable(
        material=material,
        kinetic_energy_ev=kinetic_energy_ev,
        imfp=data[:, 1],
    )


def imfp_from_file(path: str | Path, kinetic_energy_ev: float) -> float:
    """Return interpolated IMFP in Angstrom from a tabulated file."""

    return load_imfp(path).imfp_at(kinetic_energy_ev)


def imfp_path(material: str, directory: str | Path = "IMFP") -> Path:
    """Return the default IMFP path for a material in an IMFP directory."""

    return Path(directory) / f"{material}.ANG"
