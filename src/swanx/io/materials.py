"""Material table loading and request-building helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..stack.model import SimulationStack, StackLayer
from ..workflows.simulate import CoreLevelRequest
from .imfp import IMFPTable, read_imfp
from .optical_constants import OpticalConstantTable, read_optical_constants


@dataclass(frozen=True)
class MaterialTables:
    """Optical-constant and IMFP tables keyed by user material labels."""

    optical_constants: Mapping[str, OpticalConstantTable]
    imfp: Mapping[str, IMFPTable]


def load_material_tables(
    *,
    opc_files: Mapping[str, str | Path] | None = None,
    imfp_files: Mapping[str, str | Path] | None = None,
    opc_dir: str | Path | None = None,
    imfp_dir: str | Path | None = None,
    opc_suffix: str = ".dat",
    imfp_suffix: str = ".ANG",
    materials: Sequence[str] | None = None,
) -> MaterialTables:
    """Load optical-constant and IMFP tables from explicit files or folders."""

    opc_files = {} if opc_files is None else dict(opc_files)
    imfp_files = {} if imfp_files is None else dict(imfp_files)
    labels = _material_labels(opc_files, imfp_files, materials)

    optical_tables: dict[str, OpticalConstantTable] = {}
    imfp_tables: dict[str, IMFPTable] = {}
    for label in labels:
        opc_path = _resolve_material_path(
            label,
            explicit_files=opc_files,
            directory=opc_dir,
            suffix=opc_suffix,
            kind="OPC",
        )
        if opc_path is not None:
            optical_tables[label] = read_optical_constants(opc_path)

        imfp_path = _resolve_material_path(
            label,
            explicit_files=imfp_files,
            directory=imfp_dir,
            suffix=imfp_suffix,
            kind="IMFP",
        )
        if imfp_path is not None:
            imfp_tables[label] = read_imfp(imfp_path)

    return MaterialTables(optical_constants=optical_tables, imfp=imfp_tables)


def stack_from_layer_specs(
    specs: Sequence[Mapping[str, Any]],
    *,
    optical_constants: Mapping[str, OpticalConstantTable],
    energy_ev: float,
) -> SimulationStack:
    """Build a :class:`SimulationStack` from layer specs and OPC tables."""

    if len(specs) < 2:
        raise ValueError("layer specs must include at least vacuum and substrate")

    layers: list[StackLayer] = []
    for spec in specs:
        material = str(spec["material"])
        thickness = float(spec.get("thickness", 0.0))
        roughness = float(spec.get("roughness", 0.0))
        if material.lower() == "vacuum":
            delta, beta = 0.0, 0.0
        elif "delta" in spec and "beta" in spec:
            delta, beta = float(spec["delta"]), float(spec["beta"])
        else:
            if material not in optical_constants:
                raise ValueError(f"missing optical constants for material {material!r}")
            delta, beta = optical_constants[material].at_energy(energy_ev)

        layers.append(
            StackLayer(
                material=material,
                thickness=thickness,
                delta=delta,
                beta=beta,
                roughness=roughness,
            )
        )

    _validate_layer_boundaries(layers)
    return SimulationStack(tuple(layers))


def core_level_from_tables(
    *,
    name: str,
    binding_energy_ev: float,
    photon_energy_ev: float,
    concentration_by_material: Mapping[str, float],
    imfp_tables: Mapping[str, IMFPTable],
    emission_angle_deg: float = 0.0,
    emitting_layer_indices: tuple[int, ...] | None = None,
    extra_imfp_by_material: Mapping[str, float] | None = None,
) -> CoreLevelRequest:
    """Build a core-level request with IMFPs interpolated at kinetic energy."""

    kinetic_energy_ev = float(photon_energy_ev) - float(binding_energy_ev)
    if kinetic_energy_ev <= 0.0:
        raise ValueError("photoelectron kinetic energy must be positive")

    resolved_imfp = {
        material: table.at_kinetic_energy(kinetic_energy_ev)
        for material, table in imfp_tables.items()
    }
    resolved_imfp.setdefault("vacuum", np.inf)
    if extra_imfp_by_material is not None:
        resolved_imfp.update(
            {material: float(value) for material, value in extra_imfp_by_material.items()}
        )

    return CoreLevelRequest(
        name=name,
        binding_energy_ev=float(binding_energy_ev),
        concentration_by_material={
            material: float(value) for material, value in concentration_by_material.items()
        },
        imfp_by_material=resolved_imfp,
        emission_angle_deg=float(emission_angle_deg),
        emitting_layer_indices=emitting_layer_indices,
    )


def core_levels_from_specs(
    specs: Sequence[Mapping[str, Any]],
    *,
    photon_energy_ev: float,
    imfp_tables: Mapping[str, IMFPTable],
) -> tuple[CoreLevelRequest, ...]:
    """Build multiple core-level requests from compact dictionaries."""

    return tuple(
        core_level_from_tables(
            name=str(spec["name"]),
            binding_energy_ev=float(spec["binding_energy_ev"]),
            photon_energy_ev=photon_energy_ev,
            concentration_by_material=spec["concentration_by_material"],
            imfp_tables=imfp_tables,
            emission_angle_deg=float(spec.get("emission_angle_deg", 0.0)),
            emitting_layer_indices=spec.get("emitting_layer_indices"),
            extra_imfp_by_material=spec.get("extra_imfp_by_material"),
        )
        for spec in specs
    )


def _material_labels(
    opc_files: Mapping[str, str | Path],
    imfp_files: Mapping[str, str | Path],
    materials: Sequence[str] | None,
) -> tuple[str, ...]:
    labels: list[str] = []
    for source in (materials or (), opc_files.keys(), imfp_files.keys()):
        for label in source:
            if label not in labels:
                labels.append(label)
    return tuple(labels)


def _resolve_material_path(
    label: str,
    *,
    explicit_files: Mapping[str, str | Path],
    directory: str | Path | None,
    suffix: str,
    kind: str,
) -> Path | None:
    if label in explicit_files:
        path = Path(explicit_files[label])
    elif directory is not None:
        path = Path(directory) / f"{label}{suffix}"
    else:
        return None
    if not path.exists():
        raise FileNotFoundError(
            f"missing {kind} file for material {label!r}: expected {path}"
        )
    return path


def _validate_layer_boundaries(layers: Sequence[StackLayer]) -> None:
    first = layers[0]
    if (
        first.material.lower() != "vacuum"
        or first.thickness != 0.0
        or first.delta != 0.0
        or first.beta != 0.0
    ):
        raise ValueError("first layer must be vacuum")
    if layers[-1].thickness != 0.0:
        raise ValueError("final substrate layer must have thickness=0.0")
