"""Narrow public IO API for external SWANX input files."""

from .curves import read_reflectivity_data, read_rocking_curve_data
from .imfp import IMFPTable, read_imfp
from .materials import (
    MaterialTables,
    core_level_from_tables,
    core_levels_from_specs,
    load_material_tables,
    stack_from_layer_specs,
)
from .optical_constants import OpticalConstantTable, read_optical_constants

__all__ = [
    "OpticalConstantTable",
    "read_optical_constants",
    "IMFPTable",
    "read_imfp",
    "MaterialTables",
    "load_material_tables",
    "stack_from_layer_specs",
    "core_level_from_tables",
    "core_levels_from_specs",
    "read_reflectivity_data",
    "read_rocking_curve_data",
]
