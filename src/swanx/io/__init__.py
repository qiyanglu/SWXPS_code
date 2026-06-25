"""File IO, table interpolation, and preprocessing helpers."""

from .imfp import IMFPTable, read_imfp
from .materials import (
    MaterialTables,
    core_level_from_tables,
    core_levels_from_specs,
    load_material_tables,
    stack_from_layer_specs,
)
from .optical_constants import OpticalConstantTable, read_optical_constants

# Backward-compatible helpers inside the current swanx namespace.  These are
# not the preferred public workflow, but existing flat swanx scripts still use
# them while the package is approaching API freeze.
from ..imfp import imfp_from_file, imfp_path, load_imfp
from ..optical_constants import (
    OpticalConstantsTable,
    constants_from_file,
    layer_from_file,
    load_optical_constants,
    optical_constants_path,
)
from ..preprocessing import (
    BackgroundCorrection,
    normalize_by_background,
    normalize_by_mean,
    normalize_rocking_curve,
    subtract_edge_polynomial_background,
)

__all__ = [name for name in globals() if not name.startswith("_")]
