"""Parratt, transfer-matrix, field, and unified-grid optics."""

from ..constants import HC_EV_ANGSTROM
from ..fields import (
    FieldProfile,
    depth_grid,
    effective_layers_with_roughness,
    electric_field_profile,
    layer_field_amplitudes,
    parratt_reflection_amplitudes,
    transfer_matrix_electric_field_profile,
    transfer_matrix_electric_field_profiles,
    transfer_matrix_field_amplitudes,
    transfer_matrix_reflectivity,
    transfer_matrix_reflectivity_array,
    transfer_matrix_reflection_amplitude,
)
from ..reflectivity import (
    apply_roughness,
    energy_to_wavelength,
    fresnel_r_s,
    kz_in_layers,
    parratt_amplitude,
    parratt_reflectivity,
)
from ..unified_grid import (
    effective_layers_from_grid,
    field_profiles_on_grid,
    materialize_layer_grid,
    reflectivity_on_grid,
)

__all__ = [name for name in globals() if not name.startswith("_")]
