"""Stage 3 canonical optics locations and compatibility coverage."""


def test_parratt_implementation_and_compatibility_paths_share_objects():
    from swanx.optics import parratt_reflectivity
    from swanx.optics.parratt import parratt_reflectivity as canonical
    from swanx.reflectivity import parratt_reflectivity as flat_swanx
    from swanx.reflectivity import parratt_reflectivity as legacy

    assert canonical.__module__ == "swanx.optics.parratt"
    assert parratt_reflectivity is canonical
    assert flat_swanx is canonical
    assert legacy is canonical


def test_fields_implementation_and_compatibility_paths_share_objects():
    from swanx.fields import transfer_matrix_reflectivity_array as flat_swanx
    from swanx.optics import transfer_matrix_reflectivity_array
    from swanx.optics.fields import (
        transfer_matrix_reflectivity_array as canonical,
    )
    from swanx.fields import transfer_matrix_reflectivity_array as legacy

    assert canonical.__module__ == "swanx.optics.fields"
    assert transfer_matrix_reflectivity_array is canonical
    assert flat_swanx is canonical
    assert legacy is canonical


def test_requested_field_exports_are_available_from_optics():
    from swanx.optics import (
        FieldProfile,
        transfer_matrix_electric_field_profile,
        transfer_matrix_electric_field_profiles,
    )

    assert FieldProfile.__module__ == "swanx.optics.fields"
    assert transfer_matrix_electric_field_profile.__module__ == "swanx.optics.fields"
    assert transfer_matrix_electric_field_profiles.__module__ == "swanx.optics.fields"


def test_unified_grid_implementation_and_compatibility_paths_share_objects():
    from swanx.optics.unified_grid import integrate_xps_on_grid as optics_compat
    from swanx.unified_grid import integrate_xps_on_grid as flat_swanx
    from swanx.xps.grid import integrate_xps_on_grid as canonical
    from swanx.unified_grid import integrate_xps_on_grid as legacy

    assert canonical.__module__ == "swanx.xps.grid"
    assert optics_compat is canonical
    assert flat_swanx is canonical
    assert legacy is canonical


def test_unified_simulation_entry_points_are_existing_objects():
    from swanx.optics.unified_grid import (
        simulate_reflectivity_unified,
        simulate_rocking_curves_unified,
    )
    from swanx.simulation_unified import (
        simulate_reflectivity_unified as existing_reflectivity,
        simulate_rocking_curves_unified as existing_rocking_curves,
    )
    from swanx.unified_grid import simulate_reflectivity_unified as flat_swanx
    from swanx.unified_grid import simulate_reflectivity_unified as legacy

    assert simulate_reflectivity_unified is existing_reflectivity
    assert simulate_rocking_curves_unified is existing_rocking_curves
    assert flat_swanx is existing_reflectivity
    assert legacy is existing_reflectivity
