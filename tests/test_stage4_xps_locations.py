"""Stage 4 canonical XPS locations and compatibility coverage."""


def test_attenuation_implementation_and_compatibility_paths_share_objects():
    from swanx import _xps
    from swanx.xps import attenuation_factor
    from swanx.xps.attenuation import attenuation_factor as canonical
    from swxps.xps import attenuation_factor as legacy

    assert canonical.__module__ == "swanx.xps.attenuation"
    assert attenuation_factor is canonical
    assert _xps.attenuation_factor is canonical
    assert legacy is canonical


def test_intensity_implementation_and_compatibility_paths_share_objects():
    from swanx import _xps
    from swanx.xps import integrate_xps_intensity
    from swanx.xps.intensity import integrate_xps_intensity as canonical
    from swxps.xps import integrate_xps_intensity as legacy

    assert canonical.__module__ == "swanx.xps.intensity"
    assert integrate_xps_intensity is canonical
    assert _xps.integrate_xps_intensity is canonical
    assert legacy is canonical


def test_rocking_curve_implementation_and_compatibility_paths_share_objects():
    from swanx import _xps
    from swanx.xps import RockingCurve
    from swanx.xps.rocking_curve import RockingCurve as canonical
    from swxps.xps import RockingCurve as legacy

    assert canonical.__module__ == "swanx.xps.rocking_curve"
    assert RockingCurve is canonical
    assert _xps.RockingCurve is canonical
    assert legacy is canonical


def test_grid_xps_implementation_and_optics_compatibility_share_objects():
    from swanx.optics.unified_grid import (
        cell_centered_attenuation as optics_attenuation,
        integrate_xps_on_grid as optics_integrate,
    )
    from swanx.unified_grid import integrate_xps_on_grid as flat_swanx
    from swanx.xps import integrate_xps_on_grid
    from swanx.xps.grid import (
        cell_centered_attenuation as canonical_attenuation,
        integrate_xps_on_grid as canonical_integrate,
    )
    from swxps.unified_grid import integrate_xps_on_grid as legacy

    assert canonical_attenuation.__module__ == "swanx.xps.grid"
    assert canonical_integrate.__module__ == "swanx.xps.grid"
    assert optics_attenuation is canonical_attenuation
    assert optics_integrate is canonical_integrate
    assert integrate_xps_on_grid is canonical_integrate
    assert flat_swanx is canonical_integrate
    assert legacy is canonical_integrate


def test_xps_high_level_exports_are_lazy_existing_simulation_objects():
    from swanx.simulation import RockingCurveRequest as existing_request
    from swanx.simulation import simulate_rocking_curves as existing_simulate
    from swanx.xps import RockingCurveRequest, simulate_rocking_curves

    assert RockingCurveRequest is existing_request
    assert simulate_rocking_curves is existing_simulate
