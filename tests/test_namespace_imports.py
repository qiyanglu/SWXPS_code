"""Public namespace and temporary compatibility-import coverage."""


def test_primary_and_compatibility_namespaces_share_public_objects():
    import swanx as sx
    import swxps

    from swanx.stack import LayerSlicingPolicy
    from swxps.slicing import LayerSlicingPolicy as OldLayerSlicingPolicy

    assert sx.LayerSlicingPolicy is LayerSlicingPolicy
    assert OldLayerSlicingPolicy is LayerSlicingPolicy
    assert swxps.LayerSlicingPolicy is LayerSlicingPolicy


def test_first_stage_subpackage_imports():
    from swanx.diagnostics import compute_parameter_diagnostics
    from swanx.optics import transfer_matrix_reflectivity_array
    from swanx.xps import RockingCurveRequest

    assert callable(transfer_matrix_reflectivity_array)
    assert callable(compute_parameter_diagnostics)
    assert RockingCurveRequest.__module__ == "swanx.workflows.simulate"


def test_old_submodules_alias_canonical_modules():
    import swanx.simulation
    import swanx.slicing
    import swxps.simulation
    import swxps.slicing
    from swxps.simulation import ReflectivityRequest

    assert swxps.simulation is swanx.simulation
    assert swxps.slicing is swanx.slicing
    assert ReflectivityRequest is swanx.simulation.ReflectivityRequest


def test_all_first_stage_namespaces_import():
    import swanx.diagnostics
    import swanx.fitting
    import swanx.io
    import swanx.optics
    import swanx.stack
    import swanx.workflows
    import swanx.xps
