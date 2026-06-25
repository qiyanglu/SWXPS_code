"""Public namespace and package import coverage."""

import pytest


def test_primary_namespace_and_removed_legacy_namespace():
    import swanx as sx

    assert sx.ReflectivityRequest is not None
    assert sx.simulate_reflectivity is not None
    assert not hasattr(sx, "LayerSlicingPolicy")

    with pytest.raises(ModuleNotFoundError):
        __import__("swxps")


def test_first_stage_subpackage_imports():
    from swanx.diagnostics import compute_parameter_diagnostics
    from swanx.optics import transfer_matrix_reflectivity_array
    from swanx.xps import RockingCurveRequest

    assert callable(transfer_matrix_reflectivity_array)
    assert callable(compute_parameter_diagnostics)
    assert RockingCurveRequest.__module__ == "swanx.workflows.simulate"


def test_flat_swanx_submodules_still_alias_canonical_modules():
    import swanx.simulation
    import swanx.slicing
    from swanx.simulation import ReflectivityRequest
    from swanx.stack import LayerSlicingPolicy
    from swanx.slicing import LayerSlicingPolicy as FlatLayerSlicingPolicy

    assert FlatLayerSlicingPolicy is LayerSlicingPolicy
    assert ReflectivityRequest is swanx.simulation.ReflectivityRequest


def test_io_namespace_exports_file_workflow_helpers():
    from swanx.io import (
        core_level_from_tables,
        load_material_tables,
        read_imfp,
        read_optical_constants,
        read_reflectivity_data,
        read_rocking_curve_data,
        stack_from_layer_specs,
    )

    assert callable(read_optical_constants)
    assert callable(read_imfp)
    assert callable(load_material_tables)
    assert callable(stack_from_layer_specs)
    assert callable(core_level_from_tables)
    assert callable(read_reflectivity_data)
    assert callable(read_rocking_curve_data)


def test_all_first_stage_namespaces_import():
    import swanx.diagnostics
    import swanx.fitting
    import swanx.io
    import swanx.optics
    import swanx.stack
    import swanx.workflows
    import swanx.xps
