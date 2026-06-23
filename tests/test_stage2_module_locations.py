"""Stage 2 canonical locations and flat/legacy compatibility coverage."""


def test_slicing_implementation_and_compatibility_paths_share_objects():
    import swanx as sx
    from swanx.slicing import LayerSlicingPolicy as FlatSwanxPolicy
    from swanx.stack.slicing import LayerSlicingPolicy
    from swxps.slicing import LayerSlicingPolicy as LegacyPolicy

    assert LayerSlicingPolicy.__module__ == "swanx.stack.slicing"
    assert FlatSwanxPolicy is LayerSlicingPolicy
    assert LegacyPolicy is LayerSlicingPolicy
    assert sx.LayerSlicingPolicy is LayerSlicingPolicy


def test_profile_implementation_and_compatibility_paths_share_objects():
    import swanx as sx
    from swanx.profiles import StackProfiles as FlatSwanxProfiles
    from swanx.stack.profiles import StackProfiles
    from swxps.profiles import StackProfiles as LegacyProfiles

    assert StackProfiles.__module__ == "swanx.stack.profiles"
    assert FlatSwanxProfiles is StackProfiles
    assert LegacyProfiles is StackProfiles
    assert sx.StackProfiles is StackProfiles


def test_diagnostics_implementation_and_compatibility_paths_share_objects():
    import swanx as sx
    from swanx.diagnostics import compute_parameter_diagnostics
    from swanx.diagnostics.covariance import (
        compute_parameter_diagnostics as canonical_compute,
    )
    from swxps.diagnostics import compute_parameter_diagnostics as legacy_compute

    assert canonical_compute.__module__ == "swanx.diagnostics.covariance"
    assert compute_parameter_diagnostics is canonical_compute
    assert legacy_compute is canonical_compute
    assert sx.compute_parameter_diagnostics is canonical_compute


def test_diagnostics_plot_and_report_namespaces_export_existing_helpers():
    from swanx.diagnostics import plot_parameter_estimates, save_fit_history_csv
    from swanx.diagnostics.plots import (
        plot_parameter_estimates as canonical_plot,
    )
    from swanx.diagnostics.reports import (
        save_fit_history_csv as canonical_report,
    )

    assert canonical_plot.__module__ == "swanx.diagnostics.plots"
    assert plot_parameter_estimates is canonical_plot
    assert save_fit_history_csv is canonical_report
