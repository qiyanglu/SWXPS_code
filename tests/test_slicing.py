import numpy as np
import pytest

from swxps import Layer
from swxps.slicing import (
    FixedLayerGridPlan,
    LayerSlicingPolicy,
    adaptive_layer_grid,
    fixed_layer_grid,
    fixed_layer_grid_plan,
)


def make_layers(*thicknesses: float) -> list[Layer]:
    return [Layer(0.0), *(Layer(value) for value in thicknesses), Layer(0.0)]


def test_default_policy_counts_thin_ordinary_and_thick_layers():
    policy = LayerSlicingPolicy()

    assert policy.slice_count(4.0) == 10
    assert policy.slice_count(16.0) == 10
    assert policy.slice_count(160.0) == 80


def test_user_can_change_maximum_slice_thickness():
    assert LayerSlicingPolicy(max_slice_thickness=0.5).slice_count(16.0) == 32
    assert LayerSlicingPolicy(max_slice_thickness=5.0).slice_count(160.0) == 32


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_slices": 0},
        {"min_slices": 2.5},
        {"max_slice_thickness": 0.0},
        {"max_slice_thickness": np.inf},
    ],
)
def test_policy_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        LayerSlicingPolicy(**kwargs)


def test_adaptive_grid_shares_one_cell_mapping_and_conserves_thickness():
    grid = adaptive_layer_grid(make_layers(4.0, 16.0, 160.0))

    assert grid.slice_counts == (10, 10, 80)
    assert len(grid.centers) == 100
    np.testing.assert_allclose(np.diff(grid.edges), grid.widths)
    np.testing.assert_allclose(np.sum(grid.widths), 180.0)
    np.testing.assert_allclose(
        [np.sum(grid.widths[grid.nominal_layer_index == index]) for index in (1, 2, 3)],
        [4.0, 16.0, 160.0],
    )
    np.testing.assert_array_equal(
        grid.effective_layer_index,
        np.arange(1, 101),
    )


def test_fixed_plan_keeps_shape_while_trial_thickness_changes():
    capacity = make_layers(8.0, 20.0)
    plan = fixed_layer_grid_plan(capacity)

    first = fixed_layer_grid(make_layers(4.0, 15.0), plan)
    second = fixed_layer_grid(make_layers(7.5, 19.5), plan)

    assert plan.slice_counts == (10, 10)
    assert first.centers.shape == second.centers.shape == (20,)
    assert not np.allclose(first.widths, second.widths)


def test_fixed_plan_rejects_capacity_and_topology_violations():
    plan = fixed_layer_grid_plan(make_layers(8.0, 20.0))

    with pytest.raises(ValueError, match="capacity"):
        fixed_layer_grid(make_layers(8.1, 20.0), plan)
    with pytest.raises(ValueError, match="topology"):
        fixed_layer_grid(make_layers(8.0), plan)


def test_fixed_plan_validates_lengths_and_counts():
    with pytest.raises(ValueError):
        FixedLayerGridPlan((10,), (4.0, 5.0))
    with pytest.raises(ValueError):
        FixedLayerGridPlan((0,), (4.0,))
