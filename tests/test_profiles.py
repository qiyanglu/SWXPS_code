import numpy as np

from swanx.stack import (
    SimulationStack,
    StackLayer,
    sample_concentration_profiles,
    sample_stack_property,
    stack_depth_grid,
)


def make_profile_stack() -> SimulationStack:
    return SimulationStack(
        (
            StackLayer("vacuum", thickness=0.0),
            StackLayer("LNO", thickness=20.0, roughness=2.0),
            StackLayer("STO", thickness=20.0, roughness=2.0),
            StackLayer("STO", thickness=0.0, roughness=2.0),
        )
    )


def test_stack_depth_grid_spans_finite_layers():
    depth = stack_depth_grid(make_profile_stack(), step=10.0)

    np.testing.assert_allclose(depth, [0.0, 10.0, 20.0, 30.0, 40.0])


def test_sample_stack_property_uses_material_values():
    depth, la = sample_stack_property(
        make_profile_stack(),
        {"LNO": 1.0, "STO": 0.0},
        step=10.0,
    )

    assert depth.shape == la.shape
    assert la[0] > la[-1]


def test_sample_concentration_profiles_returns_common_depth_grid():
    sampled = sample_concentration_profiles(
        make_profile_stack(),
        {
            "La": {"LNO": 1.0},
            "Ti": {"STO": 1.0},
        },
        step=5.0,
    )

    assert set(sampled.profiles) == {"La", "Ti"}
    assert sampled.profiles["La"].shape == sampled.depth.shape
    assert sampled.profiles["Ti"].shape == sampled.depth.shape
    assert sampled.profiles["La"][0] > sampled.profiles["Ti"][0]
    assert sampled.profiles["Ti"][-1] > sampled.profiles["La"][-1]
