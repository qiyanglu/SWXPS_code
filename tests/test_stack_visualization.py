import pytest

from swxps import SimulationStack, StackLayer, schematic_layers


def test_schematic_layers_collapses_middle_layers():
    stack = SimulationStack(
        (
            StackLayer("vacuum", 0.0),
            StackLayer("C", 10.0),
            StackLayer("LNO", 20.0),
            StackLayer("STO", 20.0),
            StackLayer("LNO", 20.0),
            StackLayer("STO", 20.0),
            StackLayer("LNO", 20.0),
            StackLayer("STO", 0.0),
        )
    )

    visible = schematic_layers(stack, top_layers=2, bottom_layers=2)

    assert [layer.material for layer in visible] == ["C", "LNO", "...", "LNO", "STO"]
    assert visible[2].is_gap
    assert visible[2].collapsed_count == 3
    assert visible[-1].thickness == 0.0


def test_schematic_layers_keeps_short_stack_uncollapsed():
    stack = SimulationStack(
        (
            StackLayer("vacuum", 0.0),
            StackLayer("film", 12.0),
            StackLayer("substrate", 0.0),
        )
    )

    visible = schematic_layers(stack, top_layers=4, bottom_layers=2)

    assert [layer.material for layer in visible] == ["film", "substrate"]
    assert all(not layer.is_gap for layer in visible)


def test_schematic_layers_rejects_negative_layer_counts():
    stack = SimulationStack((StackLayer("vacuum", 0.0), StackLayer("substrate", 0.0)))

    with pytest.raises(ValueError, match="top_layers"):
        schematic_layers(stack, top_layers=-1)
