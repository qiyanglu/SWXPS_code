"""Layer, stack, slicing, template, and profile objects."""

from importlib import import_module

from .model import SimulationStack, StackLayer, stack_from_layers
from .slicing import (
    FixedLayerGridPlan,
    LayerGrid,
    LayerSlicingPolicy,
    adaptive_layer_grid,
    fixed_layer_grid,
    fixed_layer_grid_plan,
)
from ..layers import Layer, refractive_index, vacuum

_LAZY_EXPORTS = {
    "StackProfiles": ("swanx.stack.profiles", "StackProfiles"),
    "plot_vertical_concentration_profiles": (
        "swanx.stack.profiles",
        "plot_vertical_concentration_profiles",
    ),
    "sample_concentration_profiles": (
        "swanx.stack.profiles",
        "sample_concentration_profiles",
    ),
    "sample_layer_concentration_profiles": (
        "swanx.stack.profiles",
        "sample_layer_concentration_profiles",
    ),
    "sample_stack_property": ("swanx.stack.profiles", "sample_stack_property"),
    "stack_depth_grid": ("swanx.stack.profiles", "stack_depth_grid"),
    "LayerTemplate": ("swanx.stack_builders", "LayerTemplate"),
    "StackTemplate": ("swanx.stack_builders", "StackTemplate"),
    "SuperlatticeTemplate": ("swanx.stack_builders", "SuperlatticeTemplate"),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "FixedLayerGridPlan", "Layer", "LayerGrid", "LayerSlicingPolicy",
    "LayerTemplate", "SimulationStack", "StackLayer", "StackProfiles",
    "StackTemplate", "SuperlatticeTemplate", "adaptive_layer_grid",
    "fixed_layer_grid", "fixed_layer_grid_plan",
    "plot_vertical_concentration_profiles", "refractive_index",
    "sample_concentration_profiles", "sample_layer_concentration_profiles",
    "sample_stack_property", "stack_depth_grid", "stack_from_layers", "vacuum",
]
