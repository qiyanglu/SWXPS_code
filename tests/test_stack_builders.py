import numpy as np

from swxps import (
    LayerTemplate,
    StackTemplate,
    SuperlatticeTemplate,
)


def test_stack_template_builds_superlattice_from_parameter_names():
    template = StackTemplate(
        energy_ev=1000.0,
        parts=(
            LayerTemplate.vacuum(),
            LayerTemplate.from_constants(
                "C",
                thickness="carbon_thickness",
                roughness="carbon_roughness",
                delta=1.0e-6,
                beta=1.0e-7,
            ),
            SuperlatticeTemplate(
                repeats=3,
                period=(
                    LayerTemplate.from_constants(
                        "LNO",
                        thickness="lno_thickness",
                        roughness="superlattice_roughness",
                        delta=5.0e-6,
                        beta=2.0e-7,
                    ),
                    LayerTemplate.from_constants(
                        "STO",
                        thickness="sto_thickness",
                        roughness="superlattice_roughness",
                        delta=3.0e-6,
                        beta=1.0e-7,
                    ),
                ),
            ),
            LayerTemplate.from_constants(
                "STO",
                thickness=0.0,
                roughness="substrate_roughness",
                delta=3.0e-6,
                beta=1.0e-7,
            ),
        ),
    )

    stack = template.build(
        {
            "carbon_thickness": 10.0,
            "carbon_roughness": 2.0,
            "lno_thickness": 20.0,
            "sto_thickness": 22.0,
            "superlattice_roughness": 3.0,
            "substrate_roughness": 4.0,
        }
    )

    assert len(stack.layers) == 1 + 1 + 2 * 3 + 1
    assert [layer.material for layer in stack.layers] == [
        "vacuum",
        "C",
        "LNO",
        "STO",
        "LNO",
        "STO",
        "LNO",
        "STO",
        "STO",
    ]
    assert stack.layers[1].thickness == 10.0
    assert stack.layers[2].thickness == 20.0
    assert stack.layers[3].thickness == 22.0
    assert stack.layers[-1].thickness == 0.0
    assert stack.layers[-1].roughness == 4.0


def test_stack_template_builder_returns_fitting_callable():
    template = StackTemplate(
        energy_ev=1000.0,
        parts=(
            LayerTemplate.vacuum(),
            LayerTemplate.from_constants("film", 5.0, 1.0e-6, 1.0e-7),
            LayerTemplate.from_constants("substrate", 0.0, 2.0e-6, 1.0e-7),
        ),
    )

    stack = template.builder()({})

    assert stack.layers[1].material == "film"
    assert stack.layers[1].thickness == 5.0
    assert stack.layers[2].thickness == 0.0


def test_layer_template_rejects_missing_parameter():
    template = StackTemplate(
        energy_ev=1000.0,
        parts=(
            LayerTemplate.vacuum(),
            LayerTemplate.from_constants(
                "film",
                thickness="film_thickness",
                delta=1.0e-6,
                beta=1.0e-7,
            ),
        ),
    )

    with np.testing.assert_raises(ValueError):
        template.build({})


def test_stack_template_rejects_nonzero_substrate_thickness():
    template = StackTemplate(
        energy_ev=1000.0,
        parts=(
            LayerTemplate.vacuum(),
            LayerTemplate.from_constants("substrate", 5.0, 1.0e-6, 1.0e-7),
        ),
    )

    with np.testing.assert_raises(ValueError):
        template.build()


def test_stack_template_rejects_missing_vacuum_layer():
    template = StackTemplate(
        energy_ev=1000.0,
        parts=(
            LayerTemplate.from_constants("film", 5.0, 1.0e-6, 1.0e-7),
            LayerTemplate.from_constants("substrate", 0.0, 1.0e-6, 1.0e-7),
        ),
    )

    with np.testing.assert_raises(ValueError):
        template.build()
