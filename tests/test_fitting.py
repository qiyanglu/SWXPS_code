import numpy as np

import swanx.fitting.core as fitting
from swanx.fitting import (
    FitContribution,
    FitEvaluation,
    FitParameter,
    FittingProblem,
    JointObjective,
    LayerUpdate,
    ReflectivityData,
    RockingCurveData,
    evaluation_from_contributions,
    initial_vector,
    parameter_dict,
    reflectivity_log_mse,
    rocking_curve_mse,
    stack_with_updates,
    validate_finite_layer_roughness,
)
from swanx.stack import (
    SimulationStack,
    StackLayer,
)
from swanx.workflows.simulate import (
    CoreLevelRequest,
    CoreLevelResult,
    ReflectivityResult,
    RockingCurveResult,
)
from swanx.xps import RockingCurve


def test_parameter_dict_preserves_names_and_bounds():
    parameters = (
        FitParameter("lno_thickness", 10.0, 20.0, unit="Angstrom", initial=15.0),
        FitParameter("angle_offset", -0.2, 0.2, unit="deg"),
    )

    values = parameter_dict(parameters, [14.0, 0.05])

    assert values == {"lno_thickness": 14.0, "angle_offset": 0.05}
    assert initial_vector(parameters) == [15.0, 0.0]


def test_parameter_dict_rejects_out_of_bounds_values():
    parameters = (FitParameter("roughness", 0.0, 10.0),)

    with np.testing.assert_raises(ValueError):
        parameter_dict(parameters, [12.0])


def test_stack_with_updates_changes_only_bound_layer_attributes():
    stack = SimulationStack(
        (
            StackLayer("vacuum", 0.0),
            StackLayer("LNO", 15.0, delta=1.0e-5, beta=1.0e-7, roughness=1.0),
            StackLayer("STO", 20.0, delta=5.0e-6, beta=1.0e-7, roughness=1.5),
            StackLayer("STO", 0.0, delta=5.0e-6, beta=1.0e-7, roughness=2.0),
        )
    )
    updates = (
        LayerUpdate("period_thickness", (1, 2), "thickness"),
        LayerUpdate("interface_roughness", (1, 2), "roughness"),
    )

    updated = stack_with_updates(
        stack,
        {"period_thickness": 18.0, "interface_roughness": 3.0},
        updates,
    )

    assert [layer.thickness for layer in updated.layers] == [0.0, 18.0, 18.0, 0.0]
    assert [layer.roughness for layer in updated.layers] == [0.0, 3.0, 3.0, 2.0]
    assert updated.layers[3].thickness == stack.layers[3].thickness


def test_stack_with_updates_rejects_roughness_larger_than_finite_layer_thickness():
    stack = SimulationStack(
        (
            StackLayer("vacuum", 0.0),
            StackLayer("C", 5.0, roughness=1.0),
            StackLayer("substrate", 0.0, roughness=8.0),
        )
    )

    with np.testing.assert_raises(ValueError):
        stack_with_updates(
            stack,
            {"carbon_roughness": 6.0},
            (LayerUpdate("carbon_roughness", (1,), "roughness"),),
        )


def test_roughness_validator_allows_semi_infinite_substrate_roughness():
    stack = SimulationStack(
        (
            StackLayer("vacuum", 0.0),
            StackLayer("film", 10.0, roughness=2.0),
            StackLayer("substrate", 0.0, roughness=20.0),
        )
    )

    validate_finite_layer_roughness(stack)


def test_reflectivity_log_mse_uses_log_scale_and_floor():
    data = ReflectivityData(
        name="R",
        angles=np.array([1.0, 2.0]),
        reflectivity=np.array([1.0e-4, 1.0e-6]),
        log_floor=1.0e-12,
    )

    residual = reflectivity_log_mse(data, np.array([1.0e-5, 1.0e-7]))
    expected = np.mean(
        (
            np.log10(np.array([1.0e-4, 1.0e-6]) + 1.0e-12)
            - np.log10(np.array([1.0e-5, 1.0e-7]) + 1.0e-12)
        )
        ** 2
    )

    np.testing.assert_allclose(residual, expected)


def test_rocking_curve_mse_supports_uncertainties():
    data = RockingCurveData(
        name="La 4d",
        angles=np.array([1.0, 2.0]),
        intensity=np.array([1.0, 1.2]),
        sigma=np.array([0.1, 0.2]),
    )

    residual = rocking_curve_mse(data, np.array([0.9, 1.0]))

    np.testing.assert_allclose(residual, 1.0)


def test_evaluation_and_joint_objective_record_history():
    parameters = (FitParameter("x", -2.0, 2.0),)

    def evaluate(values):
        contribution = FitContribution("parabola", raw=values["x"] ** 2, weight=2.0)
        return evaluation_from_contributions(values, (contribution,))

    objective = JointObjective(parameters, evaluate)

    assert objective([1.5]) == 4.5
    assert objective([-0.5]) == 0.5
    assert len(objective.history.evaluations) == 2
    assert objective.history.best == FitEvaluation(
        parameters={"x": -0.5},
        objective=0.5,
        contributions=(FitContribution("parabola", raw=0.25, weight=2.0),),
    )


def test_fitting_problem_applies_unequal_dataset_weights(monkeypatch):
    angles = np.array([1.0, 2.0])
    stack = SimulationStack((StackLayer("vacuum", 0.0), StackLayer("film", 10.0)))

    def fake_reflectivity(request):
        return ReflectivityResult(
            angle=request.angles,
            calculation_angle=request.angles,
            reflectivity=np.array([1.0e-5, 1.0e-5]),
        )

    def fake_rocking_curves(request):
        curve = RockingCurve(
            angle=request.angles,
            intensity=np.array([0.8, 0.8]),
            raw_intensity=np.array([0.8, 0.8]),
            normalization=1.0,
        )
        core = CoreLevelResult(
            name="C 1s",
            binding_energy_ev=285.0,
            kinetic_energy_ev=715.0,
            curve=curve,
        )
        return RockingCurveResult(
            angle=request.angles,
            calculation_angle=request.angles,
            core_levels=(core,),
        )

    monkeypatch.setattr(fitting, "simulate_reflectivity", fake_reflectivity)
    monkeypatch.setattr(fitting, "simulate_rocking_curves", fake_rocking_curves)

    problem = FittingProblem(
        parameters=(FitParameter("thickness", 5.0, 20.0, initial=10.0),),
        stack_builder=lambda values: stack,
        photon_energy_ev=1000.0,
        reflectivity=ReflectivityData(
            "reflectivity",
            angles,
            np.array([1.0e-4, 1.0e-4]),
            weight=2.0,
        ),
        rocking_curves=(
            RockingCurveData(
                "C 1s",
                angles,
                np.array([1.0, 1.0]),
                weight=5.0,
            ),
        ),
        core_levels=(
            CoreLevelRequest(
                name="C 1s",
                binding_energy_ev=285.0,
                concentration_by_material={"film": 1.0},
                imfp_by_material={"vacuum": 10.0, "film": 10.0},
            ),
        ),
    )

    evaluation = problem.objective().evaluate([10.0])

    raw_by_name = {contribution.name: contribution.raw for contribution in evaluation.contributions}
    np.testing.assert_allclose(raw_by_name["reflectivity"], 1.0, rtol=1e-6)
    np.testing.assert_allclose(raw_by_name["C 1s"], 0.04)
    np.testing.assert_allclose(evaluation.objective, 2.0 * 1.0 + 5.0 * 0.04)
    assert evaluation.timings["reflectivity_simulation_seconds"] >= 0.0
    assert evaluation.timings["rocking_curve_simulation_seconds"] >= 0.0
    assert evaluation.timings["objective_total_seconds"] >= 0.0
