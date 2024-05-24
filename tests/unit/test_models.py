import numpy as np
import pybamm
import pytest

import pybop
from examples.standalone.model import ExponentialDecay


class TestModels:
    """
    A class to test the models.
    """

    @pytest.mark.parametrize(
        "model_class, expected_name",
        [
            (pybop.lithium_ion.SPM, "Single Particle Model"),
            (pybop.lithium_ion.SPMe, "Single Particle Model with Electrolyte"),
            (pybop.lithium_ion.DFN, "Doyle-Fuller-Newman"),
            (pybop.lithium_ion.MPM, "Many Particle Model"),
            (pybop.lithium_ion.MSMR, "Multi Species Multi Reactions Model"),
            (pybop.lithium_ion.WeppnerHuggins, "Weppner & Huggins model"),
            (pybop.empirical.Thevenin, "Equivalent Circuit Thevenin Model"),
        ],
    )
    @pytest.mark.unit
    def test_model_classes(self, model_class, expected_name):
        options = None
        if model_class is pybop.lithium_ion.MSMR:
            options = {"number of MSMR reactions": ("6", "4")}
        model = model_class(options=options)

        assert model.pybamm_model is not None
        assert model.name == expected_name

        # Test initialisation with kwargs
        parameter_set = pybop.ParameterSet(
            params_dict={"Nominal cell capacity [A.h]": 5}
        )
        model = model_class(options=options, build=True, parameter_set=parameter_set)

    @pytest.fixture(
        params=[
            pybop.lithium_ion.SPM(),
            pybop.lithium_ion.SPMe(),
            pybop.lithium_ion.DFN(),
            pybop.lithium_ion.MPM(),
            pybop.lithium_ion.MSMR(options={"number of MSMR reactions": ("6", "4")}),
            pybop.lithium_ion.WeppnerHuggins(),
            pybop.empirical.Thevenin(),
        ]
    )
    def model(self, request):
        model = request.param
        return model.copy()

    @pytest.mark.unit
    def test_simulate_without_build_model(self, model):
        with pytest.raises(
            ValueError, match="Model must be built before calling simulate"
        ):
            model.simulate(None, None)

        with pytest.raises(
            ValueError, match="Model must be built before calling simulate"
        ):
            model.simulateS1(None, None)

    @pytest.mark.unit
    def test_non_default_solver(self):
        solver = pybamm.CasadiSolver(
            mode="fast",
            atol=1e-6,
            rtol=1e-6,
        )
        model = pybop.lithium_ion.SPM(solver=solver)
        assert model.solver.mode == "fast"
        assert model.solver.atol == 1e-6
        assert model.solver.rtol == 1e-6

    @pytest.mark.unit
    def test_predict_without_pybamm(self, model):
        model._unprocessed_model = None

        with pytest.raises(ValueError):
            model.predict(None, None)

    @pytest.mark.unit
    def test_predict_with_inputs(self, model):
        # Define inputs
        t_eval = np.linspace(0, 10, 100)
        if isinstance(model, (pybop.lithium_ion.EChemBaseModel)):
            if model.pybamm_model.options["working electrode"] == "positive":
                inputs = {
                    "Positive electrode active material volume fraction": 0.63,
                }
            else:
                inputs = {
                    "Negative electrode active material volume fraction": 0.52,
                    "Positive electrode active material volume fraction": 0.63,
                }
        elif isinstance(model, (pybop.empirical.Thevenin)):
            inputs = {
                "R0 [Ohm]": 0.0002,
                "R1 [Ohm]": 0.0001,
            }
        else:
            raise ValueError("Inputs not defined for this type of model.")

        res = model.predict(t_eval=t_eval, inputs=inputs)
        assert len(res["Voltage [V]"].data) == 100

    @pytest.mark.unit
    def test_predict_without_allow_infeasible_solutions(self, model):
        if isinstance(model, (pybop.lithium_ion.SPM, pybop.lithium_ion.SPMe)):
            model.allow_infeasible_solutions = False
            t_eval = np.linspace(0, 10, 100)
            inputs = {
                "Negative electrode active material volume fraction": 0.9,
                "Positive electrode active material volume fraction": 0.9,
            }

            res = model.predict(t_eval=t_eval, inputs=inputs)
            assert np.isinf(res).any()

    @pytest.mark.unit
    def test_build(self, model):
        model.build()
        assert model.built_model is not None

        # Test that the model can be built again
        model.build()
        assert model.built_model is not None

    @pytest.mark.unit
    def test_rebuild(self, model):
        model.build()
        initial_built_model = model._built_model
        assert model._built_model is not None

        # Test that the model can be built again
        model.rebuild()
        rebuilt_model = model._built_model
        assert rebuilt_model is not None

        # Filter out special and private attributes
        attributes_to_compare = [
            "algebraic",
            "bcs",
            "boundary_conditions",
            "mass_matrix",
            "parameters",
            "submodels",
            "summary_variables",
            "rhs",
            "variables",
            "y_slices",
        ]

        # Loop through the filtered attributes and compare them
        for attribute in attributes_to_compare:
            assert getattr(rebuilt_model, attribute) == getattr(
                initial_built_model, attribute
            )

    @pytest.mark.unit
    def test_parameter_set_definition(self):
        # Test initilisation with different types of parameter set
        param_dict = {"Nominal cell capacity [A.h]": 5}
        model = pybop.BaseModel(parameter_set=None)
        assert model._parameter_set is None

        model = pybop.BaseModel(parameter_set=param_dict)
        parameter_set = pybamm.ParameterValues(param_dict)
        assert model._parameter_set == parameter_set

        model = pybop.BaseModel(parameter_set=parameter_set)
        assert model._parameter_set == parameter_set

        pybop_parameter_set = pybop.ParameterSet(params_dict=param_dict)
        model = pybop.BaseModel(parameter_set=pybop_parameter_set)
        assert model._parameter_set == parameter_set

    @pytest.mark.unit
    def test_rebuild_geometric_parameters(self):
        parameter_set = pybop.ParameterSet.pybamm("Chen2020")
        parameters = [
            pybop.Parameter(
                "Positive particle radius [m]",
                prior=pybop.Gaussian(4.8e-06, 0.05e-06),
                bounds=[4e-06, 6e-06],
                initial_value=4.8e-06,
            ),
            pybop.Parameter(
                "Negative electrode thickness [m]",
                prior=pybop.Gaussian(40e-06, 1e-06),
                bounds=[30e-06, 50e-06],
                initial_value=48e-06,
            ),
        ]

        model = pybop.lithium_ion.SPM(parameter_set=parameter_set)
        model.build(parameters=parameters)
        initial_built_model = model.copy()
        assert initial_built_model._built_model is not None

        # Run prediction
        t_eval = np.linspace(0, 100, 100)
        out_init = initial_built_model.predict(t_eval=t_eval)

        # Test that the model can be rebuilt with different geometric parameters
        parameters[0].update(5e-06)
        parameters[1].update(45e-06)
        model.rebuild(parameters=parameters)
        rebuilt_model = model
        assert rebuilt_model._built_model is not None

        # Test model geometry
        assert (
            rebuilt_model._mesh["negative electrode"].nodes[1]
            != initial_built_model._mesh["negative electrode"].nodes[1]
        )
        assert (
            rebuilt_model.geometry["negative electrode"]["x_n"]["max"]
            != initial_built_model.geometry["negative electrode"]["x_n"]["max"]
        )

        assert (
            rebuilt_model.geometry["positive particle"]["r_p"]["max"]
            != initial_built_model.geometry["positive particle"]["r_p"]["max"]
        )

        assert (
            rebuilt_model._mesh["positive particle"].nodes[1]
            != initial_built_model._mesh["positive particle"].nodes[1]
        )

        # Compare model results
        out_rebuild = rebuilt_model.predict(t_eval=t_eval)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                out_init["Terminal voltage [V]"].data,
                out_rebuild["Terminal voltage [V]"].data,
                atol=1e-5,
            )

    @pytest.mark.unit
    def test_reinit(self):
        k = 0.1
        y0 = 1
        model = ExponentialDecay(pybamm.ParameterValues({"k": k, "y0": y0}))
        model.build()
        state = model.reinit(inputs={})
        np.testing.assert_array_almost_equal(state.as_ndarray(), np.array([[y0]]))

        state = model.reinit(inputs=[])
        np.testing.assert_array_almost_equal(state.as_ndarray(), np.array([[y0]]))

        model = ExponentialDecay(pybamm.ParameterValues({"k": k, "y0": y0}))
        with pytest.raises(ValueError):
            model.reinit(inputs={})

    @pytest.mark.unit
    def test_simulate(self):
        k = 0.1
        y0 = 1
        model = ExponentialDecay(pybamm.ParameterValues({"k": k, "y0": y0}))
        model.build()
        model.signal = ["y_0"]
        inputs = {}
        t_eval = np.linspace(0, 10, 100)
        expected = y0 * np.exp(-k * t_eval)
        solved = model.simulate(inputs, t_eval)
        np.testing.assert_array_almost_equal(solved["y_0"], expected, decimal=5)

        with pytest.raises(ValueError):
            ExponentialDecay(n_states=-1)

    @pytest.mark.unit
    def test_basemodel(self):
        base = pybop.BaseModel()
        x = np.array([1, 2, 3])

        with pytest.raises(NotImplementedError):
            base.cell_mass()

        with pytest.raises(NotImplementedError):
            base.cell_volume()

        with pytest.raises(NotImplementedError):
            base.approximate_capacity(x)

    @pytest.mark.unit
    def test_check_params(self):
        base = pybop.BaseModel()
        assert base.check_params()
        assert base.check_params(inputs={"a": 1})
        assert base.check_params(inputs=[1])
        with pytest.raises(ValueError, match="Expecting inputs in the form of"):
            base.check_params(inputs=["unexpected_string"])

    @pytest.mark.unit
    def test_non_converged_solution(self):
        model = pybop.lithium_ion.DFN()
        parameters = [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.2, 0.01),
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.2, 0.01),
            ),
        ]
        dataset = pybop.Dataset(
            {
                "Time [s]": np.linspace(0, 100, 100),
                "Current function [A]": np.zeros(100),
                "Voltage [V]": np.zeros(100),
            }
        )

        problem = pybop.FittingProblem(model, parameters=parameters, dataset=dataset)
        res = problem.evaluate([-0.2, -0.2])
        _, res_grad = problem.evaluateS1([-0.2, -0.2])

        for key in problem.signal:
            assert np.isinf(res.get(key, [])).any()
        assert np.isinf(res_grad).any()
