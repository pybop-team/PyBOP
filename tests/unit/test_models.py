import sys
from io import StringIO

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
        "model_class, expected_name, options",
        [
            (pybop.lithium_ion.SPM, "Single Particle Model", None),
            (pybop.lithium_ion.SPMe, "Single Particle Model with Electrolyte", None),
            (pybop.lithium_ion.DFN, "Doyle-Fuller-Newman", None),
            (pybop.lithium_ion.MPM, "Many Particle Model", None),
            (
                pybop.lithium_ion.MSMR,
                "Multi Species Multi Reactions Model",
                {"number of MSMR reactions": ("6", "4")},
            ),
            (pybop.lithium_ion.WeppnerHuggins, "Weppner & Huggins model", None),
            (pybop.empirical.Thevenin, "Equivalent Circuit Thevenin Model", None),
        ],
    )
    @pytest.mark.unit
    def test_model_classes(self, model_class, expected_name, options):
        model = model_class(options=options)
        assert model.pybamm_model is not None
        assert model.name == expected_name

        # Test initialisation with kwargs
        if model_class is pybop.lithium_ion.MSMR:
            # Reset the options to cope with a bug in PyBaMM v23.9 msmr.py:23 which is fixed in v24.1
            options = {"number of MSMR reactions": ("6", "4")}
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
        model.pybamm_model = None

        with pytest.raises(
            ValueError,
            match="The predict method currently only supports PyBaMM models.",
        ):
            model.predict(None, None)

        # Test new_copy() without pybamm_model
        if not isinstance(model, pybop.lithium_ion.MSMR):
            new_model = model.new_copy()
            assert new_model.pybamm_model is not None
            assert new_model.parameter_set is not None

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

        with pytest.raises(
            ValueError,
            match="The predict method requires either an experiment or t_eval to be specified.",
        ):
            model.predict(inputs=inputs)

    @pytest.mark.unit
    def test_predict_without_allow_infeasible_solutions(self, model):
        if isinstance(model, (pybop.lithium_ion.SPM, pybop.lithium_ion.SPMe)):
            model.allow_infeasible_solutions = False
            t_eval = np.linspace(0, 10, 100)
            inputs = {
                "Negative electrode active material volume fraction": 0.9,
                "Positive electrode active material volume fraction": 0.9,
            }

            with pytest.raises(
                ValueError, match="These parameter values are infeasible."
            ):
                model.predict(t_eval=t_eval, inputs=inputs)

    @pytest.mark.unit
    def test_build(self, model):
        if isinstance(model, pybop.lithium_ion.SPMe):
            model.build(initial_state={"Initial SoC": 1.0})

            # Test attributes with init_soc
            assert model.built_model is not None
            assert model.disc is not None
            assert model.built_initial_soc is not None
        else:
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

        model.set_parameters()
        assert model.model_with_set_params is not None

        # Test that the model can be built again
        model.build()
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
        assert model.parameter_set is None

        model = pybop.BaseModel(parameter_set=param_dict)
        parameter_set = pybamm.ParameterValues(param_dict)
        assert model.parameter_set == parameter_set

        model = pybop.BaseModel(parameter_set=parameter_set)
        assert model.parameter_set == parameter_set

        pybop_parameter_set = pybop.ParameterSet(params_dict=param_dict)
        model = pybop.BaseModel(parameter_set=pybop_parameter_set)
        assert model.parameter_set == parameter_set

    @pytest.mark.unit
    def test_rebuild_geometric_parameters(self):
        parameter_set = pybop.ParameterSet.pybamm("Chen2020")
        parameters = pybop.Parameters(
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
        )

        model = pybop.lithium_ion.SPM(parameter_set=parameter_set)
        model.build(parameters=parameters)
        initial_built_model = model.copy()
        assert initial_built_model._built_model is not None

        # Run prediction
        t_eval = np.linspace(0, 100, 100)
        out_init = initial_built_model.predict(t_eval=t_eval)

        with pytest.raises(
            ValueError,
            match="Cannot use sensitivies for parameters which require a model rebuild",
        ):
            model.simulateS1(t_eval=t_eval, inputs=parameters.as_dict())

        # Test that the model can be rebuilt with different geometric parameters
        parameters["Positive particle radius [m]"].update(5e-06)
        parameters["Negative electrode thickness [m]"].update(45e-06)
        model.build(parameters=parameters)
        rebuilt_model = model
        assert rebuilt_model._built_model is not None

        # Test model geometry
        assert (
            rebuilt_model.mesh["negative electrode"].nodes[1]
            != initial_built_model.mesh["negative electrode"].nodes[1]
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
            rebuilt_model.mesh["positive particle"].nodes[1]
            != initial_built_model.mesh["positive particle"].nodes[1]
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

        with pytest.raises(
            ValueError, match="Model must be built before calling get_state"
        ):
            model.get_state({"k": k, "y0": y0}, 0, np.array([0]))

        model.build()
        state = model.reinit(inputs={})
        np.testing.assert_array_almost_equal(state.as_ndarray(), np.array([[y0]]))

        model.classify_parameters(pybop.Parameters(pybop.Parameter("y0")))
        state = model.reinit(inputs=[1])
        np.testing.assert_array_almost_equal(state.as_ndarray(), np.array([[y0]]))

        model = ExponentialDecay(pybamm.ParameterValues({"k": k, "y0": y0}))
        with pytest.raises(
            ValueError, match="Model must be built before calling reinit"
        ):
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
        np.testing.assert_array_almost_equal(solved["y_0"].data, expected, decimal=5)

        with pytest.raises(ValueError):
            ExponentialDecay(n_states=-1)

    @pytest.mark.unit
    def test_simulateEIS(self):
        # Test EIS on SPM
        model = pybop.lithium_ion.SPM(eis=True)
        model.build()

        # Construct frequencies and solve
        f_eval = np.linspace(100, 1000, 5)
        sol = model.simulateEIS(inputs={}, f_eval=f_eval)
        assert np.isfinite(sol["Impedance"]).all()

        # Test infeasible parameter values
        model.allow_infeasible_solutions = False
        inputs = {
            "Negative electrode active material volume fraction": 0.9,
            "Positive electrode active material volume fraction": 0.9,
        }
        # Rebuild model
        model.build(inputs=inputs)

        with pytest.raises(ValueError, match="These parameter values are infeasible."):
            model.simulateEIS(f_eval=f_eval, inputs=inputs)

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

        base.classify_parameters(parameters=None)
        assert isinstance(base.parameters, pybop.Parameters)

    @pytest.mark.unit
    def test_thevenin_model(self):
        parameter_set = pybop.ParameterSet(
            json_path="examples/scripts/parameters/initial_ecm_parameters.json"
        )
        parameter_set.import_parameters()
        assert parameter_set["Open-circuit voltage [V]"] == "default"
        model = pybop.empirical.Thevenin(
            parameter_set=parameter_set, options={"number of rc elements": 2}
        )
        assert (
            parameter_set["Open-circuit voltage [V]"]
            == model.pybamm_model.default_parameter_values["Open-circuit voltage [V]"]
        )

        model.predict(initial_state={"Initial SoC": 0.5}, t_eval=np.arange(0, 10, 5))
        assert model.parameter_set["Initial SoC"] == 0.5

        model.set_initial_state({"Initial SoC": parameter_set["Initial SoC"] / 2})
        assert model.parameter_set["Initial SoC"] == parameter_set["Initial SoC"] / 2
        model.set_initial_state(
            {
                "Initial open-circuit voltage [V]": parameter_set[
                    "Lower voltage cut-off [V]"
                ]
            }
        )
        np.testing.assert_allclose(model.parameter_set["Initial SoC"], 0.0, atol=1e-2)
        model.set_initial_state(
            {
                "Initial open-circuit voltage [V]": parameter_set[
                    "Upper voltage cut-off [V]"
                ]
            }
        )
        np.testing.assert_allclose(model.parameter_set["Initial SoC"], 1.0, atol=1e-2)

        with pytest.raises(ValueError, match="outside the voltage limits"):
            model.set_initial_state({"Initial open-circuit voltage [V]": -1.0})
        with pytest.raises(ValueError, match="Initial SOC should be between 0 and 1"):
            model.set_initial_state({"Initial SoC": -1.0})
        with pytest.raises(
            ValueError,
            match="Initial value must be a float between 0 and 1, or a string ending in 'V'",
        ):
            model.set_initial_state({"Initial SoC": "invalid string"})

    @pytest.mark.unit
    def test_check_params(self):
        base = pybop.BaseModel()
        assert base.check_params()
        assert base.check_params(inputs={"a": 1})
        assert base.check_params(inputs=[1])
        with pytest.raises(TypeError, match="Inputs must be a dictionary or numeric."):
            base.check_params(inputs=["unexpected_string"])

    @pytest.mark.unit
    def test_non_converged_solution(self):
        model = pybop.lithium_ion.DFN()
        parameters = pybop.Parameters(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.2, 0.01),
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.2, 0.01),
            ),
        )
        dataset = pybop.Dataset(
            {
                "Time [s]": np.linspace(0, 100, 100),
                "Current function [A]": np.zeros(100),
                "Voltage [V]": np.zeros(100),
            }
        )
        problem = pybop.FittingProblem(model, parameters=parameters, dataset=dataset)

        # Simulate the DFN with active material values of 0.
        # The solution elements will not change as the solver will not converge.
        output = problem.evaluate([0, 0])
        output_S1, _ = problem.evaluateS1([0, 0])

        for key in problem.signal:
            assert np.allclose(output.get(key, [])[0], output.get(key, []))
            assert np.allclose(output_S1.get(key, [])[0], output_S1.get(key, []))

    @pytest.mark.unit
    def test_set_initial_state(self):
        t_eval = np.linspace(0, 10, 100)

        model = pybop.lithium_ion.SPM()
        model.build(initial_state={"Initial SoC": 0.7})
        values_1 = model.predict(t_eval=t_eval)

        model = pybop.lithium_ion.SPM()
        model.build(initial_state={"Initial SoC": 0.4})
        model.set_initial_state({"Initial SoC": 0.7})
        values_2 = model.predict(t_eval=t_eval)

        np.testing.assert_allclose(
            values_1["Voltage [V]"].data, values_2["Voltage [V]"].data, atol=1e-8
        )

        init_ocp_p = model.parameter_set["Positive electrode OCP [V]"](0.7)
        init_ocp_n = model.parameter_set["Negative electrode OCP [V]"](0.7)
        model.set_initial_state(
            {"Initial open-circuit voltage [V]": init_ocp_p - init_ocp_n}
        )
        values_3 = model.predict(t_eval=t_eval)

        np.testing.assert_allclose(
            values_1["Voltage [V]"].data, values_3["Voltage [V]"].data, atol=0.05
        )

        with pytest.raises(ValueError, match="Expecting only one initial state."):
            model.set_initial_state(
                {"Initial open-circuit voltage [V]": 3.7, "Initial SoC": 0.7}
            )
        with pytest.raises(ValueError, match="Unrecognised initial state"):
            model.set_initial_state({"Initial voltage [V]": 3.7})

    @pytest.mark.unit
    def test_get_parameter_info(self, model):
        if isinstance(model, pybop.empirical.Thevenin):
            # Test at least one model without a built pybamm model
            model = pybop.empirical.Thevenin(build=False)

        parameter_info = model.get_parameter_info()
        assert isinstance(parameter_info, dict)

        captured_output = StringIO()
        sys.stdout = captured_output

        model.get_parameter_info(print_info=True)
        sys.stdout = sys.__stdout__

        printed_messaage = captured_output.getvalue().strip()

        for key, value in parameter_info.items():
            assert key in printed_messaage
            assert value in printed_messaage

    @pytest.mark.unit
    def test_set_current_function(self):
        dataset_1 = pybop.Dataset(
            {
                "Time [s]": np.linspace(0, 10, 100),
                "Current function [A]": 3.0 * np.ones(100),
            }
        )
        dataset_2 = pybop.Dataset(
            {
                "Time [s]": np.linspace(0, 5, 100),
                "Current function [A]": 6.0 * np.ones(100),
            }
        )

        model = pybop.lithium_ion.SPM()
        model.set_current_function(dataset=dataset_1)
        values_1 = model.predict(t_eval=dataset_1["Time [s]"])

        np.testing.assert_allclose(
            values_1["Current [A]"].data,
            dataset_1["Current function [A]"].data,
            atol=1e-8,
        )

        model.set_current_function(dataset=dataset_2)
        values_2 = model.predict(t_eval=dataset_2["Time [s]"])

        np.testing.assert_allclose(
            values_2["Current [A]"].data,
            dataset_2["Current function [A]"].data,
            atol=1e-8,
        )

        values_3 = model.simulate(inputs={}, t_eval=dataset_2["Time [s]"])

        np.testing.assert_allclose(
            values_3["Current [A]"].data,
            dataset_2["Current function [A]"].data,
            atol=1e-8,
        )
