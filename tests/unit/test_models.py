import sys
from io import StringIO

import numpy as np
import pybamm
import pytest

import pybop
from examples.standalone.model import ExponentialDecay as StandaloneDecay
from pybop.models.lithium_ion.basic_SPMe import convert_physical_to_grouped_parameters


class TestModels:
    """
    A class to test the models.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture(
        params=[
            pybop.lithium_ion.SPM(),
            pybop.lithium_ion.SPMe(),
            pybop.lithium_ion.DFN(),
            pybop.lithium_ion.MPM(),
            pybop.lithium_ion.MSMR(options={"number of MSMR reactions": ("6", "4")}),
            pybop.lithium_ion.WeppnerHuggins(),
            pybop.lithium_ion.GroupedSPMe(),
            pybop.lithium_ion.GroupedSPMe(options={"surface form": "differential"}),
            pybop.empirical.Thevenin(),
        ]
    )
    def model(self, request):
        model = request.param
        return model.copy()

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
            match="Cannot use sensitivities for parameters which require a model rebuild",
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

    @pytest.mark.parametrize(
        "model_cls", [StandaloneDecay, pybop.ExponentialDecayModel]
    )
    def test_reinit(self, model_cls):
        k = 0.1
        y0 = 1
        model = model_cls(pybamm.ParameterValues({"k": k, "y0": y0}))

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

        model = model_cls(pybamm.ParameterValues({"k": k, "y0": y0}))
        with pytest.raises(
            ValueError, match="Model must be built before calling reinit"
        ):
            model.reinit(inputs={})

    def test_base_ecircuit_model(self):
        def check_params(inputs: dict, allow_infeasible_solutions: bool):
            return True if inputs is None else inputs["a"] < 2

        base_ecircuit_model = pybop.empirical.ECircuitModel(
            pybamm_model=pybamm.equivalent_circuit.Thevenin,
            check_params=check_params,
        )
        assert base_ecircuit_model.check_params({"a": 1})

        base_ecircuit_model = pybop.empirical.ECircuitModel(
            pybamm_model=pybamm.equivalent_circuit.Thevenin,
        )
        assert base_ecircuit_model.check_params()

    def test_userdefined_check_params(self):
        def check_params(inputs: dict, allow_infeasible_solutions: bool):
            return True if inputs is None else inputs["a"] < 2

        for model in [
            pybop.BaseModel(check_params=check_params),
            pybop.empirical.Thevenin(check_params=check_params),
        ]:
            assert model.check_params(inputs={"a": 1})
            assert not model.check_params(inputs={"a": 2})
            with pytest.raises(
                TypeError, match="Inputs must be a dictionary or numeric."
            ):
                model.check_params(inputs=["unexpected_string"])

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

    def test_grouped_SPMe(self):
        parameter_set = pybop.ParameterSet.pybamm("Chen2020")
        parameter_set["Electrolyte diffusivity [m2.s-1]"] = 1.769e-10
        parameter_set["Electrolyte conductivity [S.m-1]"] = 0.9487

        grouped_parameter_set = convert_physical_to_grouped_parameters(parameter_set)
        model = pybop.lithium_ion.GroupedSPMe(parameter_set=grouped_parameter_set)

        with pytest.raises(
            ValueError, match="GroupedSPMe can currently only accept an initial SoC."
        ):
            model.set_initial_state({"Initial open-circuit voltage [V]": 3.7})

        model.set_initial_state({"Initial SoC": 1.0})
        res = model.predict(t_eval=np.linspace(0, 10, 100))
        assert len(res["Voltage [V]"].data) == 100

        variable_list = model.pybamm_model.default_quick_plot_variables
        assert isinstance(variable_list, list)
