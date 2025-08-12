import numpy as np
import pybamm
import pytest
from pybamm import IDAKLUSolver

import pybop


def create_msmr_and_params():
    model = pybamm.lithium_ion.MSMR(options={"number of MSMR reactions": ("6", "4")})
    param_values = model.default_parameter_values
    param_values.update(
        {
            "Cell volume [m3]": 0.2,
            "Positive electrode density [kg.m-3]": 123,
            "Separator density [kg.m-3]": 123,
            "Negative electrode density [kg.m-3]": 123,
            "Positive current collector thickness [m]": 5e-6,
            "Positive current collector density [kg.m-3]": 123,
            "Negative current collector thickness [m]": 5e-6,
            "Negative current collector density [kg.m-3]": 123,
        },
        check_already_exists=False,
    )
    return model, param_values


class TestBuilder:
    """
    A class to test the problem class.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture(
        params=[
            (pybamm.lithium_ion.SPM(), pybamm.ParameterValues("Chen2020")),
            (
                pybamm.lithium_ion.MPM(),
                pybamm.lithium_ion.MPM().default_parameter_values,
            ),
            (
                pybamm.lithium_ion.SPMe(),
                pybamm.lithium_ion.SPMe().default_parameter_values,
            ),
            (pybamm.lithium_ion.DFN(), pybamm.ParameterValues("Marquis2019")),
            create_msmr_and_params(),
        ]
    )
    def model_and_params(self, request):
        return request.param

    @pytest.fixture
    def dataset(self, model_and_params):
        model, parameter_values = model_and_params
        sim = pybamm.Simulation(
            model,
            parameter_values=parameter_values,
            solver=IDAKLUSolver(atol=1e-6, rtol=1e-6),
        )
        sol = sim.solve(t_eval=np.linspace(0, 10, 20))
        _, mask = np.unique(sol.t, return_index=True)
        return pybop.Dataset(
            {
                "Time [s]": sol.t[mask],
                "Current function [A]": sol["Current [A]"].data[mask],
                "Voltage [V]": sol["Voltage [V]"].data[mask],
            }
        )

    @pytest.fixture
    def experiment(self):
        return pybamm.Experiment(
            [
                (
                    "Discharge at 0.5C for 3 minutes (30 second period)",
                    "Rest for 1 minutes (30 second period)",
                    "Charge at 0.5C for 1 minutes (30 second period)",
                ),
            ]
        )

    @pytest.fixture
    def eis_dataset(self):
        return pybop.Dataset(
            {
                "Frequency [Hz]": np.logspace(-4.5, 5, 30),
                "Current function [A]": np.ones(30) * 0.0,
                "Impedance": np.ones(30) * 0.0,
            }
        )

    def test_builder(self, model_and_params, dataset):
        model, parameter_values = model_and_params
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model,
            parameter_values=parameter_values,
        )
        builder.add_parameter(
            pybop.Parameter(
                "Negative electrode active material volume fraction", initial_value=0.5
            )
        )
        builder.add_parameter(
            pybop.Parameter(
                "Positive electrode active material volume fraction", initial_value=0.5
            )
        )
        builder.add_cost(
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]")
        )
        builder.add_cost(
            pybop.costs.pybamm.MeanAbsoluteError("Voltage [V]", "Voltage [V]")
        )
        problem = builder.build()

        assert problem is not None
        problem.set_params(np.array([0.5, 0.5]))
        value1 = problem.run()
        problem.set_params(np.array([0.65, 0.65]))
        value2 = problem.run()
        assert abs((value1 - value2) / value1) > 1e-5
        problem.set_params(np.array([0.5, 0.5]))
        value1s, grad1s = problem.run_with_sensitivities()
        assert grad1s.shape == (2,)
        problem.set_params(np.array([0.65, 0.65]))
        value2s, grad2s = problem.run_with_sensitivities()
        np.testing.assert_allclose(value1s, value1, atol=5e-5)
        np.testing.assert_allclose(value2s, value2, atol=5e-5)

        # Test building twice
        problem2 = builder.build()
        assert problem2 != problem

        # Test chaining
        builder = (
            pybop.builders.Pybamm()
            .set_dataset(dataset)
            .set_simulation(model, parameter_values=parameter_values)
            .add_parameter(
                pybop.Parameter("Positive particle radius [m]", initial_value=5e-6)
            )
            .add_parameter(
                pybop.Parameter("Negative particle radius [m]", initial_value=5e-6)
            )
            .add_cost(pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]"))
        )

        builder.build()

        # Test setting number of threads
        builder.set_n_threads(3)
        problem_single_core = builder.build()
        assert problem_single_core.pipeline.n_threads == 3

    def test_builder_likelihoods(self, model_and_params, dataset):
        model, parameter_values = model_and_params
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model,
            parameter_values=parameter_values,
        )
        builder.add_parameter(
            pybop.Parameter(
                "Negative electrode active material volume fraction", initial_value=0.5
            )
        )
        builder.add_parameter(
            pybop.Parameter(
                "Positive electrode active material volume fraction", initial_value=0.5
            )
        )
        builder.add_cost(
            pybop.costs.pybamm.NegativeGaussianLogLikelihood(
                "Voltage [V]", "Voltage [V]", 1e-2
            )
        )
        problem = builder.build()

        problem.set_params(np.array([0.5, 0.5]))
        value1 = problem.run()
        assert isinstance(value1, np.ndarray)
        problem.set_params(np.array([0.65, 0.65]))
        value2 = problem.run()
        assert abs((value1 - value2) / value1) > 1e-5

        problem.set_params(np.array([0.5, 0.5]))
        value1s, grad1s = problem.run_with_sensitivities()
        assert isinstance(value1, np.ndarray)
        assert grad1s.shape == (2,)
        problem.set_params(np.array([0.65, 0.65]))
        value2s, grad2s = problem.run_with_sensitivities()
        np.testing.assert_allclose(value1s, value1, rtol=5e-4)
        np.testing.assert_allclose(value2s, value2, rtol=5e-4)

        # Test with estimated sigma
        builder.add_cost(
            pybop.costs.pybamm.NegativeGaussianLogLikelihood(
                "Voltage [V]",
                "Voltage [V]",
            )
        )
        problem2 = builder.build()
        problem2.set_params(np.array([0.5, 0.5, 1e-2]))
        value3 = problem2.run()
        np.testing.assert_allclose(2 * value1, value3)

        # Different sigma
        problem2.set_params(np.array([0.5, 0.5, 1e-3]))
        value4 = problem2.run()
        assert np.not_equal(2 * value1, value4)

    def test_builder_posterior(self, model_and_params, dataset):
        model, parameter_values = model_and_params
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model,
            parameter_values=parameter_values,
        )
        builder.add_parameter(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                initial_value=0.5,
                prior=pybop.Gaussian(0.5, 0.1),
            )
        )
        builder.add_parameter(
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                initial_value=0.5,
                prior=pybop.Gaussian(0.5, 0.1),
            )
        )
        builder.add_cost(
            pybop.costs.pybamm.NegativeGaussianLogLikelihood(
                "Voltage [V]", "Voltage [V]", 1e-2
            )
        )
        problem = builder.build()

        problem.set_params(np.array([0.5, 0.5]))
        value1 = problem.run()
        problem.set_params(np.array([0.65, 0.65]))
        value2 = problem.run()
        assert abs((value1 - value2) / value1) > 1e-5

        problem.set_params(np.array([0.5, 0.5]))
        value1s, grad1s = problem.run_with_sensitivities()
        assert grad1s.shape == (2,)
        problem.set_params(np.array([0.65, 0.65]))
        value2s, grad2s = problem.run_with_sensitivities()
        np.testing.assert_allclose(value1s, value1, rtol=5e-4)
        np.testing.assert_allclose(value2s, value2, rtol=5e-4)

    def test_builder_with_rebuild_params(self, model_and_params, dataset):
        model, parameter_values = model_and_params
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model,
            parameter_values=parameter_values,
        )
        builder.add_parameter(
            pybop.Parameter("Negative electrode thickness [m]", initial_value=5e-5)
        )
        builder.add_parameter(
            pybop.Parameter("Positive particle radius [m]", initial_value=1e-5)
        )
        builder.add_cost(
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]")
        )
        builder.add_cost(
            pybop.costs.pybamm.MeanAbsoluteError("Voltage [V]", "Voltage [V]")
        )
        problem = builder.build()

        problem.set_params(np.array([5e-5, 0.5e-6]))
        value1 = problem.run()
        problem.set_params(np.array([3e-5, 1.5e-6]))
        value2 = problem.run()
        assert abs((value1 - value2) / value1) > 1e-5

    def test_builder_with_experiment(self, model_and_params, experiment, dataset):
        model, parameter_values = model_and_params
        parameter_values.update(
            {
                "Electrolyte density [kg.m-3]": pybamm.Parameter(
                    "Separator density [kg.m-3]"
                ),
                "Negative electrode active material density [kg.m-3]": pybamm.Parameter(
                    "Negative electrode density [kg.m-3]"
                ),
                "Negative electrode carbon-binder density [kg.m-3]": pybamm.Parameter(
                    "Negative electrode density [kg.m-3]"
                ),
                "Positive electrode active material density [kg.m-3]": pybamm.Parameter(
                    "Positive electrode density [kg.m-3]"
                ),
                "Positive electrode carbon-binder density [kg.m-3]": pybamm.Parameter(
                    "Positive electrode density [kg.m-3]"
                ),
            },
            check_already_exists=False,
        )
        builder = pybop.builders.Pybamm()
        builder.set_simulation(
            model,
            parameter_values=parameter_values,
            experiment=experiment,
        )
        builder.add_parameter(
            pybop.Parameter(
                "Negative electrode active material volume fraction", initial_value=0.6
            )
        )
        builder.add_parameter(
            pybop.Parameter(
                "Positive electrode active material volume fraction", initial_value=0.6
            )
        )
        builder.add_cost(pybop.costs.pybamm.GravimetricEnergyDensity())
        problem = builder.build()

        problem.set_params(np.array([0.6, 0.6]))
        value1 = problem.run()
        problem.set_params(np.array([0.7, 0.7]))
        value2 = problem.run()
        assert abs((value1 - value2) / value1) > 1e-5
        problem.set_params(np.array([0.6, 0.6]))

    def test_builder_with_experiment_rebuild_params(
        self, model_and_params, experiment, dataset
    ):
        model, parameter_values = model_and_params
        parameter_values.update(
            {
                "Electrolyte density [kg.m-3]": pybamm.Parameter(
                    "Separator density [kg.m-3]"
                ),
                "Negative electrode active material density [kg.m-3]": pybamm.Parameter(
                    "Negative electrode density [kg.m-3]"
                ),
                "Negative electrode carbon-binder density [kg.m-3]": pybamm.Parameter(
                    "Negative electrode density [kg.m-3]"
                ),
                "Positive electrode active material density [kg.m-3]": pybamm.Parameter(
                    "Positive electrode density [kg.m-3]"
                ),
                "Positive electrode carbon-binder density [kg.m-3]": pybamm.Parameter(
                    "Positive electrode density [kg.m-3]"
                ),
            },
            check_already_exists=False,
        )
        builder = pybop.builders.Pybamm()
        builder.set_simulation(
            model,
            parameter_values=parameter_values,
            experiment=experiment,
        )
        builder.add_parameter(
            pybop.Parameter("Negative particle radius [m]", initial_value=5e-6)
        )
        builder.add_parameter(
            pybop.Parameter("Positive particle radius [m]", initial_value=5e-6)
        )
        builder.add_cost(pybop.costs.pybamm.GravimetricEnergyDensity())
        problem = builder.build()

        problem.set_params(np.array([0.6, 0.6]))
        value1 = problem.run()
        problem.set_params(np.array([0.7, 0.7]))
        value2 = problem.run()
        assert abs((value1 - value2) / value1) > 1e-5
        problem.set_params(np.array([0.6, 0.6]))

    def test_builder_with_cost_hypers(self, model_and_params, dataset):
        model, parameter_values = model_and_params
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model,
            parameter_values=parameter_values,
        )
        builder.add_parameter(
            pybop.Parameter(
                "Negative electrode active material volume fraction", initial_value=0.5
            )
        )
        builder.add_parameter(
            pybop.Parameter(
                "Positive electrode active material volume fraction", initial_value=0.5
            )
        )

        # Add cost without a sigma parameter
        builder.add_cost(
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]")
        )
        problem = builder.build()

        problem.set_params(np.array([0.5, 0.5]))
        value1 = problem.run()
        problem.set_params(np.array([0.65, 0.65]))
        value2 = problem.run()
        assert abs((value1 - value2) / value1) > 1e-5
        problem.set_params(np.array([0.5, 0.5]))
        value1s, grad1s = problem.run_with_sensitivities()
        assert grad1s.shape == (2,)
        problem.set_params(np.array([0.65, 0.65]))
        value2s, grad2s = problem.run_with_sensitivities()
        np.testing.assert_allclose(value1s, value1, rtol=5e-4)
        np.testing.assert_allclose(value2s, value2, rtol=5e-4)

    def test_eis_builder(self, model_and_params, eis_dataset):
        model, parameter_values = model_and_params
        model.options["surface form"] = "differential"
        builder = pybop.builders.PybammEIS()
        builder.set_dataset(eis_dataset)
        builder.set_simulation(
            model,
            parameter_values=parameter_values,
        )
        builder.add_parameter(
            pybop.Parameter(
                "Negative electrode active material volume fraction", initial_value=0.5
            )
        )
        builder.add_parameter(
            pybop.Parameter(
                "Positive electrode active material volume fraction", initial_value=0.5
            )
        )
        builder.add_cost(pybop.MeanSquaredError(weighting="equal"))
        problem = builder.build()

        problem.set_params(np.array([0.55, 0.55]))
        value1 = problem.run()
        problem.set_params(np.array([0.65, 0.65]))
        value2 = problem.run()
        assert abs((value1 - value2) / value1) > 1e-5

    def test_eis_builder_with_rebuild_parameters(self, model_and_params, eis_dataset):
        model, parameter_values = model_and_params
        model.options["surface form"] = "differential"
        builder = pybop.builders.PybammEIS()
        builder.set_dataset(eis_dataset)
        builder.set_simulation(
            model,
            parameter_values=parameter_values,
        )
        builder.add_parameter(
            pybop.Parameter("Negative electrode thickness [m]", initial_value=5e-5)
        )
        builder.add_parameter(
            pybop.Parameter("Positive particle radius [m]", initial_value=1e-5)
        )
        builder.add_cost(pybop.MeanSquaredError(weighting="domain"))
        problem = builder.build()

        problem.set_params(np.asarray([80e-6, 4.5e-6]))
        value1 = problem.run()
        problem.set_params(np.asarray([85e-6, 5.5e-6]))
        value2 = problem.run()
        assert abs((value1 - value2) / value1) > 1e-5

    def test_thevin_builder(self, dataset):
        model = pybamm.equivalent_circuit.thevenin.Thevenin()
        parameter_values = model.default_parameter_values
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model,
            parameter_values=parameter_values,
        )
        builder.add_parameter(pybop.Parameter("R0 [Ohm]", initial_value=1e-3))
        builder.add_parameter(pybop.Parameter("R1 [Ohm]", initial_value=3e-3))
        builder.add_cost(
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]")
        )
        builder.add_cost(
            pybop.costs.pybamm.MeanAbsoluteError("Voltage [V]", "Voltage [V]")
        )
        problem = builder.build()

        value1 = problem.run()
        problem.set_params(np.array([1.5e-3, 2e-3]))
        value2 = problem.run()
        assert abs((value1 - value2) / value1) > 1e-5

        problem.set_params(np.array([1e-3, 3e-3]))
        value1s, grad1s = problem.run_with_sensitivities()
        assert grad1s.shape == (2,)
        problem.set_params(np.array([1.5e-3, 2e-3]))
        value2s, grad2s = problem.run_with_sensitivities()
        np.testing.assert_allclose(value1s, value1, atol=1e-5)
        np.testing.assert_allclose(value2s, value2, atol=1e-5)

    def test_pure_python_builder(self):
        dataset = pybop.Dataset(
            {"Time / s": np.linspace(0, 1, 10), "Output": np.ones(10)}
        )

        def model(x: float | list):
            output = x * dataset["Time / s"] ** 2
            sse = np.sum((output - dataset["Output"]) ** 2)
            return sse

        builder = pybop.builders.Python()
        builder.add_parameter(pybop.Parameter("x", initial_value=1))
        builder.add_fun(model)
        problem = builder.build()

        problem.set_params(np.array([3.0]))
        value1 = problem.run()
        assert value1 > 0

        # Test sensitivities
        def model_with_sens(x: float | list):
            output = x * dataset["Time / s"] ** 2
            sens = 2 * x * dataset["Time / s"]
            sse = np.sum((output - dataset["Output"]) ** 2)
            sse_grad = 2 * np.sum(output - dataset["Output"]) * np.sum(sens)
            return sse, sse_grad

        builder = pybop.builders.Python()
        builder.add_parameter(pybop.Parameter("x", initial_value=1))
        builder.add_fun_with_sens(model_with_sens=model_with_sens)
        problem_sens = builder.build()
        problem_sens.set_params(np.asarray([3.0]))
        val, sens = problem_sens.run_with_sensitivities()
        assert val > 0
        assert sens > 0

        # Test incorrect model
        with pytest.raises(TypeError, match="Model must be callable"):
            builder.add_fun([2.0])

    def test_build_with_initial_state(self, model_and_params, dataset):
        model, parameter_values = model_and_params
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model,
            parameter_values=parameter_values,
            initial_state="4.0 V",
        )
        builder.add_parameter(
            pybop.Parameter(
                "Negative electrode active material volume fraction", initial_value=0.5
            )
        )
        builder.add_parameter(
            pybop.Parameter(
                "Positive electrode active material volume fraction", initial_value=0.5
            )
        )
        builder.add_cost(
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]")
        )
        problem = builder.build()

        # First build
        problem.set_params(np.array([0.5, 0.5]))
        value1 = problem.run()
        built_model_1 = problem.pipeline.built_model.new_copy()

        # Second build w/ SOC instead of Voltage
        builder.set_simulation(
            model,
            parameter_values=parameter_values,
            initial_state=0.5,
        )
        problem2 = builder.build()
        problem2.set_params(np.array([0.5, 0.5]))
        value2 = problem2.run()
        built_model_2 = problem2.pipeline.built_model.new_copy()

        # Assert builds are different
        assert abs((value1 - value2) / value1) > 1e-5
        assert built_model_1 != built_model_2

    def test_build_on_eval(self, model_and_params, dataset):
        model, parameter_values = model_and_params
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model,
            parameter_values=parameter_values,
            initial_state=0.5,
            build_on_eval=False,
        )
        builder.add_parameter(
            pybop.Parameter(
                "Negative electrode active material volume fraction", initial_value=0.6
            )
        )
        builder.add_parameter(
            pybop.Parameter(
                "Positive electrode active material volume fraction", initial_value=0.6
            )
        )
        builder.add_cost(
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]")
        )
        problem = builder.build()
        assert not problem.pipeline.requires_rebuild

        # First build w/o `build_on_eval`
        problem.set_params(np.array([0.5, 0.5]))
        value1 = problem.run()
        built_model_1 = problem.pipeline.built_model.new_copy()

        # Second build w/ `build_on_eval`
        builder.set_simulation(
            model,
            parameter_values=parameter_values,
            initial_state=0.5,
            build_on_eval=True,
        )
        problem2 = builder.build()
        problem2.set_params(np.array([0.5, 0.5]))
        value2 = problem2.run()
        built_model_2 = problem2._pipeline.built_model.new_copy()

        # Assert builds are different
        assert abs((value1 - value2) / value1) > 1e-5
        assert built_model_1 != built_model_2

    def test_build_no_parameters(self, dataset):
        builder = pybop.builders.Python()
        builder.add_fun(lambda x: x**2)
        with pytest.raises(
            ValueError, match="No parameters have been added to the builder."
        ):
            builder.build()

    def test_set_formation_concentrations(self):
        parameter_values = pybamm.ParameterValues("Chen2020")
        pybop.builders.set_formation_concentrations(parameter_values)

        assert (
            parameter_values["Initial concentration in negative electrode [mol.m-3]"]
            == 0
        )
        assert (
            parameter_values["Initial concentration in positive electrode [mol.m-3]"]
            > 0
        )

    def test_multi_fitting_builder(self, model_and_params, dataset):
        model, parameter_values = model_and_params
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model,
            parameter_values=parameter_values,
        )
        builder.add_parameter(
            pybop.Parameter(
                "Negative electrode active material volume fraction", initial_value=0.6
            )
        )
        builder.add_parameter(
            pybop.Parameter(
                "Positive electrode active material volume fraction", initial_value=0.6
            )
        )
        builder.add_cost(
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]")
        )
        problem = builder.build()
        problem2 = builder.build()

        # Compute costs for each problem
        problem.set_params(np.asarray([0.6, 0.6]))
        value1 = problem.run()

        problem2.set_params(np.asarray([0.6, 0.6]))
        value2 = problem2.run()

        # Build multi-problem
        multi_builder = pybop.builders.MultiFitting()
        multi_builder.add_problem(problem)
        multi_builder.add_problem(problem2)
        multi_problem = multi_builder.build()

        # Compute costs
        multi_problem.set_params(np.asarray([0.6, 0.6]))
        value3 = multi_problem.run()

        multi_problem.set_params(np.asarray([0.7, 0.7]))
        value4 = multi_problem.run()

        assert (value1 + value2) == value3  # Ind. problems == multi-problem
        assert abs((value3 - value4) / value3) > 1e-5

        multi_problem.set_params(np.asarray([0.6, 0.6]))
        value3s, grad3s = multi_problem.run_with_sensitivities()
        assert grad3s.shape == (2,)

        multi_problem.set_params(np.asarray([0.7, 0.7]))
        value4s, grad4s = multi_problem.run_with_sensitivities()

        np.testing.assert_allclose(value3s, value3, atol=1e-5)
        np.testing.assert_allclose(value4s, value4, atol=1e-5)
