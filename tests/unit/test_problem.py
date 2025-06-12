from typing import Union

import numpy as np
import pybamm
import pytest
from pybamm import IDAKLUSolver

import pybop


class TestProblem:
    """
    A class to test the problem class.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def model(self):
        return pybamm.lithium_ion.SPM()

    @pytest.fixture
    def parameters(self):
        return pybop.Parameters(
            pybop.Parameter(
                "Negative particle radius [m]",
                prior=pybop.Gaussian(2e-05, 0.1e-5),
                bounds=[1e-6, 5e-5],
            ),
            pybop.Parameter(
                "Positive particle radius [m]",
                prior=pybop.Gaussian(0.5e-05, 0.1e-5),
                bounds=[1e-6, 5e-5],
            ),
        )

    @pytest.fixture
    def parameter_values(self):
        return pybamm.ParameterValues("Chen2020")

    @pytest.fixture
    def experiment(self):
        return pybamm.Experiment(
            [
                (
                    "Discharge at 1C for 5 minutes (1 second period)",
                    "Rest for 2 minutes (1 second period)",
                    "Charge at 1C for 5 minutes (1 second period)",
                    "Rest for 2 minutes (1 second period)",
                ),
            ]
            * 2
        )

    @pytest.fixture
    def dataset(self, model, parameter_values, experiment):
        sim = pybamm.Simulation(model, experiment, parameter_values=parameter_values)
        sol = sim.solve()
        return pybop.Dataset(
            {
                "Time [s]": sol["Time [s]"].data,
                "Current function [A]": sol["Current [A]"].data,
                "Voltage [V]": sol["Terminal voltage [V]"].data,
            }
        )

    @pytest.fixture
    def eis_dataset(self):
        return pybop.Dataset(
            {
                "Frequency [Hz]": np.logspace(-4, 5, 30),
                "Current function [A]": np.ones(30) * 0.0,
                "Impedance": np.ones(30) * 0.0,
            }
        )

    @pytest.fixture
    def signal(self):
        return "Voltage [V]"

    def test_builder(self, model, parameter_values, experiment, dataset):
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model,
            parameter_values=parameter_values,
            solver=IDAKLUSolver(atol=1e-6, rtol=1e-6),
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
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]", 1.0)
        )
        builder.add_cost(
            pybop.costs.pybamm.MeanAbsoluteError("Voltage [V]", "Voltage [V]")
        )
        problem = builder.build()

        assert problem is not None
        problem.set_params(np.array([0.6, 0.6]))
        value1 = problem.run()
        problem.set_params(np.array([0.7, 0.7]))
        value2 = problem.run()
        assert (value1 - value2) / value1 > 1e-5
        problem.set_params(np.array([0.6, 0.6]))
        value1s, grad1s = problem.run_with_sensitivities()
        assert grad1s.shape == (2,)
        problem.set_params(np.array([0.7, 0.7]))
        value2s, grad2s = problem.run_with_sensitivities()
        np.testing.assert_allclose(value1s, value1, atol=1e-5)
        np.testing.assert_allclose(value2s, value2, atol=1e-5)

        # Test building twice
        problem2 = builder.build()
        assert problem2 is not None
        assert problem2 != problem

    def test_builder_likelihoods(self, model, parameter_values, experiment, dataset):
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model,
            parameter_values=parameter_values,
            solver=IDAKLUSolver(atol=1e-7, rtol=1e-7),
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
            pybop.costs.pybamm.NegativeGaussianLogLikelihood(
                "Voltage [V]", "Voltage [V]", 1e-2
            )
        )
        problem = builder.build()

        assert problem is not None
        problem.set_params(np.array([0.6, 0.6]))
        value1 = problem.run()
        problem.set_params(np.array([0.7, 0.7]))
        value2 = problem.run()
        assert (value1 - value2) / value1 < 0.0
        problem.set_params(np.array([0.6, 0.6]))
        value1s, grad1s = problem.run_with_sensitivities()
        assert grad1s.shape == (2,)
        problem.set_params(np.array([0.7, 0.7]))
        value2s, grad2s = problem.run_with_sensitivities()
        np.testing.assert_allclose(value1s, value1, atol=1e-5)
        np.testing.assert_allclose(value2s, value2, atol=1e-5)

        # Test with estimated sigma
        builder.add_cost(
            pybop.costs.pybamm.NegativeGaussianLogLikelihood(
                "Voltage [V]",
                "Voltage [V]",
            )
        )
        problem2 = builder.build()
        problem2.set_params(np.array([0.6, 0.6, 1e-2]))
        value3 = problem2.run()
        np.testing.assert_allclose(2 * value1, value3)

        # Different sigma
        problem2.set_params(np.array([0.6, 0.6, 1e-3]))
        value4 = problem2.run()
        assert np.not_equal(2 * value1, value4)

    def test_builder_posterior(self, model, parameter_values, experiment, dataset):
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model,
            parameter_values=parameter_values,
            solver=IDAKLUSolver(atol=1e-6, rtol=1e-6),
        )
        builder.add_parameter(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                initial_value=0.6,
                prior=pybop.Gaussian(0.6, 0.1),
            )
        )
        builder.add_parameter(
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                initial_value=0.6,
                prior=pybop.Gaussian(0.6, 0.1),
            )
        )
        builder.add_cost(
            pybop.costs.pybamm.NegativeGaussianLogLikelihood(
                "Voltage [V]", "Voltage [V]", 1e-2
            )
        )
        problem = builder.build()

        assert problem is not None
        assert problem._use_posterior is True
        problem.set_params(np.array([0.6, 0.6]))
        value1 = problem.run()
        problem.set_params(np.array([0.7, 0.7]))
        value2 = problem.run()
        assert (value1 - value2) / value1 < 0.0
        problem.set_params(np.array([0.6, 0.6]))
        value1s, grad1s = problem.run_with_sensitivities()
        assert grad1s.shape == (2,)
        problem.set_params(np.array([0.7, 0.7]))
        value2s, grad2s = problem.run_with_sensitivities()
        np.testing.assert_allclose(value1s, value1, atol=1e-5)
        np.testing.assert_allclose(value2s, value2, atol=1e-5)

    def test_builder_with_rebuild_params(
        self, model, parameter_values, experiment, dataset
    ):
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model,
            parameter_values=parameter_values,
        )
        builder.add_parameter(
            pybop.Parameter("Negative electrode thickness [m]", initial_value=1e-6)
        )
        builder.add_parameter(
            pybop.Parameter("Positive particle radius [m]", initial_value=1e-5)
        )
        sigma = pybop.Parameter("sigma", bounds=[0, 1], initial_value=0.5)
        builder.add_cost(
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]", sigma)
        )
        builder.add_cost(
            pybop.costs.pybamm.MeanAbsoluteError("Voltage [V]", "Voltage [V]")
        )
        problem = builder.build()

        assert problem is not None
        problem.set_params(np.array([1e-5, 0.5e-6, 1e-3]))
        value1 = problem.run()
        problem.set_params(np.array([2e-5, 1.5e-6, 1e-3]))
        value2 = problem.run()
        assert (value1 - value2) / value1 > 1e-5

    def test_builder_with_cost_hypers(
        self, model, parameter_values, experiment, dataset
    ):
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model,
            parameter_values=parameter_values,
            solver=IDAKLUSolver(atol=1e-6, rtol=1e-6),
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

        # Add cost without a sigma parameter
        builder.add_cost(
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]")
        )
        problem = builder.build()

        assert problem is not None
        problem.set_params(np.array([0.6, 0.6, 0.01]))
        value1 = problem.run()
        problem.set_params(np.array([0.7, 0.7, 0.01]))
        value2 = problem.run()
        assert (value1 - value2) / value1 > 1e-5
        problem.set_params(np.array([0.6, 0.6, 0.01]))
        value1s, grad1s = problem.run_with_sensitivities()
        assert grad1s.shape == (3,)
        problem.set_params(np.array([0.7, 0.7, 0.01]))
        value2s, grad2s = problem.run_with_sensitivities()
        np.testing.assert_allclose(value1s, value1)
        np.testing.assert_allclose(value2s, value2)

    def test_eis_builder(self, model, parameter_values, experiment, eis_dataset):
        builder = pybop.builders.PybammEIS()
        builder.set_dataset(eis_dataset)
        builder.set_simulation(model, parameter_values=parameter_values)
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
        builder.add_cost(pybop.MeanSquaredError(weighting="equal"))
        problem = builder.build()

        assert problem is not None
        problem.set_params(np.array([0.6, 0.6]))
        value1 = problem.run()
        problem.set_params(np.array([0.7, 0.7]))
        value2 = problem.run()
        assert (value1 - value2) / value1 > 1e-5

    def test_eis_builder_with_rebuild_parameters(
        self, model, parameter_values, experiment, eis_dataset
    ):
        builder = pybop.builders.PybammEIS()
        builder.set_dataset(eis_dataset)
        builder.set_simulation(model, parameter_values=parameter_values)
        builder.add_parameter(
            pybop.Parameter("Negative electrode thickness [m]", initial_value=1e-6)
        )
        builder.add_parameter(
            pybop.Parameter("Positive particle radius [m]", initial_value=1e-5)
        )
        builder.add_cost(pybop.MeanSquaredError(weighting="domain"))
        problem = builder.build()

        assert problem is not None
        problem.set_params(np.array([1e-5, 0.5e-6]))
        value1 = problem.run()
        problem.set_params(np.array([2e-5, 1.5e-6]))
        value2 = problem.run()
        assert (value1 - value2) / value1 > 1e-5

    def test_pure_python_builder(self):
        dataset = pybop.Dataset(
            {"Time / s": np.linspace(0, 1, 10), "Output": np.ones(10)}
        )

        def model(x: Union[float, list]):
            output = x * dataset["Time / s"] ** 2
            sse = np.sum((output - dataset["Output"]) ** 2)
            return sse

        builder = pybop.builders.Python()
        builder.add_parameter(pybop.Parameter("x", initial_value=1))
        builder.add_func(model)
        problem = builder.build()

        assert problem is not None
        problem.set_params(np.array([3.0]))
        value1 = problem.run()
        assert value1 > 0

        # Test sensitivities
        def model_with_sens(x: Union[float, list]):
            output = x * dataset["Time / s"] ** 2
            sens = 2 * x * dataset["Time / s"]
            sse = np.sum((output - dataset["Output"]) ** 2)
            sse_grad = 2 * np.sum(output - dataset["Output"]) * np.sum(sens)
            return sse, sse_grad

        builder = pybop.builders.Python()
        builder.add_parameter(pybop.Parameter("x", initial_value=1))
        builder.add_func_with_sens(model_with_sens=model_with_sens)
        problem_sens = builder.build()
        assert problem_sens is not None
        problem_sens.set_params(np.asarray([3.0]))
        val, sens = problem_sens.run_with_sensitivities()
        assert val > 0
        assert sens > 0

        # Test incorrect model
        with pytest.raises(TypeError, match="Model must be callable"):
            builder.add_func([2.0])

    def test_build_with_initial_state(
        self, model, parameter_values, experiment, dataset
    ):
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model,
            parameter_values=parameter_values,
            solver=IDAKLUSolver(atol=1e-6, rtol=1e-6),
            initial_state="4.0 V",
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
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]", 1.0)
        )
        problem = builder.build()

        assert problem is not None

        # First build
        problem.set_params(np.array([0.6, 0.6]))
        value1 = problem.run()
        built_model_1 = problem._pipeline.built_model.new_copy()

        # Second build
        problem.set_params(np.array([0.7, 0.7]))
        value2 = problem.run()
        built_model_2 = problem._pipeline.built_model.new_copy()

        # Assert builds are different
        assert (value1 - value2) / value1 > 1e-5
        assert built_model_1 != built_model_2

    def test_build_no_parameters(self, dataset):
        builder = pybop.builders.Python()
        builder.add_func(lambda x: x**2)
        with pytest.raises(
            ValueError, match="No parameters have been added to the builder."
        ):
            builder.build()

    def test_parameter_sensitivities(
        self, model, parameter_values, experiment, dataset
    ):
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model,
            parameter_values=parameter_values,
            solver=IDAKLUSolver(atol=1e-6, rtol=1e-6),
        )
        builder.add_parameter(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                initial_value=0.6,
                bounds=[0.5, 0.8],
            )
        )
        builder.add_parameter(
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                initial_value=0.6,
                bounds=[0.5, 0.8],
            )
        )
        builder.add_cost(
            pybop.costs.pybamm.NegativeGaussianLogLikelihood(
                "Voltage [V]", "Voltage [V]", 1e-2
            )
        )
        problem = builder.build()

        result = problem.sensitivity_analysis(4)

        # Assertions
        assert isinstance(result, dict)
        assert "S1" in result
        assert "ST" in result
        assert isinstance(result["S1"], np.ndarray)
        assert isinstance(result["S2"], np.ndarray)
        assert isinstance(result["ST"], np.ndarray)
        assert isinstance(result["S1_conf"], np.ndarray)
        assert isinstance(result["ST_conf"], np.ndarray)
        assert isinstance(result["S2_conf"], np.ndarray)
        assert result["S1"].shape == (2,)
        assert result["ST"].shape == (2,)
