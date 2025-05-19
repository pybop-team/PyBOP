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
    def first_model(self):
        return pybamm.lithium_ion.SPM()

    @pytest.fixture
    def second_model(self):
        return pybamm.lithium_ion.SPMe()

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
    def dataset(self, first_model, parameter_values, experiment):
        sim = pybamm.Simulation(
            first_model, experiment, parameter_values=parameter_values
        )
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

    def test_builder(self, first_model, parameter_values, experiment, dataset):
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            first_model,
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
        builder.add_cost(pybop.PybammSumSquaredError("Voltage [V]", "Voltage [V]", 1.0))
        problem = builder.build()

        assert problem is not None
        problem.set_params(np.array([0.6, 0.6]))
        value1 = problem.run()
        problem.set_params(np.array([0.7, 0.7]))
        value2 = problem.run()
        assert (value1 - value2) / value1 > 1e-5
        problem.set_params(np.array([0.6, 0.6]))
        value1s, grad1s = problem.run_with_sensitivities()
        assert grad1s.shape == (2, 1)
        problem.set_params(np.array([0.7, 0.7]))
        value2s, grad2s = problem.run_with_sensitivities()
        np.testing.assert_allclose(value1s, value1)
        np.testing.assert_allclose(value2s, value2)

    def test_builder_with_rebuild_params(
        self, first_model, parameter_values, experiment, dataset
    ):
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            first_model,
            parameter_values=parameter_values,
        )
        builder.add_parameter(
            pybop.Parameter("Negative electrode thickness [m]", initial_value=1e-6)
        )
        builder.add_parameter(
            pybop.Parameter("Positive particle radius [m]", initial_value=1e-5)
        )
        builder.add_cost(pybop.PybammSumSquaredError("Voltage [V]", "Voltage [V]"))
        problem = builder.build()

        assert problem is not None

        assert problem is not None
        problem.set_params(np.array([1e-5, 0.5e-6]))
        value1 = problem.run()
        problem.set_params(np.array([2e-5, 1.5e-6]))
        value2 = problem.run()
        assert (value1 - value2) / value1 > 1e-5

    def test_eis_builder(self, first_model, parameter_values, experiment, eis_dataset):
        builder = pybop.builders.PybammEIS()
        builder.set_dataset(eis_dataset)
        builder.set_simulation(first_model, parameter_values=parameter_values)
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
        builder.add_cost(pybop.NewMeanSquaredError(weighting="equal"))
        problem = builder.build()

        assert problem is not None
        problem.set_params(np.array([0.6, 0.6]))
        value1 = problem.run()
        problem.set_params(np.array([0.7, 0.7]))
        value2 = problem.run()
        assert (value1 - value2) / value1 > 1e-5

    def test_eis_builder_with_rebuild_parameters(
        self, first_model, parameter_values, experiment, eis_dataset
    ):
        builder = pybop.builders.PybammEIS()
        builder.set_dataset(eis_dataset)
        builder.set_simulation(first_model, parameter_values=parameter_values)
        builder.add_parameter(
            pybop.Parameter("Negative electrode thickness [m]", initial_value=1e-6)
        )
        builder.add_parameter(
            pybop.Parameter("Positive particle radius [m]", initial_value=1e-5)
        )
        builder.add_cost(pybop.NewMeanSquaredError(weighting="domain"))
        problem = builder.build()

        assert problem is not None
        problem.set_params(np.array([1e-5, 0.5e-6]))
        value1 = problem.run()
        problem.set_params(np.array([2e-5, 1.5e-6]))
        value2 = problem.run()
        assert (value1 - value2) / value1 > 1e-5

    def test_pure_python_builder(self):
        dataset = pybop.Dataset(
            {"Time / s": np.linspace(0, 1, 10), "Current [A]": np.ones(10)}
        )

        def model(x):
            return x**2

        builder = pybop.builders.Python()
        builder.add_func(model)
        builder.set_dataset(dataset)

        # builder.add_cost()
        # problem = builder.build()

        # Assertion to add
        # Parameters

    def test_build_with_initial_state(
        self, first_model, parameter_values, experiment, dataset
    ):
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            first_model,
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
        builder.add_cost(pybop.PybammSumSquaredError("Voltage [V]", "Voltage [V]", 1.0))
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
        builder.set_dataset(dataset)
        with pytest.raises(
            ValueError, match="No parameters have been added to the builder."
        ):
            problem = builder.build()
