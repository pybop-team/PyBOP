import numpy as np
import pybamm
import pytest

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
        # x0 = np.array([2e-5, 0.5e-5])
        # parameter_values.update(
        #     {
        #         "Negative particle radius [m]": x0[0],
        #         "Positive particle radius [m]": x0[1],
        #     }
        # )
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
    def signal(self):
        return "Voltage [V]"

    def test_builder(self, first_model, parameter_values, experiment, dataset):
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.add_simulation(first_model, parameter_values=parameter_values)
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
        # builder.add_cost(pybop.costs.SumSquaredError())
        problem = builder.build()

        assert problem is not None
        problem.set_params(np.array([0.6, 0.6]))
        assert problem.run() is not None

        # Assertion to add
        # Multi-simulation building
        # Parameters
        # Rebuild required construction

    def test_builder_with_rebuild_params(
        self, first_model, parameter_values, experiment, dataset
    ):
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.add_simulation(first_model, parameter_values=parameter_values)
        builder.add_parameter(
            pybop.Parameter("Negative electrode thickness [m]", initial_value=1e-6)
        )
        builder.add_parameter(
            pybop.Parameter("Positive electrode thickness [m]", initial_value=2e-6)
        )
        # builder.add_cost(pybop.costs.SumSquaredError())
        problem = builder.build()

        assert problem is not None
        problem.set_params(np.array([0.6, 0.6]))
        assert problem.run() is not None

        # Assertion to add
        # Multi-simulation building
        # Parameters
        # Rebuild required construction
