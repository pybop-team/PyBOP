import numpy as np
import pybamm
import pytest

import pybop


class TestOptimisation:
    """
    A class to run integration tests on the Optimisation class.
    """

    pytestmark = pytest.mark.integration

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ground_truth = np.clip(
            np.asarray([0.55, 0.55]) + np.random.normal(loc=0.0, scale=0.05, size=2),
            a_min=0.4,
            a_max=0.75,
        )

    @pytest.fixture
    def model(self):
        return pybamm.lithium_ion.SPM()

    @pytest.fixture
    def parameter_values(self):
        parameter_values = pybamm.ParameterValues("Chen2020")
        x = self.ground_truth
        parameter_values.update(
            {
                "Negative electrode active material volume fraction": x[0],
                "Positive electrode active material volume fraction": x[1],
            }
        )
        return parameter_values

    @pytest.fixture
    def parameters(self):
        return pybop.Parameters(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.55, 0.05),
                bounds=[0.375, 0.75],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.55, 0.05),
                # no bounds
            ),
        )

    def noisy(self, data, sigma):
        return data + np.random.normal(0, sigma, len(data))

    @pytest.fixture
    def problem(self, model, parameter_values, parameters):
        parameter_values.set_initial_state(0.5)
        dataset = self.get_data(model, parameter_values)

        # Define the cost to optimise
        simulator = pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            input_parameter_names=parameters.names,
            protocol=dataset,
        )
        cost = pybop.SumSquaredError(dataset)
        return pybop.FittingProblem(simulator, parameters, cost)

    @pytest.mark.parametrize(
        "f_guessed",
        [
            True,
            False,
        ],
    )
    def test_optimisation_f_guessed(self, f_guessed, problem):
        x0 = problem.parameters.get_initial_values()
        options = pybop.PintsOptions(
            sigma=0.05,
            max_iterations=100,
            max_unchanged_iterations=25,
            absolute_tolerance=1e-5,
            use_f_guessed=f_guessed,
        )
        optim = pybop.XNES(problem, options=options)

        initial_cost = optim.problem(x0)
        results = optim.run()

        # Assertions
        if not np.allclose(x0, self.ground_truth, atol=1e-5):
            if results.minimising:
                assert initial_cost > results.best_cost
            else:
                assert initial_cost < results.best_cost
        else:
            raise ValueError("Initial value is the same as the ground truth value.")
        np.testing.assert_allclose(results.x, self.ground_truth, atol=1.5e-2)

    def get_data(self, model, parameter_values):
        experiment = pybamm.Experiment(
            [
                "Discharge at 0.5C for 3 minutes (10 second period)",
                "Charge at 0.5C for 3 minutes (10 second period)",
            ]
        )
        solution = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        ).solve()
        return pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": self.noisy(solution["Voltage [V]"].data, 0.002),
            }
        )
