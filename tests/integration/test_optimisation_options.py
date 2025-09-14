import sys

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
        parameter_set = pybop.ParameterSet("Chen2020")
        x = self.ground_truth
        parameter_set.update(
            {
                "Negative electrode active material volume fraction": x[0],
                "Positive electrode active material volume fraction": x[1],
            }
        )
        return pybop.lithium_ion.SPM(parameter_set=parameter_set)

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
    def cost(self, model, parameters):
        # Form dataset
        initial_state = {"Initial SoC": 0.5}
        solution = self.get_data(model, initial_state)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": self.noisy(solution["Voltage [V]"].data, 0.002),
            }
        )

        # Define the cost to optimise
        problem = pybop.FittingProblem(model, parameters, dataset)
        return pybop.SumSquaredError(problem)

    @pytest.mark.parametrize(
        "f_guessed",
        [
            True,
            False,
        ],
    )
    def test_optimisation_f_guessed(self, f_guessed, cost):
        x0 = cost.parameters.get_initial_values()
        # Test each optimiser
        optim = pybop.XNES(
            cost=cost,
            sigma0=0.05,
            max_iterations=100,
            max_unchanged_iterations=25,
            absolute_tolerance=1e-5,
            use_f_guessed=f_guessed,
            compute_sensitivities=True,
            n_sensitivity_samples=3,
            allow_infeasible_solutions=False,
        )

        # Set parallelisation if not on Windows
        if sys.platform != "win32":
            optim.set_parallel(1)

        initial_cost = optim.cost(x0)
        results = optim.run()

        # Assertions
        assert results.sensitivities is not None
        if not np.allclose(x0, self.ground_truth, atol=1e-5):
            if results.minimising:
                assert initial_cost > results.final_cost
            else:
                assert initial_cost < results.final_cost
        else:
            raise ValueError("Initial value is the same as the ground truth value.")
        np.testing.assert_allclose(results.x, self.ground_truth, atol=1.5e-2)

    def get_data(self, model, initial_state):
        # Update the initial state and save the ground truth initial concentrations
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at 0.5C for 3 minutes (10 second period)",
                    "Charge at 0.5C for 3 minutes (10 second period)",
                ),
            ]
        )
        sim = model.predict(initial_state=initial_state, experiment=experiment)
        return sim
