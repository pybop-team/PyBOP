import numpy as np
import pytest

import pybop
from examples.standalone.cost import StandaloneCost
from examples.standalone.optimiser import StandaloneOptimiser
from examples.standalone.problem import StandaloneProblem


class TestStandalone:
    """
    Class for testing standalone components.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def dataset(self):
        return pybop.Dataset(
            {
                "Time [s]": np.linspace(0, 360, 10),
                "Current function [A]": 0.25 * np.ones(10),
                "Voltage [V]": np.linspace(3.8, 3.7, 10),
            }
        )

    @pytest.fixture
    def parameter(self):
        return pybop.Parameter(
            "Positive electrode active material volume fraction",
            prior=pybop.Gaussian(0.6, 0.02),
            bounds=[0.58, 0.62],
        )

    @pytest.fixture
    def model(self):
        model = pybop.lithium_ion.SPM()
        model.build(initial_state={"Initial open-circuit voltage [V]": 3.8})
        return model

    @pytest.fixture
    def problem(self, model, parameter, dataset):
        return pybop.FittingProblem(
            model,
            parameter,
            dataset,
        )

    @pytest.fixture
    def cost(self, problem):
        return pybop.SumSquaredError(problem)

    def test_standalone_optimiser(self, cost):
        # Define cost function
        optim = StandaloneOptimiser(cost, maxiter=10, x0=[0.6])
        assert optim.name() == "StandaloneOptimiser"

        results = optim.run()
        assert optim.cost(optim.x0) > results.final_cost
        assert 0.0 <= results.x <= 1.0

        # Test with bounds
        optim = StandaloneOptimiser(
            cost, maxiter=10, x0=[0.6], bounds=dict(upper=[0.8], lower=[0.3])
        )

        results = optim.run()
        assert optim.cost(optim.x0) > results.final_cost
        assert 0.3 <= results.x <= 0.8

    def test_optimisation_on_standalone_cost(self, problem):
        # Build an Optimisation problem with a StandaloneCost
        cost = StandaloneCost(problem)
        optim = pybop.SciPyDifferentialEvolution(cost=cost)
        results = optim.run()

        optim.x0 = optim.log["x"][0]
        initial_cost = optim.cost(optim.x0)
        assert initial_cost > results.final_cost
        np.testing.assert_allclose(results.final_cost, 1460, atol=1)

    def test_standalone_problem(self):
        # Define parameters to estimate
        parameters = pybop.Parameters(
            pybop.Parameter(
                "Gradient",
                prior=pybop.Gaussian(4.2, 0.02),
                bounds=[-1, 10],
            ),
            pybop.Parameter(
                "Intercept",
                prior=pybop.Gaussian(3.3, 0.02),
                bounds=[-1, 10],
            ),
        )

        # Define target data
        t_eval = np.linspace(0, 1, 100)
        x0 = np.array([3, 4])
        dataset = pybop.Dataset(
            {
                "Time [s]": t_eval,
                "Output": x0[0] * t_eval + x0[1],
            }
        )
        signal = "Output"

        # Define a Problem without a Model
        problem = StandaloneProblem(parameters, dataset, signal=signal)

        # Test the Problem with a Cost
        rmse_cost = pybop.RootMeanSquaredError(problem)
        rmse_x = rmse_cost([1, 2])
        rmse_grad_x = rmse_cost([1, 2], calculate_grad=True)

        np.testing.assert_allclose(rmse_x, 3.05615, atol=1e-2)
        np.testing.assert_allclose(rmse_grad_x[1], [-0.54645, 0.0], atol=1e-2)

        # Test the sensitivities
        sums_cost = pybop.SumSquaredError(problem)
        x = sums_cost([1, 2], calculate_grad=True)

        np.testing.assert_allclose(x[0], 934.006734006734, atol=1e-2)
        np.testing.assert_allclose(x[1], [-334.006734, 0.0], atol=1e-2)

        # Test problem construction errors
        for bad_dataset in [
            pybop.Dataset({"Time [s]": np.array([0])}),
            pybop.Dataset(
                {
                    "Time [s]": np.array([-1]),
                    "Output": np.array([0]),
                }
            ),
            pybop.Dataset(
                {
                    "Time [s]": np.array([1, 0]),
                    "Output": np.array([0, 0]),
                }
            ),
            pybop.Dataset(
                {
                    "Time [s]": np.array([0]),
                    "Output": np.array([0, 0]),
                }
            ),
            pybop.Dataset(
                {
                    "Time [s]": np.array([[0], [0]]),
                    "Output": np.array([0, 0]),
                }
            ),
        ]:
            with pytest.raises(ValueError):
                StandaloneProblem(parameters, bad_dataset, signal=signal)
