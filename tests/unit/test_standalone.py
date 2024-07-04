import numpy as np
import pytest

import pybop
from examples.standalone.cost import StandaloneCost
from examples.standalone.optimiser import StandaloneOptimiser
from examples.standalone.problem import StandaloneProblem


class TestStandalone:
    """
    Class for testing stanadalone components.
    """

    @pytest.mark.unit
    def test_standalone_optimiser(self):
        optim = StandaloneOptimiser()
        assert optim.name() == "StandaloneOptimiser"

        x, final_cost = optim.run()
        assert optim.cost(optim.x0) > final_cost
        np.testing.assert_allclose(x, [2, 4], atol=1e-2)

        # Test with bounds
        optim = StandaloneOptimiser(bounds=dict(upper=[5, 6], lower=[1, 2]))

        x, final_cost = optim.run()
        assert optim.cost(optim.x0) > final_cost
        np.testing.assert_allclose(x, [2, 4], atol=1e-2)

    @pytest.mark.unit
    def test_optimisation_on_standalone_cost(self):
        # Build an Optimisation problem with a StandaloneCost
        cost = StandaloneCost()
        optim = pybop.SciPyDifferentialEvolution(cost=cost)
        x, final_cost = optim.run()

        optim.x0 = optim.log["x"][0][0]
        initial_cost = optim.cost(optim.x0)
        assert initial_cost > final_cost
        np.testing.assert_allclose(final_cost, 42, atol=1e-1)

    @pytest.mark.unit
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
        rmse_grad_x = rmse_cost.evaluateS1([1, 2])

        np.testing.assert_allclose(rmse_x, 3.05615, atol=1e-2)
        np.testing.assert_allclose(rmse_grad_x[1], [-0.54645, 0.0], atol=1e-2)

        # Test the sensitivities
        sums_cost = pybop.SumSquaredError(problem)
        x = sums_cost.evaluateS1([1, 2])

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
