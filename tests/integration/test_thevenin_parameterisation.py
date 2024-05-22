import numpy as np
import pytest
from flaky import flaky

import pybop


class TestTheveninParameterisation:
    """
    A class to test the flaky optimisers on a simple model.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        # Set random seed for reproducibility
        np.random.seed(0)
        self.ground_truth = np.array([0.05, 0.05]) + np.random.normal(
            loc=0.0, scale=0.01, size=2
        )

    @pytest.fixture
    def model(self):
        parameter_set = pybop.ParameterSet(
            json_path="examples/scripts/parameters/initial_ecm_parameters.json"
        )
        parameter_set.import_parameters()
        return pybop.empirical.Thevenin(parameter_set=parameter_set)

    @pytest.fixture
    def parameters(self):
        return [
            pybop.Parameter(
                "R0 [Ohm]",
                prior=pybop.Gaussian(0.05, 0.01),
                bounds=[0.01, 0.1],
            ),
            pybop.Parameter(
                "R1 [Ohm]",
                prior=pybop.Gaussian(0.05, 0.01),
                bounds=[0.01, 0.1],
            ),
        ]

    @pytest.fixture(params=[pybop.RootMeanSquaredError, pybop.SumSquaredError])
    def cost_class(self, request):
        return request.param

    @pytest.fixture
    def cost(self, model, parameters, cost_class):
        # Form dataset
        solution = self.getdata(model, self.ground_truth)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data,
            }
        )

        # Define the cost to optimise
        problem = pybop.FittingProblem(model, parameters, dataset)
        return cost_class(problem)

    @pytest.mark.parametrize(
        "optimiser",
        [pybop.SciPyMinimize, pybop.GradientDescent, pybop.PSO],
    )
    @flaky(max_runs=8, min_passes=1)  # These can be very flaky
    @pytest.mark.integration
    def test_optimisers_on_simple_model(self, optimiser, cost):
        x0 = cost.x0
        if optimiser in [pybop.GradientDescent]:
            optim = optimiser(
                cost=cost,
                sigma0=2.5e-4,
                max_iterations=250,
            )
        else:
            optim = optimiser(
                cost=cost,
                sigma0=0.03,
                max_iterations=250,
            )
        if isinstance(optimiser, pybop.BasePintsOptimiser):
            optim.set_max_unchanged_iterations(iterations=55, threshold=1e-5)

        initial_cost = optim.cost(x0)
        x, final_cost = optim.run()

        # Assertions
        if not np.allclose(x0, self.ground_truth, atol=1e-5):
            if optim.minimising:
                assert initial_cost > final_cost
            else:
                assert initial_cost < final_cost
        np.testing.assert_allclose(x, self.ground_truth, atol=1e-2)

    def getdata(self, model, x):
        model.parameter_set.update(
            {
                "R0 [Ohm]": x[0],
                "R1 [Ohm]": x[1],
            }
        )
        experiment = pybop.Experiment(
            [
                ("Discharge at 0.5C for 2 minutes (1 second period)",),
            ]
        )
        sim = model.predict(experiment=experiment)
        return sim
