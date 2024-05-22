import numpy as np
import pytest

import pybop


class TestTheveninParameterisation:
    """
    A class to test a subset of optimisers on a simple model.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        # Set random seed for reproducibility
        np.random.seed(0)
        self.ground_truth = np.array([0.03, 0.03]) + np.random.normal(
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
        return pybop.Parameters(
            [
                pybop.Parameter(
                    "R0 [Ohm]",
                    prior=pybop.Gaussian(0.03, 0.01),
                    bounds=[0, 0.06],
                ),
                pybop.Parameter(
                    "R1 [Ohm]",
                    prior=pybop.Gaussian(0.03, 0.01),
                    bounds=[0, 0.06],
                ),
            ]
        )

    @pytest.fixture(params=[pybop.RootMeanSquaredError, pybop.SumSquaredError])
    def cost_class(self, request):
        return request.param

    @pytest.fixture
    def cost(self, model, parameters, cost_class):
        # Form dataset
        solution = self.get_data(model, parameters, self.ground_truth)
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
    @pytest.mark.integration
    def test_optimisers_on_simple_model(self, optimiser, cost):
        x0 = cost.x0
        if optimiser in [pybop.GradientDescent]:
            parameterisation = pybop.Optimisation(
                cost=cost, optimiser=optimiser, sigma0=2.5e-4
            )
        else:
            parameterisation = pybop.Optimisation(
                cost=cost, optimiser=optimiser, sigma0=0.03
            )

        parameterisation.set_max_unchanged_iterations(iterations=35, threshold=1e-5)
        parameterisation.set_max_iterations(375)
        initial_cost = parameterisation.cost(x0)
        x, final_cost = parameterisation.run()

        # Assertions
        if not np.allclose(x0, self.ground_truth, atol=1e-5):
            assert initial_cost > final_cost
        np.testing.assert_allclose(x, self.ground_truth, atol=1e-2)

    def get_data(self, model, parameters, x):
        model.parameters = parameters
        experiment = pybop.Experiment(
            [
                (
                    "Discharge at 0.5C for 2 minutes (4 seconds period)",
                    "Charge at 0.5C for 2 minutes (4 seconds period)",
                ),
            ]
        )
        sim = model.predict(experiment=experiment, inputs=x)
        return sim
