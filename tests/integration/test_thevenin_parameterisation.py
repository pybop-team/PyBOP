import numpy as np
import pytest

import pybop


class TestTheveninParameterisation:
    """
    A class to test a subset of optimisers on a simple model.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ground_truth = np.asarray([0.05, 0.05]) + np.random.normal(
            loc=0.0, scale=0.01, size=2
        )

    @pytest.fixture
    def model(self):
        parameter_set = pybop.ParameterSet(
            json_path="examples/scripts/parameters/initial_ecm_parameters.json"
        )
        parameter_set.import_parameters()
        parameter_set.params.update({"C1 [F]": 1000})
        return pybop.empirical.Thevenin(parameter_set=parameter_set)

    @pytest.fixture
    def parameters(self):
        return pybop.Parameters(
            pybop.Parameter(
                "R0 [Ohm]",
                prior=pybop.Gaussian(0.05, 0.01),
                bounds=[0, 0.1],
            ),
            pybop.Parameter(
                "R1 [Ohm]",
                prior=pybop.Gaussian(0.05, 0.01),
                bounds=[0, 0.1],
            ),
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
        x0 = cost.parameters.initial_value()
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
            optim.set_max_unchanged_iterations(iterations=35, absolute_tolerance=1e-5)

        initial_cost = optim.cost(optim.parameters.initial_value())
        x, final_cost = optim.run()

        # Assertions
        if not np.allclose(x0, self.ground_truth, atol=1e-5):
            if optim.minimising:
                assert initial_cost > final_cost
            else:
                assert initial_cost < final_cost
        np.testing.assert_allclose(x, self.ground_truth, atol=1.5e-2)

    def get_data(self, model, parameters, x):
        model.classify_and_update_parameters(parameters)
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
