import numpy as np
import pytest

import pybop


class TestTheveninParameterisation:
    """
    A class to test a subset of optimisers on a simple model.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ground_truth = np.clip(
            np.asarray([0.05, 0.05]) + np.random.normal(loc=0.0, scale=0.01, size=2),
            a_min=0.0,
            a_max=0.1,
        )

    @pytest.fixture
    def model(self):
        parameter_set = pybop.ParameterSet(
            json_path="examples/parameters/initial_ecm_parameters.json"
        )
        parameter_set.import_parameters()
        parameter_set.params.update(
            {
                "C1 [F]": 1000,
                "R0 [Ohm]": self.ground_truth[0],
                "R1 [Ohm]": self.ground_truth[1],
            }
        )
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

    @pytest.fixture
    def dataset(self, model):
        # Form dataset
        solution = self.get_data(model)
        return pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data,
            }
        )

    @pytest.mark.parametrize(
        "cost_class", [pybop.RootMeanSquaredError, pybop.SumSquaredError]
    )
    @pytest.mark.parametrize(
        "optimiser, method",
        [
            (pybop.SciPyMinimize, "trust-constr"),
            (pybop.SciPyMinimize, "SLSQP"),
            (pybop.SciPyMinimize, "COBYLA"),
            (pybop.GradientDescent, ""),
            (pybop.PSO, ""),
        ],
    )
    @pytest.mark.integration
    def test_optimisers_on_simple_model(
        self, model, parameters, dataset, cost_class, optimiser, method
    ):
        # Define the cost to optimise
        problem = pybop.FittingProblem(model, parameters, dataset)
        cost = cost_class(problem)

        x0 = cost.parameters.initial_value()
        if optimiser in [pybop.GradientDescent]:
            optim = optimiser(
                cost=cost, sigma0=2.5e-4, max_iterations=250, method=method
            )
        else:
            optim = optimiser(cost=cost, sigma0=0.03, max_iterations=250, method=method)
        if isinstance(optimiser, pybop.BasePintsOptimiser):
            optim.set_max_unchanged_iterations(iterations=35, absolute_tolerance=1e-5)

        initial_cost = optim.cost(optim.parameters.initial_value())
        results = optim.run()

        # Assertions
        if not np.allclose(x0, self.ground_truth, atol=1e-5):
            if optim.minimising:
                assert initial_cost > results.final_cost
            else:
                assert initial_cost < results.final_cost
        else:
            raise ValueError("Initial value is the same as the ground truth value.")
        np.testing.assert_allclose(results.x, self.ground_truth, atol=1.5e-2)

        if isinstance(optimiser, pybop.SciPyMinimize):
            assert results.scipy_result.success is True

    def get_data(self, model):
        experiment = pybop.Experiment(
            [
                (
                    "Discharge at 0.5C for 2 minutes (4 seconds period)",
                    "Rest for 20 seconds (4 seconds period)",
                    "Charge at 0.5C for 2 minutes (4 seconds period)",
                ),
            ]
        )
        sim = model.predict(experiment=experiment)
        return sim
