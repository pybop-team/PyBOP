import numpy as np
import pytest

import pybop


class TestTransformation:
    """
    A class for integration tests of the transformation methods.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ground_truth = np.array([0.5, 0.2]) + np.random.normal(
            loc=0.0, scale=0.05, size=2
        )

    @pytest.fixture
    def model(self):
        parameter_set = pybop.ParameterSet.pybamm("Chen2020")
        x = self.ground_truth
        parameter_set.update(
            {
                "Positive electrode active material volume fraction": x[0],
                "Positive electrode conductivity [S.m-1]": x[1],
            }
        )
        return pybop.lithium_ion.SPMe(parameter_set=parameter_set)

    @pytest.fixture
    def parameters(self):
        return pybop.Parameters(
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Uniform(0.4, 0.7),
                bounds=[0.375, 0.725],
                transformation=pybop.LogTransformation(),
            ),
            pybop.Parameter(
                "Positive electrode conductivity [S.m-1]",
                prior=pybop.Uniform(0.05, 0.45),
                bounds=[0.04, 0.5],
                transformation=pybop.LogTransformation(),
            ),
        )

    @pytest.fixture(params=[0.4, 0.7])
    def init_soc(self, request):
        return request.param

    def noise(self, sigma, values):
        return np.random.normal(0, sigma, values)

    @pytest.fixture
    def cost(self, model, parameters, init_soc):
        # Form dataset
        solution = self.get_data(model, init_soc)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data
                + self.noise(0.002, len(solution["Time [s]"].data)),
            }
        )

        # Define the cost to optimise
        problem = pybop.FittingProblem(model, parameters, dataset)
        return pybop.RootMeanSquaredError(problem)

    @pytest.mark.parametrize(
        "optimiser",
        [
            pybop.AdamW,
            pybop.CMAES,
        ],
    )
    @pytest.mark.integration
    def test_spm_optimisers(self, optimiser, cost):
        x0 = cost.parameters.initial_value()
        optim = optimiser(
            cost=cost,
            max_unchanged_iterations=35,
            max_iterations=125,
        )

        initial_cost = optim.cost(x0)
        x, final_cost = optim.run()

        # Assertions
        if not np.allclose(x0, self.ground_truth, atol=1e-5):
            assert initial_cost > final_cost
            np.testing.assert_allclose(x, self.ground_truth, atol=1.5e-2)
        else:
            raise ValueError("Initial value is the same as the ground truth value.")

    def get_data(self, model, init_soc):
        initial_state = {"Initial SoC": init_soc}
        experiment = pybop.Experiment(
            [
                (
                    "Discharge at 1C for 3 minutes (4 second period)",
                    "Charge at 1C for 3 minutes (4 second period)",
                ),
            ]
        )
        sim = model.predict(initial_state=initial_state, experiment=experiment)
        return sim
