import numpy as np
import pytest

import pybop


class TestWeightedCost:
    """
    A class to test the weighted cost function.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.sigma0 = 0.002
        self.ground_truth = np.clip(
            np.asarray([0.55, 0.55]) + np.random.normal(loc=0.0, scale=0.05, size=2),
            a_min=0.4,
            a_max=0.75,
        )

    @pytest.fixture
    def model(self):
        parameter_set = pybop.ParameterSet.pybamm("Chen2020")
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
                prior=pybop.Uniform(0.4, 0.75),
                bounds=[0.375, 0.75],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Uniform(0.4, 0.75),
                # no bounds
            ),
        )

    @pytest.fixture(params=[0.4])
    def init_soc(self, request):
        return request.param

    @pytest.fixture(
        params=[
            (
                pybop.GaussianLogLikelihoodKnownSigma,
                pybop.RootMeanSquaredError,
                pybop.SumSquaredError,
                pybop.MAP,
            )
        ]
    )
    def cost_class(self, request):
        return request.param

    def noise(self, sigma, values):
        return np.random.normal(0, sigma, values)

    @pytest.fixture
    def weighted_fitting_cost(self, model, parameters, cost_class, init_soc):
        # Form dataset
        solution = self.get_data(model, init_soc)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data
                + self.noise(self.sigma0, len(solution["Time [s]"].data)),
            }
        )

        # Define the cost to optimise
        problem = pybop.FittingProblem(model, parameters, dataset)
        costs = []
        for cost in cost_class:
            if issubclass(cost, pybop.MAP):
                costs.append(
                    cost(
                        problem,
                        pybop.GaussianLogLikelihoodKnownSigma,
                        sigma0=self.sigma0,
                    )
                )
            elif issubclass(cost, pybop.BaseLikelihood):
                costs.append(cost(problem, sigma0=self.sigma0))
            else:
                costs.append(cost(problem))

        return pybop.WeightedCost(*costs, weights=[0.1, 1, 0.5, 0.6])

    @pytest.mark.integration
    def test_fitting_costs(self, weighted_fitting_cost):
        x0 = weighted_fitting_cost.parameters.initial_value()
        optim = pybop.CuckooSearch(
            cost=weighted_fitting_cost,
            sigma0=0.03,
            max_iterations=250,
            max_unchanged_iterations=35,
        )

        initial_cost = optim.cost(optim.parameters.initial_value())
        x, final_cost = optim.run()

        # Assertions
        if not np.allclose(x0, self.ground_truth, atol=1e-5):
            if optim.minimising:
                assert initial_cost > final_cost
            else:
                assert initial_cost < final_cost
        np.testing.assert_allclose(x, self.ground_truth, atol=1.5e-2)

    @pytest.fixture(
        params=[
            (
                pybop.GravimetricEnergyDensity,
                pybop.VolumetricEnergyDensity,
            )
        ]
    )
    def design_cost(self, request):
        return request.param

    @pytest.fixture
    def weighted_design_cost(self, model, design_cost):
        initial_state = {"Initial SoC": 1.0}
        parameters = pybop.Parameters(
            pybop.Parameter(
                "Positive electrode thickness [m]",
                prior=pybop.Gaussian(5e-05, 5e-06),
                bounds=[2e-06, 10e-05],
            ),
            pybop.Parameter(
                "Negative electrode thickness [m]",
                prior=pybop.Gaussian(5e-05, 5e-06),
                bounds=[2e-06, 10e-05],
            ),
        )
        experiment = pybop.Experiment(
            ["Discharge at 1C until 3.5 V (5 seconds period)"],
        )

        problem = pybop.DesignProblem(
            model, parameters, experiment=experiment, initial_state=initial_state
        )
        costs = [cost(problem) for cost in design_cost]

        return pybop.WeightedCost(*costs, weights=[1.0, 0.1])

    @pytest.mark.integration
    def test_design_costs(self, weighted_design_cost):
        cost = weighted_design_cost
        optim = pybop.CuckooSearch(
            cost,
            max_iterations=15,
            allow_infeasible_solutions=False,
        )
        initial_values = optim.parameters.initial_value()
        initial_cost = optim.cost(initial_values)
        x, final_cost = optim.run()

        # Assertions
        assert initial_cost < final_cost
        for i, _ in enumerate(x):
            assert x[i] > initial_values[i]

    def get_data(self, model, init_soc):
        initial_state = {"Initial SoC": init_soc}
        experiment = pybop.Experiment(
            [
                (
                    "Discharge at 0.5C for 3 minutes (4 second period)",
                    "Charge at 0.5C for 3 minutes (4 second period)",
                ),
            ]
        )
        sim = model.predict(initial_state=initial_state, experiment=experiment)
        return sim
