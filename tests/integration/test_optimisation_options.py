import sys

import numpy as np
import pytest

import pybop


class TestOptimisation:
    """
    A class to run integration tests on the Optimisation class.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ground_truth = np.array([0.55, 0.55]) + np.random.normal(
            loc=0.0, scale=0.05, size=2
        )

    @pytest.fixture
    def model(self):
        parameter_set = pybop.ParameterSet.pybamm("Chen2020")
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

    @pytest.fixture(
        params=[
            pybop.GaussianLogLikelihoodKnownSigma,
            pybop.RootMeanSquaredError,
            pybop.SumSquaredError,
        ]
    )
    def cost_class(self, request):
        return request.param

    def noise(self, sigma, values):
        return np.random.normal(0, sigma, values)

    @pytest.fixture
    def spm_costs(self, model, parameters, cost_class):
        # Form dataset
        init_soc = 0.5
        solution = self.get_data(model, parameters, self.ground_truth, init_soc)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data
                + self.noise(0.002, len(solution["Time [s]"].data)),
            }
        )

        # Define the cost to optimise
        problem = pybop.FittingProblem(model, parameters, dataset, init_soc=init_soc)
        if cost_class in [pybop.GaussianLogLikelihoodKnownSigma]:
            return cost_class(problem, sigma=[0.03, 0.03])
        else:
            return cost_class(problem)

    @pytest.mark.parametrize(
        "f_guessed",
        [
            True,
            False,
        ],
    )
    @pytest.mark.integration
    def test_optimisation_f_guessed(self, f_guessed, spm_costs):
        x0 = spm_costs.x0
        # Test each optimiser
        optim = pybop.XNES(
            cost=spm_costs,
            sigma0=0.05,
            max_iterations=250,
            max_unchanged_iterations=35,
            absolute_tolerance=1e-5,
            use_f_guessed=f_guessed,
        )

        # Set parallelisation if not on Windows
        if sys.platform != "win32":
            optim.set_parallel(True)

        initial_cost = optim.cost(x0)
        x, final_cost = optim.run()

        # Assertions
        if not np.allclose(x0, self.ground_truth, atol=1e-5):
            if optim.minimising:
                assert initial_cost > final_cost
            else:
                assert initial_cost < final_cost
        np.testing.assert_allclose(x, self.ground_truth, atol=1.5e-2)

    def get_data(self, model, parameters, x, init_soc):
        model.parameters = parameters
        experiment = pybop.Experiment(
            [
                (
                    "Discharge at 0.5C for 3 minutes (5 second period)",
                    "Charge at 0.5C for 3 minutes (5 second period)",
                ),
            ]
            * 2
        )
        sim = model.predict(init_soc=init_soc, experiment=experiment, inputs=x)
        return sim
