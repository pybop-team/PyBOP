import numpy as np
from pybamm import IDAKLUSolver
import pytest

import pybop


class Test_Jax_Parameterisation:
    """
    A class to test the model parameterisation methods.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.sigma0 = 0.002
        self.ground_truth = np.clip(
            np.asarray([0.55, 0.55]) + np.random.normal(loc=0.0, scale=0.05, size=2),
            a_min=0.425,
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
                prior=pybop.Uniform(0.3, 0.9),
                initial_value=pybop.Uniform(0.4, 0.75).rvs()[0],
                bounds=[0.3, 0.8],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Uniform(0.3, 0.9),
                initial_value=pybop.Uniform(0.4, 0.75).rvs()[0],
                # no bounds
            ),
        )

    @pytest.fixture(params=[0.4, 0.7])
    def init_soc(self, request):
        return request.param

    @pytest.fixture(
        params=[
            pybop.JaxSumSquaredError,
            pybop.JaxLogNormalLikelihood,
            pybop.JaxGaussianLogLikelihoodKnownSigma,
        ]
    )
    def cost_cls(self, request):
        return request.param

    def noise(self, sigma, values):
        return np.random.normal(0, sigma, values)

    @pytest.fixture(
        params=[
            pybop.SciPyDifferentialEvolution,
            pybop.SimulatedAnnealing,
            pybop.CuckooSearch,
            pybop.NelderMead,
            pybop.IRPropMin,
            pybop.AdamW,
            pybop.CMAES,
            pybop.SNES,
            pybop.XNES,
        ]
    )
    def optimiser(self, request):
        return request.param

    @pytest.fixture
    def optim(self, optimiser, model, parameters, cost_cls, init_soc):
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

        # Define the problem
        model.solver = IDAKLUSolver()
        problem = pybop.FittingProblem(model, parameters, dataset)

        # Construct the cost
        if cost_cls is pybop.JaxLogNormalLikelihood:
            cost = cost_cls(problem, sigma0=self.sigma0)
        elif cost_cls is pybop.JaxGaussianLogLikelihoodKnownSigma:
            cost = cost_cls(problem, sigma0=self.sigma0)
        else:
            cost = cost_cls(problem)

        max_unchanged_iter = 120
        sigma0 = 0.01
        if optimiser is pybop.SimulatedAnnealing:
            max_unchanged_iter = 450
            sigma0 = 0.05

        # Construct optimisation object
        common_args = {
            "cost": cost,
            "max_iterations": 450,
            "max_unchanged_iterations": max_unchanged_iter,
            "sigma0": sigma0,
        }

        if optimiser in [
            pybop.SciPyDifferentialEvolution,
            pybop.CuckooSearch,
        ]:
            common_args["bounds"] = [[0.375, 0.775], [0.375, 0.775]]
            if isinstance(cost, pybop.GaussianLogLikelihood):
                common_args["bounds"].extend([[0.0, 0.05]])

        # Set sigma0 and create optimiser
        optim = optimiser(**common_args)

        # Set Hypers
        if isinstance(optim, pybop.SimulatedAnnealing):
            optim.optimiser.cooling_rate = 0.95  # Cool quickly
        if isinstance(optim, pybop.CuckooSearch):
            optim.optimiser.pa = 0.55  # Increase abandon rate for limited iterations
        if isinstance(optim, pybop.AdamW):
            optim.optimiser.b1 = 0.9
            optim.optimiser.b2 = 0.9
        return optim

    @pytest.mark.integration
    def test_optimisers(self, optim):
        x0 = optim.parameters.initial_value()
        initial_cost = optim.cost(x0)
        results = optim.run()

        # Assertions
        if np.allclose(x0, self.ground_truth, atol=1e-5):
            raise AssertionError("Initial guess is too close to ground truth")

        if optim.minimising:
            assert initial_cost > results.final_cost
        else:
            assert initial_cost < results.final_cost

        np.testing.assert_allclose(results.x, self.ground_truth, atol=2e-2)

    def get_data(self, model, init_soc):
        initial_state = {"Initial SoC": init_soc}
        experiment = pybop.Experiment(
            [
                (
                    "Discharge at 0.5C for 12 minutes (10 second period)",
                    "Charge at 0.5C for 12 minutes (10 second period)",
                )
            ]
        )
        sim = model.predict(initial_state=initial_state, experiment=experiment)
        return sim
