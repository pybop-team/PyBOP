import numpy as np
import pybamm
import pytest

import pybop
from pybop import (
    DREAM,
    DifferentialEvolutionMCMC,
    HaarioACMC,
    HaarioBardenetACMC,
    MetropolisRandomWalkMCMC,
    PopulationMCMC,
)


class Test_Sampling_SPM:
    """
    A class to test the MCMC samplers on a physics-based model.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ground_truth = np.clip(
            np.asarray([0.55, 0.55, 3e-3])
            + np.random.normal(0, [5e-2, 5e-2, 1e-4], size=3),
            [0.4, 0.4, 1e-5],
            [0.7, 0.7, 1e-2],
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
        solver = pybamm.IDAKLUSolver()
        return pybop.lithium_ion.SPM(parameter_set=parameter_set, solver=solver)

    @pytest.fixture
    def parameters(self):
        return pybop.Parameters(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Uniform(0.4, 0.7),
                bounds=[0.375, 0.725],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Uniform(0.4, 0.7),
                # no bounds
            ),
        )

    @pytest.fixture(params=[0.5])
    def init_soc(self, request):
        return request.param

    def noise(self, sigma, values):
        return np.random.normal(0, sigma, values)

    @pytest.fixture
    def log_posterior(self, model, parameters, init_soc):
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

        # Define the posterior to optimise
        problem = pybop.FittingProblem(model, parameters, dataset)
        likelihood = pybop.GaussianLogLikelihood(problem, sigma0=0.002 * 1.2)
        return pybop.LogPosterior(likelihood)

    @pytest.fixture
    def map_estimate(self, log_posterior):
        common_args = {
            "max_iterations": 100,
            "max_unchanged_iterations": 35,
            "absolute_tolerance": 1e-7,
        }
        optim = pybop.CMAES(log_posterior, **common_args)
        results = optim.run()

        return results.x

    @pytest.mark.parametrize(
        "quick_sampler",
        [
            DREAM,
            DifferentialEvolutionMCMC,
            HaarioACMC,
            HaarioBardenetACMC,
            MetropolisRandomWalkMCMC,
            PopulationMCMC,
        ],
    )
    @pytest.mark.integration
    def test_sampling_spm(self, quick_sampler, log_posterior, map_estimate):
        x0 = np.clip(
            map_estimate + np.random.normal(0, [5e-3, 5e-3, 1e-4], size=3),
            [0.4, 0.4, 1e-5],
            [0.75, 0.75, 5e-2],
        )
        # set common args
        common_args = {
            "log_pdf": log_posterior,
            "chains": 3,
            "x0": x0,
            "warm_up": 150,
            "max_iterations": 550,
        }

        # construct and run
        sampler = quick_sampler(**common_args)
        chains = sampler.run()

        # Assert both final sample and posterior mean
        x = np.mean(chains, axis=1)
        for i in range(len(x)):
            np.testing.assert_allclose(x[i], self.ground_truth, atol=1.5e-2)

    def get_data(self, model, init_soc):
        initial_state = {"Initial SoC": init_soc}
        experiment = pybop.Experiment(
            [
                (
                    "Discharge at 0.5C for 4 minutes (12 second period)",
                    "Charge at 0.5C for 4 minutes (12 second period)",
                ),
            ]
        )
        sim = model.predict(initial_state=initial_state, experiment=experiment)
        return sim
