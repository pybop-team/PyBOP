import json

import numpy as np
import pybamm
import pytest

import pybop
from pybop import (
    MALAMCMC,
    NUTS,
    DramACMC,
    HamiltonianMCMC,
    MonomialGammaHamiltonianMCMC,
    RaoBlackwellACMC,
    RelativisticMCMC,
    SliceDoublingMCMC,
    SliceRankShrinkingMCMC,
    SliceStepoutMCMC,
)


class TestSamplingThevenin:
    """
    A class to test a subset of samplers on the simple Thevenin Model.
    """

    pytestmark = pytest.mark.integration

    @pytest.fixture(autouse=True)
    def setup(self):
        self.sigma0 = 1e-3
        self.ground_truth = np.clip(
            np.asarray([0.05, 0.05]) + np.random.normal(loc=0.0, scale=0.01, size=2),
            a_min=1e-4,
            a_max=0.1,
        )
        self.fast_samplers = [
            MALAMCMC,
            RaoBlackwellACMC,
            SliceDoublingMCMC,
            SliceStepoutMCMC,
            DramACMC,
        ]

    @pytest.fixture
    def model(self):
        with open("examples/parameters/initial_ecm_parameters.json") as file:
            parameter_set = pybamm.ParameterValues(json.load(file))
        parameter_set.update(
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
                prior=pybop.Gaussian(5e-2, 5e-3),
                transformation=pybop.LogTransformation(),
                initial_value=pybop.Uniform(2e-3, 8e-2).rvs()[0],
                bounds=[1e-4, 1e-1],
            ),
            pybop.Parameter(
                "R1 [Ohm]",
                prior=pybop.Gaussian(5e-2, 5e-3),
                transformation=pybop.LogTransformation(),
                initial_value=pybop.Uniform(2e-3, 8e-2).rvs()[0],
                bounds=[1e-4, 1e-1],
            ),
        )

    @pytest.fixture(params=[0.5])
    def init_soc(self, request):
        return request.param

    def noisy(self, data, sigma):
        return data + np.random.normal(0, sigma, len(data))

    @pytest.fixture
    def posterior(self, model, parameters, init_soc):
        # Form dataset
        solution = self.get_data(model, init_soc)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": self.noisy(solution["Voltage [V]"].data, self.sigma0),
            }
        )

        # Define the cost to optimise
        problem = pybop.FittingProblem(model, parameters, dataset)
        likelihood = pybop.GaussianLogLikelihoodKnownSigma(problem, sigma0=self.sigma0)
        return pybop.LogPosterior(likelihood)

    @pytest.fixture
    def map_estimate(self, posterior):
        options = pybop.PintsOptions(
            max_iterations=80,
            sigma=[3e-4, 3e-4],
            verbose=True,
        )
        optim = pybop.CMAES(posterior, options=options)
        results = optim.run()

        return results.x

    # Parameterize the samplers
    @pytest.mark.parametrize(
        "sampler",
        [
            NUTS,
            HamiltonianMCMC,
            MonomialGammaHamiltonianMCMC,
            RelativisticMCMC,
            SliceRankShrinkingMCMC,
            MALAMCMC,
            RaoBlackwellACMC,
            SliceDoublingMCMC,
            SliceStepoutMCMC,
            DramACMC,
        ],
    )
    def test_sampling_thevenin(self, sampler, posterior, map_estimate):
        x0 = np.clip(map_estimate + np.random.normal(0, 5e-3, size=2), 1e-4, 1e-1)
        posterior.parameters.update(initial_values=x0)
        options = pybop.PintsSamplerOptions(
            n_chains=2,
            warm_up_iterations=50,
            cov=[6e-3, 6e-3],
            max_iterations=500 if sampler is SliceRankShrinkingMCMC else 350,
        )

        # construct and run
        sampler = sampler(log_pdf=posterior, options=options)
        if isinstance(sampler, SliceRankShrinkingMCMC):
            for i, _j in enumerate(sampler._samplers):
                sampler._samplers[i].set_hyper_parameters([1e-3])
        chains = sampler.run()

        # Test PosteriorSummary
        summary = pybop.PosteriorSummary(chains)
        ess = summary.effective_sample_size()
        np.testing.assert_array_less(0, ess)
        np.testing.assert_array_less(0, summary.rhat())

        # Assert both final sample and posterior mean
        x = np.mean(chains, axis=1)
        for i in range(len(x)):
            np.testing.assert_allclose(x[i], self.ground_truth, atol=5e-3)
            np.testing.assert_allclose(chains[i][-1], self.ground_truth, atol=1e-2)

    def get_data(self, model, init_soc):
        initial_state = {"Initial SoC": init_soc}
        experiment = pybamm.Experiment(
            ["Discharge at 0.5C for 3 minutes (20 second period)"]
        )
        return model.predict(initial_state=initial_state, experiment=experiment)
