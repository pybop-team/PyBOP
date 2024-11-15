import numpy as np
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
        parameter_set = pybop.ParameterSet(
            json_path="examples/scripts/parameters/initial_ecm_parameters.json"
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
                prior=pybop.Uniform(1e-2, 8e-2),
                bounds=[1e-4, 1e-1],
            ),
            pybop.Parameter(
                "R1 [Ohm]",
                prior=pybop.Uniform(1e-2, 8e-2),
                bounds=[1e-4, 1e-1],
            ),
        )

    @pytest.fixture(params=[0.5])
    def init_soc(self, request):
        return request.param

    def noise(self, sigma, values):
        return np.random.normal(0, sigma, values)

    @pytest.fixture
    def posterior(self, model, parameters, init_soc):
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
        likelihood = pybop.GaussianLogLikelihoodKnownSigma(problem, sigma0=self.sigma0)
        return pybop.LogPosterior(likelihood)

    @pytest.fixture
    def map_estimate(self, posterior):
        common_args = {
            "max_iterations": 100,
            "max_unchanged_iterations": 35,
            "absolute_tolerance": 1e-7,
            "sigma0": [3e-4, 3e-4],
        }
        optim = pybop.CMAES(posterior, **common_args)
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
    @pytest.mark.integration
    def test_sampling_thevenin(self, sampler, posterior, map_estimate):
        x0 = np.clip(map_estimate + np.random.normal(0, 1e-3, size=2), 1e-4, 1e-1)
        common_args = {
            "log_pdf": posterior,
            "chains": 1,
            "warm_up": 550,
            "cov0": [2e-3, 2e-3],
            "max_iterations": 1000,
            "x0": x0,
        }

        # construct and run
        sampler = sampler(**common_args)
        if isinstance(sampler, SliceRankShrinkingMCMC):
            sampler._samplers[0].set_hyper_parameters([1e-3])
        chains = sampler.run()

        # Test PosteriorSummary
        summary = pybop.PosteriorSummary(chains)
        ess = summary.effective_sample_size()
        np.testing.assert_array_less(0, ess)
        if not isinstance(sampler, RelativisticMCMC):
            np.testing.assert_array_less(
                summary.rhat(), 1.5
            )  # Large rhat, to enable faster tests

        # Assert both final sample and posterior mean
        x = np.mean(chains, axis=1)
        for i in range(len(x)):
            np.testing.assert_allclose(x[i], self.ground_truth, atol=5e-3)
            np.testing.assert_allclose(chains[i][-1], self.ground_truth, atol=1e-2)

    def get_data(self, model, init_soc):
        initial_state = {"Initial SoC": init_soc}
        experiment = pybop.Experiment(
            [
                ("Discharge at 0.5C for 6 minutes (20 second period)",),
            ]
        )
        sim = model.predict(initial_state=initial_state, experiment=experiment)
        return sim
