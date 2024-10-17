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
            a_min=0.0,
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
                "R0 [Ohm]", prior=pybop.Uniform(1e-2, 8e-2), bounds=[1e-2, 8e-2]
            ),
            pybop.Parameter(
                "R1 [Ohm]", prior=pybop.Uniform(1e-2, 8e-2), bounds=[1e-2, 8e-2]
            ),
        )

    @pytest.fixture(params=[0.5])
    def init_soc(self, request):
        return request.param

    def noise(self, sigma, values):
        return np.random.normal(0, sigma, values)

    @pytest.fixture
    def likelihood(self, model, parameters, init_soc):
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
        return pybop.GaussianLogLikelihoodKnownSigma(problem, sigma0=0.0075)

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
    def test_sampling_thevenin(self, sampler, likelihood):
        posterior = pybop.LogPosterior(likelihood)

        # set common args
        common_args = {
            "log_pdf": posterior,
            "chains": 1,
            "warm_up": 250,
            "max_iterations": 500,
            "cov0": [3e-4, 3e-4],
        }
        if sampler in self.fast_samplers:
            common_args["warm_up"] = 600
            common_args["max_iterations"] = 1200

        # construct and run
        sampler = sampler(**common_args)
        if isinstance(sampler, SliceRankShrinkingMCMC):
            sampler._samplers[0].set_hyper_parameters([1e-3])
        results = sampler.run()

        # Test PosteriorSummary
        summary = pybop.PosteriorSummary(results)
        ess = summary.effective_sample_size()
        np.testing.assert_array_less(0, ess)

        # Assert both final sample and posterior mean
        x = np.mean(results, axis=1)
        for i in range(len(x)):
            np.testing.assert_allclose(x[i], self.ground_truth, atol=1.5e-2)
            np.testing.assert_allclose(results[i][-1], self.ground_truth, atol=1e-2)

    def get_data(self, model, init_soc):
        initial_state = {"Initial SoC": init_soc}
        experiment = pybop.Experiment(
            [
                (
                    "Discharge at 0.5C for 2 minutes (4 second period)",
                    "Rest for 1 minute (4 second period)",
                ),
            ]
        )
        sim = model.predict(initial_state=initial_state, experiment=experiment)
        return sim
