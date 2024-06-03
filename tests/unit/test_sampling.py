from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import pybop
from pybop import (
    DREAM,
    MALAMCMC,
    NUTS,
    AdaptiveCovarianceMCMC,
    DifferentialEvolutionMCMC,
    DramACMC,
    EmceeHammerMCMC,
    HaarioACMC,
    HaarioBardenetACMC,
    HamiltonianMCMC,
    MetropolisRandomWalkMCMC,
    MonomialGammaHamiltonianMCMC,
    PopulationMCMC,
    RaoBlackwellACMC,
    RelativisticMCMC,
    SliceDoublingMCMC,
    SliceRankShrinkingMCMC,
    SliceStepoutMCMC,
)


class TestPintsSamplers:
    """
    Class for testing the Pints-based MCMC Samplers
    """

    @pytest.fixture
    def dataset(self):
        return pybop.Dataset(
            {
                "Time [s]": np.linspace(0, 360, 10),
                "Current function [A]": np.zeros(10),
                "Voltage [V]": np.ones(10),
            }
        )

    @pytest.fixture
    def two_parameters(self):
        return [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.6, 0.2),
                bounds=[0.58, 0.62],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.55, 0.05),
                bounds=[0.53, 0.57],
            ),
        ]

    @pytest.fixture
    def model(self):
        return pybop.lithium_ion.SPM()

    @pytest.fixture
    def cost(self, model, one_parameter, dataset):
        problem = pybop.FittingProblem(
            model,
            one_parameter,
            dataset,
        )
        return pybop.SumSquaredError(problem)

    @pytest.fixture
    def log_posterior(self, model, two_parameters, dataset):
        problem = pybop.FittingProblem(
            model,
            two_parameters,
            dataset,
        )
        likelihood = pybop.GaussianLogLikelihoodKnownSigma(problem, sigma=[0.02, 0.02])
        prior1 = pybop.GaussianLogPrior(0.7, 0.02)
        prior2 = pybop.GaussianLogPrior(0.6, 0.02)
        composed_prior = pybop.ComposedLogPrior(prior1, prior2)
        log_posterior = pybop.LogPosterior(likelihood, composed_prior)

        return log_posterior

    @pytest.fixture
    def x0(self):
        return [[0.68, 0.58], [0.68, 0.58], [0.68, 0.58]]

    @pytest.fixture
    def chains(self):
        return 3

    @pytest.mark.parametrize(
        "MCMC",
        [
            NUTS,
            DREAM,
            AdaptiveCovarianceMCMC,
            DifferentialEvolutionMCMC,
            DramACMC,
            EmceeHammerMCMC,
            HaarioACMC,
            HaarioBardenetACMC,
            HamiltonianMCMC,
            MALAMCMC,
            MetropolisRandomWalkMCMC,
            MonomialGammaHamiltonianMCMC,
            PopulationMCMC,
            RaoBlackwellACMC,
            RelativisticMCMC,
            SliceDoublingMCMC,
            SliceRankShrinkingMCMC,
            SliceStepoutMCMC,
        ],
    )
    @pytest.mark.unit
    def test_initialization_and_run(self, log_posterior, x0, chains, MCMC):
        sampler = MCMC(log_pdf=log_posterior, chains=chains, x0=x0, max_iterations=1)
        assert sampler._n_chains == chains
        assert sampler._log_pdf == log_posterior
        assert (sampler._samplers[0]._x0 == x0[0]).all()

        # Run the sampler
        samples = sampler.run()
        assert samples is not None
        assert samples.shape == (chains, 1, 2)

    @pytest.mark.unit
    def test_invalid_initialization(self, log_posterior, x0):
        with pytest.raises(ValueError, match="Number of chains must be greater than 0"):
            AdaptiveCovarianceMCMC(
                log_pdf=log_posterior,
                chains=0,
                x0=x0,
            )

        with pytest.raises(
            ValueError, match="Number of log pdf's must match number of chains"
        ):
            AdaptiveCovarianceMCMC(
                log_pdf=[log_posterior, log_posterior, log_posterior],
                chains=2,
                x0=x0,
            )

    @pytest.mark.unit
    def test_apply_transformation(self, log_posterior, x0, chains):
        sampler = AdaptiveCovarianceMCMC(
            log_pdf=log_posterior, chains=chains, x0=x0, transformation=MagicMock()
        )

        with patch.object(sampler, "_apply_transformation") as mock_method:
            sampler._apply_transformation(sampler._transformation)
            mock_method.assert_called_once_with(sampler._transformation)

    @pytest.mark.unit
    def test_logging_initialisation(self, log_posterior, x0, chains):
        sampler = AdaptiveCovarianceMCMC(
            log_pdf=log_posterior,
            chains=chains,
            x0=x0,
        )

        with patch("logging.basicConfig"), patch("logging.info") as mock_info:
            sampler._initialise_logging()
            assert mock_info.call_count > 0

    @pytest.mark.unit
    def test_check_stopping_criteria(self, log_posterior, x0, chains):
        sampler = AdaptiveCovarianceMCMC(
            log_pdf=log_posterior, chains=chains, x0=x0, max_iterations=10
        )

        sampler._max_iterations = None
        with pytest.raises(
            ValueError, match="At least one stopping criterion must be set."
        ):
            sampler._check_stopping_criteria()
