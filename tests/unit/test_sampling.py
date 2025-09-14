import copy
import logging
import re
from unittest.mock import call, patch

import numpy as np
import pytest
from pints import ParallelEvaluator

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

    pytestmark = pytest.mark.unit

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
    def parameters(self):
        return pybop.Parameters(
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
        )

    @pytest.fixture
    def model(self):
        return pybop.lithium_ion.SPM()

    @pytest.fixture
    def log_posterior(self, model, parameters, dataset):
        problem = pybop.FittingProblem(
            model,
            parameters,
            dataset,
        )
        likelihood = pybop.GaussianLogLikelihoodKnownSigma(problem, sigma0=0.01)
        prior1 = pybop.Gaussian(0.7, 0.02)
        prior2 = pybop.Gaussian(0.6, 0.02)
        composed_prior = pybop.JointLogPrior(prior1, prior2)
        log_posterior = pybop.LogPosterior(likelihood, composed_prior)

        return log_posterior

    @pytest.fixture
    def x0(self):
        return [0.68, 0.58]

    @pytest.fixture
    def chains(self):
        return 3

    @pytest.fixture
    def multi_samplers(self):
        return pybop.DREAM | pybop.EmceeHammerMCMC | pybop.DifferentialEvolutionMCMC

    @pytest.fixture(
        params=[
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
        ]
    )
    def MCMC(self, request):
        return request.param

    def test_initialisation_and_run(
        self, log_posterior, x0, chains, MCMC, multi_samplers
    ):
        sampler = MCMC(
            log_pdf=log_posterior,
            chains=chains,
            x0=x0,
            max_iterations=1,
            verbose=True,
        )
        assert sampler._n_chains == chains
        assert sampler._log_pdf == log_posterior
        if isinstance(sampler, multi_samplers):
            np.testing.assert_allclose(sampler._samplers[0]._x0[0], x0)
        else:
            np.testing.assert_allclose(sampler._samplers[0]._x0, x0)

        # Test __setattr__
        sampler.some_attribute = 1
        assert sampler.some_attribute == 1
        sampler.verbose = True
        assert sampler.verbose is True

        # Run the sampler
        samples = sampler.run()
        assert samples is not None
        assert samples.shape == (chains, 1, 2)

    def test_effective_sample_size(self, log_posterior):
        chains = np.asarray([[[0, 0]]])
        summary = pybop.PosteriorSummary(chains)

        with pytest.raises(ValueError, match="At least two samples must be given."):
            summary.effective_sample_size()

        n_chains = 3
        sampler = pybop.HaarioBardenetACMC(
            log_pdf=log_posterior,
            chains=n_chains,
            max_iterations=3,
        )
        chains = sampler.run()
        summary = pybop.PosteriorSummary(chains)

        # Non mixed chains
        ess = summary.effective_sample_size()
        assert len(ess) == log_posterior.n_parameters * n_chains
        assert all(e > 0 for e in ess)  # ESS should be positive

        # Mixed chains
        ess = summary.effective_sample_size(mixed_chains=True)
        assert len(ess) == log_posterior.n_parameters
        assert all(e > 0 for e in ess)

    def test_single_parameter_sampling(self, model, dataset, MCMC, chains):
        parameters = pybop.Parameters(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.6, 0.2),
                bounds=[0.58, 0.62],
            )
        )
        problem = pybop.FittingProblem(
            model,
            parameters,
            dataset,
        )
        likelihood = pybop.GaussianLogLikelihoodKnownSigma(problem, sigma0=0.01)
        log_posterior = pybop.LogPosterior(likelihood)

        # Skip RelativisticMCMC as it requires > 1 parameter
        if issubclass(MCMC, RelativisticMCMC):
            return

        # Construct and run
        sampler = MCMC(
            log_pdf=log_posterior,
            chains=chains,
            max_iterations=3,
            verbose=True,
        )
        result = sampler.run()
        summary = pybop.PosteriorSummary(result)
        autocorr = summary.autocorrelation(result[0, :, 0])
        assert autocorr.shape == (result[0, :, 0].shape[0] - 2,)

    def test_multi_log_pdf(self, log_posterior, x0, chains):
        multi_log_posterior = [log_posterior, log_posterior, log_posterior]
        sampler = HamiltonianMCMC(
            log_pdf=multi_log_posterior,
            chains=chains,
            x0=x0,
            max_iterations=1,
        )
        assert sampler._n_chains == chains
        assert sampler._log_pdf == multi_log_posterior

        # Run the sampler
        samples = sampler.run()
        assert samples is not None
        assert samples.shape == (chains, 1, 2)

        # Test incorrect multi log pdf
        incorrect_multi_log_posterior = [log_posterior, log_posterior, chains]
        with pytest.raises(
            ValueError, match="All log pdf's must be instances of BaseCost"
        ):
            pybop.HaarioBardenetACMC(
                log_pdf=incorrect_multi_log_posterior,
                chains=chains,
                x0=x0,
                max_iterations=1,
            )

        # Test incorrect number of parameters
        likelihood_copy = copy.copy(log_posterior.likelihood)
        likelihood_copy.parameters = pybop.Parameters(
            likelihood_copy.parameters[
                "Positive electrode active material volume fraction"
            ]
        )
        new_multi_log_posterior = pybop.LogPosterior(likelihood_copy)

        with pytest.raises(
            ValueError, match="All log pdf's must have the same number of parameters"
        ):
            pybop.HaarioBardenetACMC(
                log_pdf=[log_posterior, log_posterior, new_multi_log_posterior],
                chains=chains,
                x0=x0,
                max_iterations=1,
            )

    def test_invalid_initialisation(self, log_posterior, x0):
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

        with pytest.raises(
            ValueError, match="x0 must have the same number of parameters as log_pdf"
        ):
            AdaptiveCovarianceMCMC(
                log_pdf=[log_posterior, log_posterior, log_posterior],
                chains=3,
                x0=[0.4, 0.4, 0.4, 0.4],
            )

    # SingleChain & MultiChain Sampler
    @pytest.mark.parametrize(
        "sampler",
        [
            AdaptiveCovarianceMCMC,
            DifferentialEvolutionMCMC,
        ],
    )
    def test_no_chains_in_memory(self, log_posterior, x0, chains, sampler):
        sampler = sampler(
            log_pdf=log_posterior,
            chains=chains,
            x0=x0,
            max_iterations=1,
            chains_in_memory=False,
        )
        assert sampler._chains_in_memory is False

        # Run the sampler
        samples = sampler.run()
        assert sampler._samples is not None
        assert samples is None

    @patch("logging.basicConfig")
    @patch("logging.info")
    def test_initialise_logging(
        self, mock_info, mock_basicConfig, log_posterior, x0, chains
    ):
        sampler = AdaptiveCovarianceMCMC(
            log_pdf=log_posterior,
            chains=chains,
            x0=x0,
            parallel=True,
            evaluation_files=["eval1.txt", "eval2.txt"],
            chain_files=["chain1.txt", "chain2.txt"],
        )

        # Set parallel workers
        sampler.set_parallel(parallel=2)
        sampler._initialise_logging()

        # Check if basicConfig was called with correct arguments
        mock_basicConfig.assert_called_once_with(
            format="%(message)s", level=logging.INFO
        )

        # Check if correct messages were called
        expected_calls = [
            call("Using Haario-Bardenet adaptive covariance MCMC"),
            call("Generating 3 chains."),
            call("Running in parallel with 2 worker processes."),
            call("Writing chains to chain1.txt etc."),
            call("Writing evaluations to eval1.txt etc."),
        ]
        mock_info.assert_has_calls(expected_calls, any_order=False)

        # Test when _log_to_screen is False
        sampler._log_to_screen = False
        sampler._initialise_logging()
        assert mock_info.call_count == len(expected_calls)  # No additional calls

    def test_check_stopping_criteria(self, log_posterior, x0, chains):
        sampler = AdaptiveCovarianceMCMC(
            log_pdf=log_posterior,
            chains=chains,
            x0=x0,
        )
        # Set stopping criteria
        sampler.set_max_iterations(10)
        assert sampler._max_iterations == 10

        # Remove stopping criteria
        sampler._max_iterations = None
        with pytest.raises(
            ValueError, match="At least one stopping criterion must be set."
        ):
            sampler._check_stopping_criteria()

        # Incorrect stopping criteria
        with pytest.raises(
            ValueError, match="Number of iterations must be greater than 0"
        ):
            sampler.set_max_iterations(-1)

    def test_set_parallel(self, log_posterior, x0, chains):
        sampler = AdaptiveCovarianceMCMC(
            log_pdf=log_posterior,
            chains=chains,
            x0=x0,
        )

        # Disable parallelism
        sampler.set_parallel(False)
        assert sampler._parallel is False
        assert sampler._n_workers == 1

        # Enable parallelism
        sampler.set_parallel(True)
        assert sampler._parallel is True

        # Enable parallelism with number of workers
        sampler.set_parallel(2)
        assert sampler._parallel is True
        assert sampler._n_workers == 2

        # Test evaluator construction
        sampler.set_parallel(2)
        evaluator = sampler._create_evaluator()
        assert isinstance(evaluator, ParallelEvaluator)

    def test_base_sampler(self, log_posterior, x0):
        sampler = pybop.BaseSampler(log_posterior, x0, chains=1, cov0=0.1)
        with pytest.raises(NotImplementedError):
            sampler.run()

        with pytest.raises(
            ValueError,
            match=re.escape("log_pdf must be a LogPosterior or List[LogPosterior]"),
        ):
            pybop.BaseSampler(pybop.WeightedCost(log_posterior), x0, chains=1, cov0=0.1)

    def test_base_chain_processor(self, log_posterior, x0):
        sampler = pybop.MALAMCMC(log_posterior, chains=1)
        chain_processor = pybop.ChainProcessor(sampler)
        with pytest.raises(NotImplementedError):
            chain_processor.process_chain()

        with pytest.raises(NotImplementedError):
            chain_processor._extract_log_pdf(log_posterior, 0)
