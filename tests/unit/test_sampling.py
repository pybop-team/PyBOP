import logging
from unittest.mock import call, patch

import numpy as np
import pybamm
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
        return {
            "Negative electrode active material volume fraction": pybop.Parameter(
                prior=pybop.Gaussian(0.6, 0.2),
                bounds=[0.58, 0.62],
            ),
            "Positive electrode active material volume fraction": pybop.Parameter(
                prior=pybop.Gaussian(0.55, 0.05),
                bounds=[0.53, 0.57],
            ),
        }

    @pytest.fixture
    def model(self):
        return pybamm.lithium_ion.SPM()

    @pytest.fixture
    def posterior_problem(self, model, parameters, dataset):
        parameter_values = model.default_parameter_values
        parameter_values.update(parameters)
        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )
        likelihood = pybop.GaussianLogLikelihoodKnownSigma(dataset, sigma0=0.01)
        prior1 = pybop.Gaussian(0.7, 0.02)
        prior2 = pybop.Gaussian(0.6, 0.02)
        composed_prior = pybop.JointPrior(prior1, prior2)
        posterior = pybop.LogPosterior(likelihood, prior=composed_prior)
        return pybop.Problem(simulator, posterior)

    @pytest.fixture
    def n_chains(self):
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
        self, posterior_problem, n_chains, MCMC, multi_samplers
    ):
        options = pybop.PintsSamplerOptions(
            n_chains=n_chains,
            max_iterations=1,
            verbose=True,
        )
        sampler = MCMC(log_pdf=posterior_problem, options=options)
        assert sampler.options.n_chains == n_chains
        assert sampler._log_pdf == posterior_problem
        x0 = posterior_problem.parameters.get_initial_values()
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
        result = sampler.run()
        assert result.chains is not None
        assert result.chains.shape == (n_chains, 1, 2)

    def test_effective_sample_size(self, posterior_problem):
        chains = np.asarray([[[0, 0]]])
        summary = pybop.PosteriorSummary(chains)

        with pytest.raises(ValueError, match="At least two samples must be given."):
            summary.effective_sample_size()

        n_chains = 3
        options = pybop.PintsSamplerOptions(n_chains=n_chains, max_iterations=3)
        sampler = pybop.HaarioBardenetACMC(log_pdf=posterior_problem, options=options)
        result = sampler.run()
        summary = pybop.PosteriorSummary(result.chains)

        # Non mixed chains
        ess = summary.effective_sample_size()
        assert len(ess) == posterior_problem.n_parameters * n_chains
        assert all(e > 0 for e in ess)  # ESS should be positive

        # Mixed chains
        ess = summary.effective_sample_size(mixed_chains=True)
        assert len(ess) == posterior_problem.n_parameters
        assert all(e > 0 for e in ess)

    def test_single_parameter_sampling(self, model, dataset, MCMC, n_chains):
        parameter_values = model.default_parameter_values
        parameter_values.update(
            {
                "Negative electrode active material volume fraction": pybop.Parameter(
                    prior=pybop.Gaussian(0.6, 0.2),
                    bounds=[0.58, 0.62],
                )
            }
        )
        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )
        likelihood = pybop.GaussianLogLikelihoodKnownSigma(dataset, sigma0=0.01)
        posterior = pybop.LogPosterior(likelihood)
        posterior = pybop.Problem(simulator, posterior)

        # Skip RelativisticMCMC as it requires > 1 parameter
        if issubclass(MCMC, RelativisticMCMC):
            return

        # Construct and run
        options = pybop.PintsSamplerOptions(
            n_chains=n_chains,
            max_iterations=3,
            verbose=True,
        )
        sampler = MCMC(log_pdf=posterior, options=options)
        result = sampler.run()
        summary = pybop.PosteriorSummary(result.chains)
        autocorr = summary.autocorrelation(result.chains[0, :, 0])
        assert autocorr.shape == (result.chains[0, :, 0].shape[0] - 2,)

    def test_invalid_initialisation(self, posterior_problem):
        with pytest.raises(ValueError, match="Number of chains must be greater than 0"):
            options = pybop.PintsSamplerOptions(n_chains=0)
            AdaptiveCovarianceMCMC(log_pdf=posterior_problem, options=options)

    # SingleChain & MultiChain Sampler
    @pytest.mark.parametrize(
        "sampler",
        [
            AdaptiveCovarianceMCMC,
            DifferentialEvolutionMCMC,
        ],
    )
    def test_no_chains_in_memory(self, posterior_problem, n_chains, sampler):
        options = sampler.default_options()
        options.n_chains = n_chains
        options.max_iterations = 1
        options.chains_in_memory = False
        sampler = sampler(posterior_problem, options=options)

        # Run the sampler
        samples = sampler.run()
        assert samples.shape == (
            0,
            options.max_iterations,
            len(posterior_problem.parameters),
        )

    @patch("logging.basicConfig")
    @patch("logging.info")
    def test_initialise_logging(
        self, mock_info, mock_basicConfig, posterior_problem, n_chains
    ):
        options = AdaptiveCovarianceMCMC.default_options()
        options.n_chains = n_chains
        options.evaluation_files = ["eval1.txt", "eval2.txt"]
        options.chain_files = ["chain1.txt", "chain2.txt"]
        sampler = AdaptiveCovarianceMCMC(posterior_problem, options=options)
        sampler._initialise_logging()

        # Check if basicConfig was called with correct arguments
        mock_basicConfig.assert_called_once_with(
            format="%(message)s", level=logging.INFO
        )

        # Check if correct messages were called
        expected_calls = [
            call("Using Haario-Bardenet adaptive covariance MCMC"),
            call("Generating 3 chains."),
            call("Writing chains to chain1.txt etc."),
            call("Writing evaluations to eval1.txt etc."),
        ]
        mock_info.assert_has_calls(expected_calls, any_order=False)

        # Test when _log_to_screen is False
        sampler._log_to_screen = False
        sampler._initialise_logging()
        assert mock_info.call_count == len(expected_calls)  # No additional calls

    def test_check_stopping_criteria(self, posterior_problem, n_chains):
        options = pybop.PintsSamplerOptions(n_chains=n_chains)
        sampler = AdaptiveCovarianceMCMC(log_pdf=posterior_problem, options=options)
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

    def test_base_sampler(self, posterior_problem):
        options = pybop.SamplerOptions(n_chains=1, cov=0.1)
        sampler = pybop.BaseSampler(log_pdf=posterior_problem, options=options)
        with pytest.raises(NotImplementedError):
            sampler.run()

    def test_base_chain_processor(self, posterior_problem):
        options = pybop.PintsSamplerOptions(n_chains=1)
        sampler = pybop.MALAMCMC(log_pdf=posterior_problem, options=options)
        chain_processor = pybop.ChainProcessor(sampler)
        with pytest.raises(NotImplementedError):
            chain_processor.process_chain()

        with pytest.raises(NotImplementedError):
            chain_processor._extract_log_pdf(posterior_problem, 0)
