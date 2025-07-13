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
        return pybamm.lithium_ion.SPM()

    @pytest.fixture
    def log_posterior(self, model, parameters, dataset):
        builder = pybop.Pybamm()
        builder.set_simulation(model, build_on_eval=False)
        builder.set_dataset(dataset)
        for p in parameters:
            builder.add_parameter(p)
        builder.add_cost(
            pybop.costs.pybamm.NegativeGaussianLogLikelihood(
                "Voltage [V]", "Voltage [V]", sigma=0.01
            )
        )
        return builder.build()

    @pytest.fixture
    def x0(self):
        return [0.68, 0.58]

    @pytest.fixture
    def chains(self):
        return 3

    @pytest.fixture
    def multi_samplers(self):
        return (pybop.DREAM, pybop.EmceeHammerMCMC, pybop.DifferentialEvolutionMCMC)

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

    def test_initialisation_and_run(self, log_posterior, chains, MCMC):
        options = MCMC.default_options()
        options.n_chains = chains
        options.max_iterations = 1
        options.verbose = True
        sampler = MCMC(
            log_posterior,
            options=options,
        )

        # Run the sampler
        samples = sampler.run()
        assert samples is not None
        assert samples.shape == (chains, 1, 2)

    def test_effective_sample_size(self, log_posterior):
        chains = np.asarray([[[0, 0]]])
        summary = pybop.PosteriorSummary(chains)

        with pytest.raises(ValueError, match="At least two samples must be given."):
            summary.effective_sample_size()

        options = pybop.HaarioBardenetACMC.default_options()
        options.n_chains = 3
        options.max_iterations = 3
        sampler = pybop.HaarioBardenetACMC(
            log_posterior,
            options=options,
        )
        chains = sampler.run()
        summary = pybop.PosteriorSummary(chains)

        # Non mixed chains
        ess = summary.effective_sample_size()
        assert len(ess) == len(log_posterior.params) * options.n_chains
        assert all(e > 0 for e in ess)  # ESS should be positive

        # Mixed chains
        ess = summary.effective_sample_size(mixed_chains=True)
        assert len(ess) == len(log_posterior.params)
        assert all(e > 0 for e in ess)

    def test_single_parameter_sampling(self, model, dataset, MCMC, chains):
        p = pybop.Parameter(
            "Negative electrode active material volume fraction",
            prior=pybop.Gaussian(0.6, 0.2),
            bounds=[0.58, 0.62],
        )
        builder = pybop.Pybamm()
        builder.set_simulation(model, build_on_eval=False)
        builder.set_dataset(dataset)
        builder.add_parameter(p)
        builder.add_cost(
            pybop.costs.pybamm.NegativeGaussianLogLikelihood(
                "Voltage [V]", "Voltage [V]", sigma=0.01
            )
        )
        problem = builder.build()

        # Skip RelativisticMCMC as it requires > 1 parameter
        if issubclass(MCMC, RelativisticMCMC):
            return

        # Construct and run
        options = MCMC.default_options()
        options.n_chains = chains
        options.max_iterations = 3
        options.verbose = True
        sampler = MCMC(
            problem,
            options=options,
        )
        result = sampler.run()
        summary = pybop.PosteriorSummary(result)
        autocorr = summary.autocorrelation(result[0, :, 0])
        assert autocorr.shape == (result[0, :, 0].shape[0] - 2,)

    def test_invalid_initialisation(self, log_posterior, x0):
        options = AdaptiveCovarianceMCMC.default_options()
        options.n_chains = 0
        with pytest.raises(ValueError, match="Number of chains must be greater than 0"):
            AdaptiveCovarianceMCMC(log_posterior, options=options)

    # SingleChain & MultiChain Sampler
    @pytest.mark.parametrize(
        "sampler",
        [
            AdaptiveCovarianceMCMC,
            DifferentialEvolutionMCMC,
        ],
    )
    def test_no_chains_in_memory(self, log_posterior, x0, chains, sampler):
        options = sampler.default_options()
        options.n_chains = chains
        options.max_iterations = 1
        options.chains_in_memory = False
        sampler = sampler(log_posterior, options=options)

        # Run the sampler
        samples = sampler.run()
        assert samples.shape == (0, options.max_iterations, len(log_posterior.params))

    @patch("logging.basicConfig")
    @patch("logging.info")
    def test_initialise_logging(
        self, mock_info, mock_basicConfig, log_posterior, x0, chains
    ):
        options = AdaptiveCovarianceMCMC.default_options()
        options.n_chains = chains
        options.evaluation_files = ["eval1.txt", "eval2.txt"]
        options.chain_files = ["chain1.txt", "chain2.txt"]
        options.parallel = True
        sampler = AdaptiveCovarianceMCMC(log_posterior, options=options)

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
        options = AdaptiveCovarianceMCMC.default_options()
        options.n_chains = chains
        sampler = AdaptiveCovarianceMCMC(log_posterior, options=options)

        # Set stopping criteria
        sampler.set_max_iterations(options.max_iterations)

        # Incorrect stopping criteria
        with pytest.raises(
            ValueError, match="Number of iterations must be greater than 0"
        ):
            sampler.set_max_iterations(-1)

    def test_set_parallel(self, log_posterior, x0, chains):
        options = AdaptiveCovarianceMCMC.default_options()
        options.n_chains = chains
        sampler = AdaptiveCovarianceMCMC(log_posterior, options=options)

        # Disable parallelism
        sampler.set_parallel(False)
        assert sampler.options.parallel is False
        assert sampler.options.n_workers == 1

        # Enable parallelism
        sampler.set_parallel(True)
        assert sampler.options.parallel is True

        # Enable parallelism with number of workers
        sampler.set_parallel(2)
        assert sampler.options.parallel is True
        assert sampler.options.n_workers == 2

    def test_base_chain_processor(self, log_posterior, x0):
        options = pybop.MALAMCMC.default_options()
        options.n_chains = 1
        sampler = pybop.MALAMCMC(log_posterior, options=options)
        chain_processor = pybop.ChainProcessor(sampler)
        with pytest.raises(NotImplementedError):
            chain_processor.process_chain()

        with pytest.raises(NotImplementedError):
            chain_processor._extract_log_pdf(log_posterior, 0)
