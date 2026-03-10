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
from pybop.samplers.base_sampler import SamplingResult


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
                "Current [A]": np.zeros(10),
                "Voltage [V]": np.ones(10),
            }
        )

    @pytest.fixture
    def parameters(self):
        return {
            "Negative electrode active material volume fraction": pybop.Parameter(
                distribution=pybop.Gaussian(0.7, 0.02, truncated_at=[0.58, 0.62])
            ),
            "Positive electrode active material volume fraction": pybop.Parameter(
                distribution=pybop.Gaussian(0.6, 0.02, truncated_at=[0.53, 0.57])
            ),
        }

    @pytest.fixture
    def model(self):
        return pybamm.lithium_ion.SPM()

    @pytest.fixture
    def log_pdf(self, model, parameters, dataset):
        parameter_values = model.default_parameter_values
        parameter_values.update(parameters)
        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )
        likelihood = pybop.GaussianLogLikelihoodKnownSigma(dataset, sigma=0.01)
        return pybop.LogPosterior(simulator, likelihood)

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

    def test_initialisation_and_run(self, log_pdf, n_chains, MCMC, multi_samplers):
        options = pybop.PintsSamplerOptions(
            n_chains=n_chains,
            max_iterations=1,
            verbose=True,
        )
        sampler = MCMC(log_pdf=log_pdf, options=options)
        assert sampler.options.n_chains == n_chains
        assert sampler._log_pdf == log_pdf
        mean0 = log_pdf.parameters.get_mean()
        if isinstance(sampler, multi_samplers):
            np.testing.assert_allclose(sampler._samplers[0]._x0[0], mean0)
        else:
            np.testing.assert_allclose(sampler._samplers[0]._x0, mean0)

        # Test __setattr__
        sampler.some_attribute = 1
        assert sampler.some_attribute == 1
        sampler.verbose = True
        assert sampler.verbose is True

        # Run the sampler
        result = sampler.run()
        assert result.chains is not None
        assert result.chains.shape == (n_chains, 1, 2)

    def test_effective_sample_size(self, log_pdf, MCMC):
        if MCMC is NUTS:
            # Test sample size error only once
            logger = pybop.Logger(minimising=True)
            logger.iteration = 1
            logger.extend_log(
                x_search=[np.asarray([1e-3])], x_model=[np.asarray([1e-3])], cost=[0.1]
            )
            sampler = MCMC(log_pdf)
            sampler._logger = logger
            result = SamplingResult(
                sampler=sampler,
                method_name="Test name",
                chains=np.asarray([[[0, 0]]]),
                time=0.1,
                message="Test message",
            )

            with pytest.raises(ValueError, match="At least two samples must be given."):
                result.effective_sample_size()

        n_chains = 3
        options = pybop.PintsSamplerOptions(n_chains=n_chains, max_iterations=3)
        sampler = pybop.HaarioBardenetACMC(log_pdf=log_pdf, options=options)
        result = sampler.run()

        # Non mixed chains
        ess = result.effective_sample_size()
        assert len(ess) == log_pdf.n_parameters * n_chains
        assert all(e > 0 for e in ess)  # ESS should be positive

        # Mixed chains
        ess = result.effective_sample_size(mixed_chains=True)
        assert len(ess) == log_pdf.n_parameters
        assert all(e > 0 for e in ess)

    def test_single_parameter_sampling(self, model, dataset, MCMC, n_chains):
        parameter_values = model.default_parameter_values
        parameter_values.update(
            {
                "Negative electrode active material volume fraction": pybop.Parameter(
                    distribution=pybop.Gaussian(0.6, 0.2, truncated_at=[0.58, 0.62])
                )
            }
        )
        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )
        likelihood = pybop.GaussianLogLikelihoodKnownSigma(dataset, sigma=0.01)
        log_pdf = pybop.LogPosterior(simulator, likelihood)

        # Skip RelativisticMCMC as it requires > 1 parameter
        if issubclass(MCMC, RelativisticMCMC):
            return

        # Construct and run
        options = pybop.PintsSamplerOptions(
            n_chains=n_chains,
            max_iterations=3,
            verbose=True,
        )
        sampler = MCMC(log_pdf=log_pdf, options=options)
        result = sampler.run()
        summary = result.get_summary_statistics()
        assert isinstance(summary, dict)

        autocorr = result.autocorrelation(result.chains[0, :, 0])
        assert autocorr.shape == (result.chains[0, :, 0].shape[0] - 2,)

    def test_invalid_initialisation(self, log_pdf):
        with pytest.raises(ValueError, match="Number of chains must be greater than 0"):
            options = pybop.PintsSamplerOptions(n_chains=0)
            HaarioBardenetACMC(log_pdf=log_pdf, options=options)

    # SingleChain & MultiChain Sampler
    @pytest.mark.parametrize(
        "sampler",
        [
            HaarioBardenetACMC,
            DifferentialEvolutionMCMC,
        ],
    )
    def test_no_chains_in_memory(self, log_pdf, n_chains, sampler):
        options = sampler.default_options()
        options.n_chains = n_chains
        options.max_iterations = 1
        options.chains_in_memory = False
        sampler = sampler(log_pdf, options=options)

        # Run the sampler
        samples = sampler.run()
        assert samples.shape == (
            0,
            options.max_iterations,
            len(log_pdf.parameters),
        )

    @patch("logging.basicConfig")
    @patch("logging.info")
    def test_initialise_logging(self, mock_info, mock_basicConfig, log_pdf, n_chains):
        options = HaarioBardenetACMC.default_options()
        options.n_chains = n_chains
        options.evaluation_files = ["eval1.txt", "eval2.txt"]
        options.chain_files = ["chain1.txt", "chain2.txt"]
        sampler = HaarioBardenetACMC(log_pdf, options=options)
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

    def test_check_stopping_criteria(self, log_pdf, n_chains):
        options = pybop.PintsSamplerOptions(n_chains=n_chains)
        sampler = HaarioBardenetACMC(log_pdf=log_pdf, options=options)
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

    def test_base_sampler(self, log_pdf):
        options = pybop.SamplerOptions(n_chains=1)
        sampler = pybop.BaseSampler(log_pdf=log_pdf, options=options)
        with pytest.raises(NotImplementedError):
            sampler.run()

        # test default covariance
        sampler._cov0 = None
        sampler._validate_covariance_matrix()
        np.testing.assert_allclose(sampler.cov0, 0.05 * np.eye(2))

        # throws error for negative entry in covariance
        with pytest.raises(ValueError, match="Covariance values must be nonnegative."):
            sampler._cov0 = np.asarray([[-1.0, 0], [0, 1.0]])
            sampler._validate_covariance_matrix()

    def test_base_chain_processor(self, log_pdf):
        options = pybop.PintsSamplerOptions(n_chains=1)
        sampler = pybop.MALAMCMC(log_pdf=log_pdf, options=options)
        chain_processor = pybop.ChainProcessor(sampler)
        with pytest.raises(NotImplementedError):
            chain_processor.process_chain()

        with pytest.raises(NotImplementedError):
            chain_processor._extract_log_pdf(log_pdf, 0)

    def compare_result_data(self, result1, result2):
        assert result1.method_name == result2.method_name
        assert result1.n_runs == result2.n_runs
        assert result1._best_run == result2._best_run
        np.testing.assert_array_equal(result1._x, result2._x)
        np.testing.assert_array_equal(result1._x_model, result2._x_model)
        np.testing.assert_array_equal(result1._x0, result2._x0)
        np.testing.assert_array_equal(result1._best_cost, result2._best_cost)
        np.testing.assert_array_equal(result1._cost, result2._cost)
        np.testing.assert_array_equal(
            result1._cost_convergence, result2._cost_convergence
        )
        np.testing.assert_array_equal(result1.initial_cost, result2.initial_cost)
        np.testing.assert_array_equal(result1._n_iterations, result2._n_iterations)
        np.testing.assert_array_equal(
            result1._iteration_number, result2._iteration_number
        )
        np.testing.assert_array_equal(result1._n_evaluations, result2._n_evaluations)
        assert result1._message == result2._message
        np.testing.assert_array_equal(result1._scipy_result, result2._scipy_result)
        np.testing.assert_array_equal(result1._time, result2._time)

    def test_save(self, log_pdf, n_chains, MCMC, tmp_path):
        test_stub = tmp_path / "test"

        # Set up sampler
        options = pybop.PintsSamplerOptions(
            n_chains=n_chains,
            max_iterations=1,
            verbose=True,
        )
        sampler = MCMC(log_pdf=log_pdf, options=options)

        # Run the sampler
        result = sampler.run()

        # Re-define sampler to reconstruct result from data
        sampler2 = MCMC(log_pdf=log_pdf, options=options)

        for to_format in ["json", "matlab", "pickle"]:
            # Define filename
            if to_format == "matlab":
                filename = f"{test_stub}.mat"
            elif to_format == "json":
                filename = f"{test_stub}.json"
            else:
                filename = f"{test_stub}.pickle"
            # Test save result
            result.save_data(filename, to_format=to_format)

            # load result
            result_load = SamplingResult.load_data(filename, file_format=to_format)
            self.compare_result_data(result, result_load)
            assert sampler2.logger is None

        # test save whole result
        filename = f"{test_stub}.pickle"
        result.save(filename)
        result_load = SamplingResult.load(filename)
        self.compare_result_data(result, result_load)
        assert result.problem.parameters.names == result_load.problem.parameters.names
        np.testing.assert_array_equal(result.chains, result_load.chains)
