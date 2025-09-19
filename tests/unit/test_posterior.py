import numpy as np
import pybamm
import pytest
import scipy.stats as st

import pybop


class TestLogPosterior:
    """
    Class for log posterior unit tests
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def model(self):
        return pybamm.lithium_ion.SPM()

    @pytest.fixture
    def ground_truth(self):
        return 0.52

    @pytest.fixture
    def parameter_values(self, model, ground_truth):
        parameter_values = model.default_parameter_values
        parameter_values.update(
            {"Negative electrode active material volume fraction": ground_truth}
        )
        return parameter_values

    @pytest.fixture
    def parameter(self, ground_truth):
        return pybop.Parameter(
            "Negative electrode active material volume fraction",
            prior=pybop.Gaussian(0.5, 0.01),
            bounds=[0.375, 0.625],
            initial_value=ground_truth,
        )

    @pytest.fixture
    def experiment(self):
        return pybamm.Experiment(["Discharge at 1C for 1 minutes (5 second period)"])

    @pytest.fixture
    def dataset(self, model, parameter_values, experiment):
        solution = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        ).solve()
        return pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Terminal voltage [V]"].data,
            }
        )

    @pytest.fixture
    def one_signal_problem(self, model, parameter_values, parameter, dataset):
        simulator = pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            input_parameter_names=parameter.name,
            protocol=dataset,
        )
        return pybop.FittingProblem(simulator, parameter, dataset)

    @pytest.fixture
    def likelihood(self, one_signal_problem):
        return pybop.GaussianLogLikelihoodKnownSigma(one_signal_problem, sigma0=0.01)

    @pytest.fixture
    def prior(self):
        return pybop.Gaussian(0.5, 0.01)

    def test_log_posterior_construction(self, likelihood, prior):
        # Test log posterior construction
        posterior = pybop.LogPosterior(likelihood, prior)
        keys = list(likelihood.parameters.keys())

        assert posterior._log_likelihood == likelihood
        assert posterior._prior == prior
        assert posterior.parameters[keys[0]] == likelihood.parameters[keys[0]]
        assert posterior.has_separable_problem == likelihood.has_separable_problem

    def test_log_posterior_construction_no_prior(self, likelihood):
        # Test log posterior construction without prior
        posterior = pybop.LogPosterior(likelihood, None)

        assert posterior._prior is not None
        assert isinstance(posterior._prior, pybop.JointLogPrior)

        for i, p in enumerate(posterior._prior._priors):
            assert p == posterior._log_likelihood.problem.parameters.priors()[i]

    @pytest.fixture
    def posterior(self, likelihood, prior):
        return pybop.LogPosterior(likelihood, prior)

    def test_log_posterior(self, posterior):
        # Test log posterior
        x = np.array([0.50])
        assert np.allclose(posterior(x), 51.6033, atol=2e-2)

        # Test log posterior evaluateS1
        p, dp = posterior(x, calculate_grad=True)
        assert np.allclose(p, 51.6033, atol=2e-2)
        assert np.allclose(dp, 0.4266, atol=2e-2)

        # Get log likelihood and log prior
        likelihood = posterior.likelihood
        prior = posterior.prior

        assert likelihood == posterior._log_likelihood
        assert prior == posterior._prior

    @pytest.fixture
    def posterior_uniform_prior(self, likelihood):
        return pybop.LogPosterior(likelihood, pybop.Uniform(0.45, 0.55))

    def test_log_posterior_inf(self, posterior_uniform_prior):
        # Test prior np.inf
        assert not np.isfinite(posterior_uniform_prior([1]))
        assert not np.isfinite(posterior_uniform_prior([1], calculate_grad=True)[0])

    def test_non_logpdfS1_prior(self, likelihood):
        # Scipy distribution
        prior = st.norm(loc=0.8, scale=0.01)
        posterior = pybop.LogPosterior(likelihood, log_prior=prior)
        l, dl = likelihood([0.6], calculate_grad=True)
        p, dp = posterior([0.6], calculate_grad=True)

        # Assert to pybop.Gaussian
        p2, dp2 = pybop.Gaussian(mean=0.8, sigma=0.01).logpdfS1(0.6)
        np.testing.assert_allclose(p - l, p2, atol=2e-3)
        np.testing.assert_allclose(dp - dl, dp2, atol=2e-3)
