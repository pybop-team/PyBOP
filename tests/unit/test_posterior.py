import numpy as np
import pytest

import pybop


class TestLogPosterior:
    """
    Class for log posterior unit tests
    """

    @pytest.fixture
    def model(self):
        return pybop.lithium_ion.SPM()

    @pytest.fixture
    def ground_truth(self):
        return 0.52

    @pytest.fixture
    def parameters(self, ground_truth):
        return pybop.Parameter(
            "Negative electrode active material volume fraction",
            prior=pybop.Gaussian(0.5, 0.01),
            bounds=[0.375, 0.625],
            initial_value=ground_truth,
        )

    @pytest.fixture
    def experiment(self):
        return pybop.Experiment(
            [
                ("Discharge at 1C for 1 minutes (5 second period)"),
            ]
        )

    @pytest.fixture
    def dataset(self, model, experiment, ground_truth):
        model._parameter_set = model.pybamm_model.default_parameter_values
        model._parameter_set.update(
            {
                "Negative electrode active material volume fraction": ground_truth,
            }
        )
        solution = model.predict(experiment=experiment)
        return pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Terminal voltage [V]"].data,
            }
        )

    @pytest.fixture
    def one_signal_problem(self, model, parameters, dataset):
        return pybop.FittingProblem(model, parameters, dataset)

    @pytest.fixture
    def likelihood(self, one_signal_problem):
        return pybop.GaussianLogLikelihoodKnownSigma(one_signal_problem, sigma0=0.01)

    @pytest.fixture
    def prior(self):
        return pybop.Gaussian(0.5, 0.01)

    @pytest.mark.unit
    def test_log_posterior_construction(self, likelihood, prior):
        # Test log posterior construction
        posterior = pybop.LogPosterior(likelihood, prior)

        assert posterior._log_likelihood == likelihood
        assert posterior._prior == prior

        # Test log posterior construction without parameters
        likelihood.parameters.priors = None

        with pytest.raises(TypeError, match="'NoneType' object is not callable"):
            pybop.LogPosterior(likelihood, log_prior=None)

    @pytest.mark.unit
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

    @pytest.mark.unit
    def test_log_posterior(self, posterior):
        # Test log posterior
        x = np.array([0.50])
        assert np.allclose(posterior(x), 51.5236, atol=2e-2)

        # Test log posterior evaluateS1
        p, dp = posterior(x, calculate_grad=True)
        assert np.allclose(p, 51.5236, atol=2e-2)
        assert np.allclose(dp, 2.0, atol=2e-2)

        # Get log likelihood and log prior
        likelihood = posterior.likelihood
        prior = posterior.prior

        assert likelihood == posterior._log_likelihood
        assert prior == posterior._prior

    @pytest.fixture
    def posterior_uniform_prior(self, likelihood):
        return pybop.LogPosterior(likelihood, pybop.Uniform(0.45, 0.55))

    @pytest.mark.unit
    def test_log_posterior_inf(self, posterior_uniform_prior):
        # Test prior np.inf
        assert not np.isfinite(posterior_uniform_prior([1]))
        assert not np.isfinite(posterior_uniform_prior([1], calculate_grad=True)[0])