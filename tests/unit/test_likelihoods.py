import numpy as np
import pybamm
import pytest
from scipy import stats

import pybop


class TestLikelihoods:
    """
    Class for likelihood unit tests
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def model_and_parameter_values(self, ground_truth):
        model = pybamm.lithium_ion.SPM()
        parameter_values = model.default_parameter_values
        parameter_values.update(
            {"Negative electrode active material volume fraction": ground_truth}
        )
        return model, parameter_values

    @pytest.fixture
    def ground_truth(self):
        return 0.52

    @pytest.fixture
    def parameters(self):
        return {
            "Negative electrode active material volume fraction": pybop.Parameter(
                distribution=pybop.Gaussian(0.5, 0.01),
                bounds=[0.375, 0.625],
            )
        }

    @pytest.fixture
    def experiment(self):
        return pybamm.Experiment(["Discharge at 1C for 1 minutes (5 second period)"])

    @pytest.fixture
    def dataset(self, model_and_parameter_values, experiment):
        model, parameter_values = model_and_parameter_values
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
    def simulator(self, model_and_parameter_values, parameters, dataset):
        model, parameter_values = model_and_parameter_values
        parameter_values.update(parameters)
        return pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )

    def test_base_likelihood(self, dataset):
        likelihood = pybop.LogLikelihood(dataset)
        with pytest.raises(NotImplementedError):
            likelihood(np.array([0.5]))

    def test_likelihood_check_sigma0(self, dataset):
        with pytest.raises(ValueError, match="Sigma0 must be positive"):
            pybop.GaussianLogLikelihoodKnownSigma(dataset, sigma0=None)

        likelihood = pybop.GaussianLogLikelihoodKnownSigma(dataset, 0.1)
        likelihood.set_sigma0(0.2)
        assert likelihood.sigma2 == 0.2**2.0

        with pytest.raises(
            ValueError,
            match=r"sigma0 must be either a scalar value",
        ):
            pybop.GaussianLogLikelihoodKnownSigma(dataset, sigma0=[0.2, 0.3])

    @pytest.mark.parametrize(
        "target",
        [["Voltage [V]"], ["Time [s]", "Voltage [V]"]],
    )
    def test_gaussian_log_likelihood_known_sigma(self, simulator, dataset, target):
        likelihood = pybop.GaussianLogLikelihoodKnownSigma(
            dataset, target=target, sigma0=np.array([0.01])
        )
        problem = pybop.Problem(simulator, likelihood)
        result = problem.evaluate([0.5]).values
        grad_result, grad_likelihood = problem.evaluate(
            [0.5], calculate_sensitivities=True
        ).get_values()
        assert isinstance(result[0], float)
        np.testing.assert_allclose(result, grad_result, atol=1e-5)
        # Since 0.5 < ground_truth, the likelihood should be increasing
        assert grad_likelihood >= 0

    def test_gaussian_log_likelihood(self, simulator, dataset):
        likelihood = pybop.GaussianLogLikelihood(dataset, sigma0=0.01)
        problem = pybop.Problem(simulator, likelihood)
        result = problem.evaluate(np.array([0.8, 0.02])).values
        grad_result, grad_likelihood = problem.evaluate(
            np.array([0.8, 0.025]), calculate_sensitivities=True
        ).get_values()
        assert isinstance(result[0], float)
        np.testing.assert_allclose(result, grad_result, atol=1e-5)
        # Since 0.8 > ground_truth, the likelihood should be decreasing
        assert grad_likelihood[0][0] <= 0
        # Since sigma < 0.02, the likelihood should be decreasing
        assert grad_likelihood[0][1] <= 0

        # Test construction with sigma as a Parameter
        sigma = pybop.Parameter(stats.Uniform(a=0.4, b=0.6))
        likelihood = pybop.GaussianLogLikelihood(dataset, sigma0=sigma)

        # Test invalid sigma
        with pytest.raises(
            TypeError,
            match=r"Expected sigma0 to contain Parameter objects or numeric values.",
        ):
            likelihood = pybop.GaussianLogLikelihood(dataset, sigma0="Invalid string")
            pybop.Problem(simulator, likelihood)

    def test_gaussian_log_likelihood_dsigma_scale(self, dataset):
        likelihood = pybop.GaussianLogLikelihood(dataset, dsigma_scale=0.05)
        assert likelihood.dsigma_scale == 0.05
        likelihood.dsigma_scale = 1e3
        assert likelihood.dsigma_scale == 1e3

        # Test incorrect sigma scale
        with pytest.raises(ValueError):
            likelihood.dsigma_scale = -1e3

    def test_gaussian_log_likelihood_returns_negative_inf(self, simulator, dataset):
        likelihood = pybop.GaussianLogLikelihood(dataset)
        problem = pybop.Problem(simulator, likelihood)
        assert (
            problem.evaluate(np.array([0.01, 0.1])).values == -np.inf
        )  # parameter value too small
        assert (
            problem.evaluate(np.array([0.01, 0.1]), calculate_sensitivities=True).values
            == -np.inf
        )  # parameter value too small

    def test_gaussian_log_likelihood_known_sigma_returns_negative_inf(
        self, simulator, dataset
    ):
        likelihood = pybop.GaussianLogLikelihoodKnownSigma(
            dataset, sigma0=np.array([0.2])
        )
        problem = pybop.Problem(simulator, likelihood)
        assert (
            problem.evaluate(np.array([0.01])).values == -np.inf
        )  # parameter value too small
        assert (
            problem.evaluate(np.array([0.01]), calculate_sensitivities=True).values
            == -np.inf
        )  # parameter value too small
