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
        return {
            "Negative electrode active material volume fraction": pybop.Parameter(
                distribution=pybop.Gaussian(
                    0.5,
                    0.01,
                    truncated_at=[0.375, 0.625],
                ),
                initial_value=ground_truth,
            )
        }

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
    def simulator(self, model, parameter_values, parameter, dataset):
        parameter_values.update(parameter)
        return pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )

    @pytest.fixture
    def likelihood(self, dataset):
        return pybop.GaussianLogLikelihoodKnownSigma(dataset, sigma0=0.01)

    @pytest.fixture
    def prior(self):
        return pybop.Gaussian(0.5, 0.01)

    def test_log_posterior_construction(self, simulator, parameter, likelihood, prior):
        # Test log posterior construction
        posterior = pybop.LogPosterior(likelihood, prior=prior)
        problem = pybop.Problem(simulator, posterior)

        assert problem._cost == posterior
        assert problem._cost.log_likelihood == likelihood
        assert problem._cost.prior == prior
        key = "Negative electrode active material volume fraction"
        assert problem.parameters[key] is parameter[key]
        assert problem._cost.parameters is problem.parameters

        # Test construction with Parameter
        posterior = pybop.LogPosterior(
            likelihood, prior=pybop.Parameter(distribution=prior)
        )
        problem = pybop.Problem(
            simulator, posterior
        )  # uses posterior.set_joint_prior()
        assert posterior.joint_prior == prior

        with pytest.raises(
            TypeError,
            match="All priors must either be of type pybop.Parameter, pybop.Distribution or scipy.stats.distributions.rv_frozen",
        ):
            posterior = pybop.LogPosterior(likelihood, prior=st.norm)
            posterior.set_joint_prior()

    def test_log_posterior_construction_no_prior(self, simulator, likelihood):
        # Test log posterior construction without prior
        posterior = pybop.LogPosterior(likelihood, prior=None)
        problem = pybop.Problem(simulator, posterior)

        problem._cost.set_joint_prior()
        assert problem._cost.joint_prior is not None
        assert isinstance(problem._cost.joint_prior, pybop.JointDistribution)

        for i, p in enumerate(problem.parameters.distribution()):
            assert p == problem._cost.joint_prior._distribution[i]

    @pytest.fixture
    def problem(self, simulator, likelihood, prior):
        posterior = pybop.LogPosterior(likelihood, prior=prior)
        return pybop.Problem(simulator, posterior)

    def test_log_posterior(self, problem):
        # Test log posterior
        x = np.array([0.50])
        assert np.allclose(problem.evaluate(x).values, 51.6033, atol=2e-2)

        # Test log posterior evaluateS1
        p, dp = problem.evaluate(x, calculate_sensitivities=True).get_values()
        assert np.allclose(p, 51.6033, atol=2e-2)
        assert np.allclose(dp, 0.4266, atol=2e-2)

    @pytest.fixture
    def posterior_uniform_prior(self, simulator, likelihood):
        posterior = pybop.LogPosterior(likelihood, prior=pybop.Uniform(0.45, 0.55))
        return pybop.Problem(simulator, posterior)

    def test_log_posterior_inf(self, posterior_uniform_prior):
        # Test prior np.inf
        assert not np.isfinite(posterior_uniform_prior.evaluate([1]).values)
        assert not np.isfinite(
            posterior_uniform_prior.evaluate([1], calculate_sensitivities=True).values
        )

    def test_non_logpdf_prior(self, simulator, likelihood):
        problem = pybop.Problem(simulator, likelihood)
        l = problem.evaluate([0.6]).values

        # Scipy distribution
        prior = st.norm(loc=0.8, scale=0.01)
        posterior = pybop.LogPosterior(likelihood, prior=prior)
        problem = pybop.Problem(simulator, posterior)
        p = problem.evaluate([0.6]).values

        # Assert to pybop.Gaussian
        p2 = pybop.Gaussian(mean=0.8, sigma=0.01).logpdf(0.6)
        np.testing.assert_allclose(p - l, p2, atol=2e-3)
