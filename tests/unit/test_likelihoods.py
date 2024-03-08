from __future__ import annotations
import pytest
import pybop
import numpy as np


class TestLikelihoods:
    """
    Class for likelihood unit tests
    """

    @pytest.fixture
    def model(self):
        return pybop.lithium_ion.SPM()

    @pytest.fixture
    def parameters(self):
        return [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.5, 0.01),
                bounds=[0.375, 0.625],
            ),
        ]

    @pytest.fixture
    def experiment(self):
        return pybop.Experiment(
            [
                ("Discharge at 1C for 10 minutes (20 second period)"),
            ]
        )

    @pytest.fixture
    def x0(self):
        return np.array([0.52])

    @pytest.fixture
    def dataset(self, model, experiment, x0):
        model.parameter_set = model.pybamm_model.default_parameter_values
        model.parameter_set.update(
            {
                "Negative electrode active material volume fraction": x0[0],
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
    def signal(self):
        return "Voltage [V]"

    @pytest.fixture()
    def problem(self, model, parameters, dataset, signal, x0):
        problem = pybop.FittingProblem(
            model, parameters, dataset, signal=signal, x0=x0, init_soc=1.0
        )
        return problem

    @pytest.mark.unit
    def test_base_likelihood_init(self, problem):
        likelihood = pybop.BaseLikelihood(problem, sigma=np.array([0.2]))
        assert likelihood.problem == problem
        assert likelihood._n_output == 1
        assert likelihood._n_times == problem.n_time_data
        assert np.array_equal(likelihood.get_sigma(), np.array([0.2]))
        assert likelihood.x0 == problem.x0
        assert likelihood.bounds == problem.bounds
        assert likelihood._n_parameters == 1
        assert np.array_equal(likelihood._target, problem._target)

    @pytest.mark.unit
    def test_base_likelihood_set_get_sigma(self, problem):
        likelihood = pybop.BaseLikelihood(problem)
        likelihood.set_sigma(np.array([0.3]))
        assert np.array_equal(likelihood.get_sigma(), np.array([0.3]))

    @pytest.mark.unit
    def test_base_likelihood_set_sigma_raises_value_error_for_negative_sigma(
        self, problem
    ):
        likelihood = pybop.BaseLikelihood(problem)
        with pytest.raises(ValueError):
            likelihood.set_sigma(np.array([-0.2]))

    @pytest.mark.unit
    def test_base_likelihood_get_n_parameters(self, problem):
        likelihood = pybop.BaseLikelihood(problem)
        assert likelihood.get_n_parameters() == 1

    @pytest.mark.unit
    def test_base_likelihood_n_parameters_property(self, problem):
        likelihood = pybop.BaseLikelihood(problem)
        assert likelihood.n_parameters == 1

    @pytest.mark.unit
    def test_gaussian_log_likelihood_known_sigma(self, problem):
        likelihood = pybop.GaussianLogLikelihoodKnownSigma(
            problem, sigma=np.array([1.0])
        )
        result = likelihood(np.array([0.5]))
        grad_result, grad_likelihood = likelihood._evaluateS1(np.array([0.5]))
        assert isinstance(result, float)
        np.testing.assert_allclose(result, grad_result, atol=1e-5)
        assert np.all(grad_likelihood <= 0)

    @pytest.mark.unit
    def test_gaussian_log_likelihood(self, problem):
        likelihood = pybop.GaussianLogLikelihood(problem)
        result = likelihood(np.array([0.5, 0.5]))
        grad_result, grad_likelihood = likelihood._evaluateS1(np.array([0.5, 0.5]))
        assert isinstance(result, float)
        np.testing.assert_allclose(result, grad_result, atol=1e-5)
        assert np.all(grad_likelihood <= 0)

    @pytest.mark.unit
    def test_gaussian_log_likelihood_call_returns_negative_inf_for_non_positive_sigma(
        self, problem
    ):
        likelihood = pybop.GaussianLogLikelihood(problem)
        result = likelihood(np.array([-0.5]))
        assert result == -np.inf
