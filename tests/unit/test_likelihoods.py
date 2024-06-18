import numpy as np
import pytest

import pybop


class TestLikelihoods:
    """
    Class for likelihood unit tests
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
        model.parameter_set = model.pybamm_model.default_parameter_values
        model.parameter_set.update(
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
        signal = ["Voltage [V]"]
        return pybop.FittingProblem(
            model, parameters, dataset, signal=signal, init_soc=1.0
        )

    @pytest.fixture
    def two_signal_problem(self, model, parameters, dataset):
        signal = ["Time [s]", "Voltage [V]"]
        return pybop.FittingProblem(
            model, parameters, dataset, signal=signal, init_soc=1.0
        )

    @pytest.mark.parametrize(
        "problem_name, n_outputs",
        [("one_signal_problem", 1), ("two_signal_problem", 2)],
    )
    @pytest.mark.unit
    def test_base_likelihood_init(self, problem_name, n_outputs, request):
        problem = request.getfixturevalue(problem_name)
        likelihood = pybop.BaseLikelihood(problem)
        assert likelihood.problem == problem
        assert likelihood.n_outputs == n_outputs
        assert likelihood.n_time_data == problem.n_time_data
        assert likelihood.x0 == problem.x0
        assert likelihood.n_parameters == 1
        assert np.array_equal(likelihood._target, problem._target)

    @pytest.mark.unit
    def test_base_likelihood_call_raises_not_implemented_error(
        self, one_signal_problem
    ):
        likelihood = pybop.BaseLikelihood(one_signal_problem)
        with pytest.raises(NotImplementedError):
            likelihood(np.array([0.5, 0.5]))

    @pytest.mark.unit
    def test_set_get_sigma(self, one_signal_problem):
        likelihood = pybop.GaussianLogLikelihoodKnownSigma(one_signal_problem, 0.1)
        likelihood.set_sigma(np.array([0.3]))
        assert np.array_equal(likelihood.get_sigma(), np.array([0.3]))

        with pytest.raises(
            ValueError,
            match="The GaussianLogLikelihoodKnownSigma cost requires sigma to be "
            + "either a scalar value or an array with one entry per dimension.",
        ):
            pybop.GaussianLogLikelihoodKnownSigma(one_signal_problem, sigma=None)

        likelihood = pybop.GaussianLogLikelihoodKnownSigma(one_signal_problem, 0.1)
        with pytest.raises(ValueError):
            likelihood.set_sigma(np.array([-0.2]))

    @pytest.mark.unit
    def test_base_likelihood_n_parameters_property(self, one_signal_problem):
        likelihood = pybop.BaseLikelihood(one_signal_problem)
        assert likelihood.n_parameters == 1

    @pytest.mark.parametrize(
        "problem_name", ["one_signal_problem", "two_signal_problem"]
    )
    @pytest.mark.unit
    def test_gaussian_log_likelihood_known_sigma(self, problem_name, request):
        problem = request.getfixturevalue(problem_name)
        likelihood = pybop.GaussianLogLikelihoodKnownSigma(
            problem, sigma=np.array([1.0])
        )
        result = likelihood(np.array([0.5]))
        grad_result, grad_likelihood = likelihood.evaluateS1(np.array([0.5]))
        assert isinstance(result, float)
        np.testing.assert_allclose(result, grad_result, atol=1e-5)
        assert np.all(grad_likelihood <= 0)

    @pytest.mark.unit
    def test_gaussian_log_likelihood(self, one_signal_problem):
        likelihood = pybop.GaussianLogLikelihood(one_signal_problem)
        result = likelihood(np.array([0.5, 0.5]))
        grad_result, grad_likelihood = likelihood.evaluateS1(np.array([0.5, 0.5]))
        assert isinstance(result, float)
        np.testing.assert_allclose(result, grad_result, atol=1e-5)
        assert np.all(grad_likelihood <= 0)

    @pytest.mark.unit
    def test_gaussian_log_likelihood_returns_negative_inf(self, one_signal_problem):
        likelihood = pybop.GaussianLogLikelihood(one_signal_problem)
        assert likelihood(np.array([-0.5, -0.5])) == -np.inf  # negative sigma value
        assert (
            likelihood.evaluateS1(np.array([-0.5, -0.5]))[0] == -np.inf
        )  # negative sigma value
        assert likelihood(np.array([0.01, 0.1])) == -np.inf  # parameter value too small
        assert (
            likelihood.evaluateS1(np.array([0.01, 0.1]))[0] == -np.inf
        )  # parameter value too small

    @pytest.mark.unit
    def test_gaussian_log_likelihood_known_sigma_returns_negative_inf(
        self, one_signal_problem
    ):
        likelihood = pybop.GaussianLogLikelihoodKnownSigma(
            one_signal_problem, sigma=np.array([0.2])
        )
        assert likelihood(np.array([0.01])) == -np.inf  # parameter value too small
        assert (
            likelihood.evaluateS1(np.array([0.01]))[0] == -np.inf
        )  # parameter value too small
