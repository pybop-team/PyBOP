import numpy as np
import pybamm
import pytest

import pybop


class TestLikelihoods:
    """
    Class for likelihood unit tests
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def model(self, ground_truth):
        solver = pybamm.IDAKLUSolver()
        model = pybop.lithium_ion.SPM(solver=solver)
        model.parameter_set.update(
            {
                "Negative electrode active material volume fraction": ground_truth,
            }
        )
        return model

    @pytest.fixture
    def ground_truth(self):
        return 0.52

    @pytest.fixture
    def parameters(self):
        return pybop.Parameters(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.5, 0.01),
                bounds=[0.375, 0.625],
            )
        )

    @pytest.fixture
    def experiment(self):
        return pybop.Experiment(
            [
                ("Discharge at 1C for 1 minutes (5 second period)"),
            ]
        )

    @pytest.fixture
    def dataset(self, model, experiment):
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
        return pybop.FittingProblem(model, parameters, dataset, signal=signal)

    @pytest.fixture
    def two_signal_problem(self, model, parameters, dataset):
        signal = ["Time [s]", "Voltage [V]"]
        return pybop.FittingProblem(model, parameters, dataset, signal=signal)

    @pytest.mark.parametrize(
        "problem_name, n_outputs",
        [("one_signal_problem", 1), ("two_signal_problem", 2)],
    )
    def test_base_likelihood_init(self, problem_name, n_outputs, request):
        problem = request.getfixturevalue(problem_name)
        likelihood = pybop.BaseLikelihood(problem)
        assert likelihood.problem == problem
        assert likelihood.n_outputs == n_outputs
        assert likelihood.n_data == problem.n_data
        assert likelihood.n_parameters == 1
        assert np.array_equal(likelihood.target, problem.target)

    def test_base_likelihood_call_raises_not_implemented_error(
        self, one_signal_problem
    ):
        likelihood = pybop.BaseLikelihood(one_signal_problem)
        with pytest.raises(NotImplementedError):
            likelihood(np.array([0.5]))

    def test_likelihood_check_sigma0(self, one_signal_problem):
        with pytest.raises(
            ValueError,
            match="Sigma0 must be positive",
        ):
            pybop.GaussianLogLikelihoodKnownSigma(one_signal_problem, sigma0=None)

        likelihood = pybop.GaussianLogLikelihoodKnownSigma(one_signal_problem, 0.1)
        sigma = likelihood.check_sigma0(0.2)
        assert sigma == np.array(0.2)

        with pytest.raises(
            ValueError,
            match=r"sigma0 must be either a scalar value",
        ):
            pybop.GaussianLogLikelihoodKnownSigma(one_signal_problem, sigma0=[0.2, 0.3])

    def test_base_likelihood_n_parameters_property(self, one_signal_problem):
        likelihood = pybop.BaseLikelihood(one_signal_problem)
        assert likelihood.n_parameters == 1

    @pytest.mark.parametrize(
        "problem_name", ["one_signal_problem", "two_signal_problem"]
    )
    def test_gaussian_log_likelihood_known_sigma(self, problem_name, request):
        problem = request.getfixturevalue(problem_name)
        likelihood = pybop.GaussianLogLikelihoodKnownSigma(
            problem, sigma0=np.array([1.0])
        )
        result = likelihood(np.array([0.5]))
        grad_result, grad_likelihood = likelihood(np.array([0.5]), calculate_grad=True)
        assert isinstance(result, float)
        np.testing.assert_allclose(result, grad_result, atol=1e-5)
        # Since 0.5 < ground_truth, the likelihood should be increasing
        assert grad_likelihood >= 0

    def test_gaussian_log_likelihood(self, one_signal_problem):
        likelihood = pybop.GaussianLogLikelihood(one_signal_problem)
        result = likelihood(np.array([0.8, 0.2]))
        grad_result, grad_likelihood = likelihood(
            np.array([0.8, 0.2]), calculate_grad=True
        )
        assert isinstance(result, float)
        np.testing.assert_allclose(result, grad_result, atol=1e-5)
        # Since 0.8 > ground_truth, the likelihood should be decreasing
        assert grad_likelihood[0] <= 0
        # Since sigma < 0.5, the likelihood should be decreasing
        assert grad_likelihood[1] <= 0

        # Test construction with sigma as a Parameter
        sigma = pybop.Parameter("sigma", prior=pybop.Uniform(0.4, 0.6))
        likelihood = pybop.GaussianLogLikelihood(one_signal_problem, sigma0=sigma)

        # Test invalid sigma
        with pytest.raises(
            TypeError,
            match=r"Expected sigma0 to contain Parameter objects or numeric values.",
        ):
            likelihood = pybop.GaussianLogLikelihood(
                one_signal_problem, sigma0="Invalid string"
            )

    def test_gaussian_log_likelihood_dsigma_scale(self, one_signal_problem):
        likelihood = pybop.GaussianLogLikelihood(one_signal_problem, dsigma_scale=0.05)
        assert likelihood.dsigma_scale == 0.05
        likelihood.dsigma_scale = 1e3
        assert likelihood.dsigma_scale == 1e3

        # Test incorrect sigma scale
        with pytest.raises(ValueError):
            likelihood.dsigma_scale = -1e3

    def test_gaussian_log_likelihood_returns_negative_inf(self, one_signal_problem):
        likelihood = pybop.GaussianLogLikelihood(one_signal_problem)
        assert likelihood(np.array([0.01, 0.1])) == -np.inf  # parameter value too small
        assert (
            likelihood(np.array([0.01, 0.1]), calculate_grad=True)[0] == -np.inf
        )  # parameter value too small

    def test_gaussian_log_likelihood_known_sigma_returns_negative_inf(
        self, one_signal_problem
    ):
        likelihood = pybop.GaussianLogLikelihoodKnownSigma(
            one_signal_problem, sigma0=np.array([0.2])
        )
        assert likelihood(np.array([0.01])) == -np.inf  # parameter value too small
        assert (
            likelihood(np.array([0.01]), calculate_grad=True)[0] == -np.inf
        )  # parameter value too small

    def test_scaled_log_likelihood(self, one_signal_problem):
        likelihood = pybop.GaussianLogLikelihoodKnownSigma(
            one_signal_problem, sigma0=0.02
        )
        scaled_likelihood = pybop.ScaledLogLikelihood(likelihood)

        assert scaled_likelihood._log_likelihood == likelihood
        assert (
            scaled_likelihood(np.array([0.01])) == -np.inf
        )  # parameter value too small
        assert (
            likelihood(np.array([0.01]), calculate_grad=True)[0] == -np.inf
        )  # parameter value too small

        # Test scaling
        scaled_values = likelihood([0.6]) / likelihood.n_data
        np.testing.assert_allclose(scaled_values, scaled_likelihood([0.6]))

        # Test w/ gradient
        values, grad = tuple(
            i / likelihood.n_data for i in likelihood([0.6], calculate_grad=True)
        )
        np.testing.assert_allclose(
            grad, scaled_likelihood([0.6], calculate_grad=True)[1]
        )

    @pytest.mark.parametrize(
        "likelihood_cls",
        [
            pybop.GaussianLogLikelihoodKnownSigma,
            pybop.JaxGaussianLogLikelihoodKnownSigma,
        ],
    )
    def test_fisher_matrix(
        self, likelihood_cls, one_signal_problem, model, dataset, parameters
    ):
        likelihood = likelihood_cls(one_signal_problem, sigma0=1e-3)
        fisher = likelihood.observed_fisher([0.5])
        assert isinstance(fisher, np.ndarray)

        # Test fisher does not compute for non-gradient available parameters

        parameters.add(
            pybop.Parameter(
                "Negative particle radius [m]",
                prior=pybop.Gaussian(6e-06, 0.1e-6),
                bounds=[1e-6, 9e-6],
            ),
        )
        problem = pybop.FittingProblem(model, parameters, dataset)
        likelihood_non_grad = likelihood_cls(problem, sigma0=1e-3)
        fisher = likelihood_non_grad.observed_fisher([0.5, 5e-06])
        assert fisher is None
