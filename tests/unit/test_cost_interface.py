import numpy as np
import pytest

import pybop


class TestCostInterface:
    """
    Class for unit testing cost functions with transformations applied.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture(autouse=True)
    def setup(self):
        self.x_model = [0.525, 1.5]
        self.x_search = [1 / 0.25 * (self.x_model[0] - 0.375), np.log(self.x_model[1])]

    @pytest.fixture
    def model(self):
        return pybop.lithium_ion.SPM()

    @pytest.fixture
    def parameters(self):
        parameters = pybop.Parameters(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.5, 0.01),
                bounds=[0.375, 0.625],
                transformation=pybop.ScaledTransformation(
                    coefficient=1 / 0.25, intercept=-0.375
                ),
            ),
            pybop.Parameter(
                "Positive electrode Bruggeman coefficient (electrode)",
                prior=pybop.Gaussian(1.5, 0.1),
                transformation=pybop.LogTransformation(),
            ),
        )
        return parameters

    @pytest.fixture
    def experiment(self):
        return pybop.Experiment(
            [
                ("Discharge at 1C for 1 minutes (6 second period)"),
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
    def problem(self, model, parameters, dataset, request):
        problem = pybop.FittingProblem(model, parameters, dataset)
        return problem

    @pytest.fixture(
        params=[
            pybop.RootMeanSquaredError,
            pybop.SumSquaredError,
            pybop.Minkowski,
            pybop.SumofPower,
            pybop.ObserverCost,
            pybop.LogPosterior,
            pybop.GaussianLogLikelihood,
            pybop.GaussianLogLikelihoodKnownSigma,
        ]
    )
    def cost(self, problem, request):
        cls = request.param
        if cls in [pybop.SumSquaredError, pybop.RootMeanSquaredError]:
            return cls(problem)
        elif cls is pybop.LogPosterior:
            return cls(pybop.GaussianLogLikelihoodKnownSigma(problem, sigma0=0.002))
        elif cls is pybop.GaussianLogLikelihoodKnownSigma:
            return pybop.GaussianLogLikelihoodKnownSigma(problem, sigma0=0.002)
        elif cls is pybop.ObserverCost:
            inputs = problem.parameters.initial_value()
            state = problem.model.reinit(inputs)
            n = len(state)
            sigma_diag = [0.0] * n
            sigma_diag[0] = 1e-4
            sigma_diag[1] = 1e-4
            process_diag = [0.0] * n
            process_diag[0] = 1e-4
            process_diag[1] = 1e-4
            sigma0 = np.diag(sigma_diag)
            process = np.diag(process_diag)
            dataset = pybop.Dataset(data_dictionary=problem.dataset)
            return cls(
                pybop.UnscentedKalmanFilterObserver(
                    problem.parameters,
                    problem.model,
                    sigma0=sigma0,
                    process=process,
                    measure=1e-4,
                    dataset=dataset,
                    signal=problem.signal,
                ),
            )
        else:
            return cls(problem)

    def test_cost_transformations(self, cost):
        if isinstance(cost, pybop.GaussianLogLikelihood):
            self.x_model.append(0.002)
            self.x_search.append(0.002)

        optim = pybop.Optimisation(cost=cost, x0=self.x_model)

        true_cost = cost(self.x_model)
        cost_with_transformation = optim.call_cost(self.x_search, cost=optim.cost) * (
            1 if cost.minimising else -1
        )
        np.testing.assert_allclose(true_cost, cost_with_transformation)

    def test_cost_gradient_transformed(self, cost):
        # Gradient transformations are not implemented on ObserverCost
        if isinstance(cost, pybop.ObserverCost):
            return

        if isinstance(cost, pybop.GaussianLogLikelihood):
            self.x_model.append(0.002)
            self.x_search.append(0.002)

        optim = pybop.Optimisation(cost=cost, x0=self.x_model)

        true_cost, grad_wrt_model_parameters = cost(self.x_model, calculate_grad=True)
        cost_with_transformation, gradient_with_transformation = optim.call_cost(
            self.x_search, cost=optim.cost, calculate_grad=True
        )
        cost_with_transformation = cost_with_transformation * (
            1 if cost.minimising else -1
        )
        gradient_with_transformation = gradient_with_transformation * (
            1 if cost.minimising else -1
        )

        np.testing.assert_allclose(true_cost, cost_with_transformation)

        numerical_grad = []
        for i in range(len(self.x_model)):
            delta = 1e-6 * self.x_model[i]
            self.x_model[i] += delta / 2
            cost_right = cost(self.x_model)
            self.x_model[i] -= delta
            cost_left = cost(self.x_model)
            self.x_model[i] += delta / 2
            numerical_grad.append((cost_right - cost_left) / delta)

        np.testing.assert_allclose(grad_wrt_model_parameters, numerical_grad, rtol=6e-4)

        jac = optim.transformation.jacobian(self.x_search)
        gradient_wrt_search_parameters = np.matmul(grad_wrt_model_parameters, jac)

        np.testing.assert_allclose(
            gradient_wrt_search_parameters, gradient_with_transformation
        )
