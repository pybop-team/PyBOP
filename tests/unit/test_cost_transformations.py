import numpy as np
import pytest

import pybop


class TestCostsTransformations:
    """
    Class for unit testing cost functions with transformations applied.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.x1 = [0.5, 1]
        self.x2 = [(self.x1[0] + 1) * -2.5, np.log(1)]

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
                    coefficient=-2.5, intercept=1
                ),
            ),
            pybop.Parameter(
                "Positive electrode Bruggeman coefficient (electrode)",
                prior=pybop.Gaussian(1, 0.1),
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

    @pytest.mark.unit
    def test_cost_transformations(self, cost):
        if isinstance(cost, pybop.GaussianLogLikelihood):
            self.x1.append(0.002)
            self.x2.append(0.002)

        # Asserts
        non_transformed_cost = cost(self.x1)
        transformed_cost = cost(self.x2, apply_transform=True)
        np.testing.assert_allclose(non_transformed_cost, transformed_cost)

    @pytest.mark.unit
    def test_cost_gradient_transformed(self, cost):
        # Gradient transformations are not implmented on ObserverCost
        if isinstance(cost, pybop.ObserverCost):
            return

        if isinstance(cost, pybop.GaussianLogLikelihood):
            self.x1.append(0.002)
            self.x2.append(0.002)
        # Asserts
        non_transformed_cost, grad = cost(self.x1, calculate_grad=True)
        transformed_cost, transformed_gradient = cost(
            self.x2, calculate_grad=True, apply_transform=True
        )
        np.testing.assert_allclose(transformed_cost, non_transformed_cost)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(transformed_gradient, grad)
