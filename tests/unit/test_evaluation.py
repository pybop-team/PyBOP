import numpy as np
import pybamm
import pytest

import pybop


class TestEvaluation:
    """
    Class for unit testing cost and sensitivity evaluations with transformations applied.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture(autouse=True)
    def setup(self):
        self.x_model = [0.525, 1.5]
        self.x_search = [1 / 0.25 * (self.x_model[0] - 0.375), np.log(self.x_model[1])]

    @pytest.fixture
    def solver(self):
        return pybamm.IDAKLUSolver(atol=5e-6, rtol=5e-6)

    @pytest.fixture
    def model(self):
        return pybamm.lithium_ion.SPM()

    @pytest.fixture
    def parameters(self):
        return pybop.Parameters(
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

    @pytest.fixture
    def experiment(self):
        return pybamm.Experiment(["Discharge at 1C for 1 minutes (6 second period)"])

    @pytest.fixture
    def dataset(self, model, experiment, solver):
        solution = pybamm.Simulation(
            model, experiment=experiment, solver=solver
        ).solve()
        return pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Terminal voltage [V]"].data,
            }
        )

    @pytest.fixture
    def problem(self, model, parameters, dataset, solver, request):
        simulator = pybop.pybamm.Simulator(
            model,
            input_parameter_names=parameters.names,
            protocol=dataset,
            solver=solver,
        )
        return pybop.FittingProblem(simulator, parameters, dataset)

    @pytest.fixture(
        params=[
            pybop.RootMeanSquaredError,
            pybop.SumSquaredError,
            pybop.Minkowski,
            pybop.SumOfPower,
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
        else:
            return cls(problem)

    def test_evaluator_transformations(self, cost):
        if isinstance(cost, pybop.GaussianLogLikelihood):
            self.x_model.append(0.002)
            self.x_search.append(0.002)

        # First compute the cost and sensitivities in the model space
        cost1 = cost.__call__(self.x_model)
        cost1_ws, grad1_wrt_model_parameters = cost.__call__(
            self.x_model, calculate_grad=True
        )

        numerical_grad1 = []
        for i in range(len(self.x_model)):
            delta = 1e-8 * self.x_model[i]
            self.x_model[i] += delta / 2
            cost_right = cost.__call__(self.x_model)
            self.x_model[i] -= delta
            cost_left = cost.__call__(self.x_model)
            self.x_model[i] += delta / 2
            numerical_grad1.append((cost_right - cost_left) / delta)
        numerical_grad1 = np.asarray(numerical_grad1).reshape(-1)
        np.testing.assert_allclose(
            grad1_wrt_model_parameters, numerical_grad1, rtol=5e-5
        )

        jac = cost.parameters.transformation.jacobian(self.x_search)
        grad1_wrt_search_parameters = np.matmul(grad1_wrt_model_parameters, jac)

        # Test the transformed cost and sensitivities
        logger = pybop.Logger(minimising=True)
        evaluator = pybop.ScalarEvaluator(
            cost=cost,
            minimise=True,
            with_sensitivities=False,
            logger=logger,
        )
        cost2 = evaluator.evaluate(self.x_search)
        np.testing.assert_allclose(cost2, cost1)

        numerical_grad2 = []
        for i in range(len(self.x_search)):
            delta = 1e-8 * self.x_search[i]
            self.x_search[i] += delta / 2
            cost_right = evaluator.evaluate(self.x_search)
            self.x_search[i] -= delta
            cost_left = evaluator.evaluate(self.x_search)
            self.x_search[i] += delta / 2
            numerical_grad2.append((cost_right - cost_left) / delta)
        numerical_grad2 = np.asarray(numerical_grad2).reshape(-1)
        np.testing.assert_allclose(
            grad1_wrt_search_parameters, numerical_grad2, rtol=5e-5
        )

        evaluator_ws = pybop.ScalarEvaluator(
            cost=cost,
            minimise=True,
            with_sensitivities=True,
            logger=logger,
        )
        cost2_ws, grad2_with_transformations = evaluator_ws.evaluate(self.x_search)
        np.testing.assert_allclose(cost2_ws, cost1, rtol=1e-5)
        np.testing.assert_allclose(
            grad2_with_transformations, grad1_wrt_search_parameters, rtol=5e-5
        )

        # Also test the sign change for maximisation
        evaluator = pybop.ScalarEvaluator(
            cost=cost,
            minimise=False,
            with_sensitivities=False,
            logger=logger,
        )
        cost3 = evaluator.evaluate(self.x_search)
        np.testing.assert_allclose(cost3, -cost1)

        evaluator_ws = pybop.ScalarEvaluator(
            cost=cost,
            minimise=False,
            with_sensitivities=True,
            logger=logger,
        )
        cost_ws3, grad3_with_transformations = evaluator_ws.evaluate(self.x_search)
        np.testing.assert_allclose(cost_ws3, -cost1, rtol=1e-5)
        np.testing.assert_allclose(
            grad3_with_transformations, -grad1_wrt_search_parameters, rtol=5e-5
        )
