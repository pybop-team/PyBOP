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
    def model(self):
        return pybamm.lithium_ion.SPM()

    @pytest.fixture
    def parameters(self):
        return [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.5, 0.01),
                bounds=[0.375, 0.625],
                transformation=pybop.ScaledTransformation(
                    coefficient=1 / 0.25, intercept=-0.375
                ),
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.6, 0.2),
                transformation=pybop.LogTransformation(),
            ),
        ]

    @pytest.fixture
    def experiment(self):
        return pybamm.Experiment(["Discharge at 1C for 1 minute (6 second period)"])

    @pytest.fixture
    def dataset(self, model, experiment):
        sim = pybamm.Simulation(model, experiment=experiment)
        sol = sim.solve()
        return pybop.Dataset(
            {
                "Time [s]": sol.t,
                "Current function [A]": sol["Current [A]"].data,
                "Voltage [V]": sol["Terminal voltage [V]"].data,
            }
        )

    @pytest.fixture
    def builder(self, model, parameters, dataset):
        builder = pybop.builders.Pybamm().set_simulation(model).set_dataset(dataset)
        for parameter in parameters:
            builder.add_parameter(parameter)
        return builder

    @pytest.fixture(
        params=[
            pybop.costs.pybamm.RootMeanSquaredError,
            pybop.costs.pybamm.SumSquaredError,
        ]
    )
    def problem(self, builder, request):
        cost_class = request.param
        builder.add_cost(cost_class("Voltage [V]", "Voltage [V]"))
        return builder.build()

    # def test_evaluator_transformations(self, problem):
    #     for minimise in [True, False]:
    #         # Test the transformed cost and sensitivities
    #         logger = pybop.Logger()
    #         evaluator = pybop.ScalarEvaluator(
    #             problem=problem,
    #             minimise=minimise,
    #             with_sensitivities=False,
    #             logger=logger,
    #         )
    #         cost1 = evaluator.evaluate(self.x_search)

    #         numerical_grad = []
    #         for i in range(len(self.x_search)):
    #             delta = 1e-2 * self.x_search[i]
    #             self.x_search[i] += delta / 2
    #             cost_right = evaluator.evaluate(self.x_search)
    #             self.x_search[i] -= delta
    #             cost_left = evaluator.evaluate(self.x_search)
    #             self.x_search[i] += delta / 2
    #             assert np.abs(cost_right - cost_left) > 0
    #             numerical_grad.append((cost_right - cost_left) / delta)
    #         numerical_grad = np.asarray(numerical_grad).reshape(-1)

    #         evaluator_ws = pybop.ScalarEvaluator(
    #             problem=problem,
    #             minimise=minimise,
    #             with_sensitivities=True,
    #             logger=logger,
    #         )
    #         cost2, grad2 = evaluator_ws.evaluate(self.x_search)

    #         np.testing.assert_allclose(cost2, cost1, rtol=3e-5)
    #         np.testing.assert_allclose(grad2, numerical_grad, rtol=6e-4)

    def test_evaluator_transformations(self, problem):
        # First compute the cost and sensitivities in the model space
        cost1 = problem.run(self.x_model)
        cost1_ws, grad1_wrt_model_parameters = problem.run_with_sensitivities(
            self.x_model
        )

        numerical_grad1 = []
        for i in range(len(self.x_model)):
            delta = 1e-8 * self.x_model[i]
            self.x_model[i] += delta / 2
            cost_right = problem.run(self.x_model)
            self.x_model[i] -= delta
            cost_left = problem.run(self.x_model)
            self.x_model[i] += delta / 2
            assert np.abs(cost_right - cost_left) > 0
            numerical_grad1.append((cost_right - cost_left) / delta)
        numerical_grad1 = np.asarray(numerical_grad1).reshape(-1)
        np.testing.assert_allclose(
            grad1_wrt_model_parameters, numerical_grad1, rtol=5e-5
        )

        jac = problem.params.transformation.jacobian(self.x_search)
        grad1_wrt_search_parameters = np.matmul(grad1_wrt_model_parameters, jac)

        # Test the transformed cost and sensitivities
        logger = pybop.Logger()
        evaluator = pybop.ScalarEvaluator(
            problem=problem,
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
            assert np.abs(cost_right - cost_left) > 0
            numerical_grad2.append((cost_right - cost_left) / delta)
        numerical_grad2 = np.asarray(numerical_grad2).reshape(-1)
        np.testing.assert_allclose(
            grad1_wrt_search_parameters, numerical_grad2, rtol=5e-5
        )

        evaluator_ws = pybop.ScalarEvaluator(
            problem=problem,
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
            problem=problem,
            minimise=False,
            with_sensitivities=False,
            logger=logger,
        )
        cost3 = evaluator.evaluate(self.x_search)
        np.testing.assert_allclose(cost3, -cost1)

        evaluator_ws = pybop.ScalarEvaluator(
            problem=problem,
            minimise=False,
            with_sensitivities=True,
            logger=logger,
        )
        cost_ws3, grad3_with_transformations = evaluator_ws.evaluate(self.x_search)
        np.testing.assert_allclose(cost_ws3, -cost1, rtol=1e-5)
        np.testing.assert_allclose(
            grad3_with_transformations, -grad1_wrt_search_parameters, rtol=5e-5
        )
