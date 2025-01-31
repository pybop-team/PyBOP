import numpy as np
import pytest
from pybamm import Parameter

import pybop


class TestClassification:
    """
    A class to test the classification of different optimisation results.
    """

    pytestmark = pytest.mark.integration

    @pytest.fixture(
        params=[
            np.asarray([0.05, 0.05]),
            np.asarray([0.1, 0.05]),
            np.asarray([0.05, 0.01]),
        ]
    )
    def parameters(self, request):
        ground_truth = request.param
        return pybop.Parameters(
            pybop.Parameter(
                "R0 [Ohm]",
                prior=pybop.Gaussian(0.05, 0.01),
                bounds=[0.02, 0.08],
                true_value=ground_truth[0],
            ),
            pybop.Parameter(
                "R1 [Ohm]",
                prior=pybop.Gaussian(0.05, 0.01),
                bounds=[0.02, 0.08],
                true_value=ground_truth[1],
            ),
        )

    @pytest.fixture
    def parameter_set(self):
        parameter_set = pybop.ParameterSet(
            json_path="examples/parameters/initial_ecm_parameters.json"
        )
        parameter_set.update({"C1 [F]": 1000})
        return parameter_set

    @pytest.fixture
    def model(self, parameter_set, parameters):
        parameter_set.update(parameters.as_dict(parameters.true_value()))
        return pybop.empirical.Thevenin(parameter_set=parameter_set)

    @pytest.fixture
    def dataset(self, model):
        experiment = pybop.Experiment(
            [
                "Discharge at 0.5C for 2 minutes (4 seconds period)",
                "Charge at 0.5C for 2 minutes (4 seconds period)",
            ]
        )
        solution = model.predict(experiment=experiment)
        return pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data,
            }
        )

    @pytest.fixture
    def problem(self, model, parameters, dataset):
        return pybop.FittingProblem(model, parameters, dataset)

    def test_classify_using_hessian(self, problem):
        cost = pybop.RootMeanSquaredError(problem)
        x = cost.parameters.true_value()
        bounds = cost.parameters.get_bounds()
        x0 = np.clip(x, bounds["lower"], bounds["upper"])
        optim = pybop.Optimisation(cost=cost)
        results = pybop.OptimisationResult(x=x0, optim=optim)

        if np.all(x == np.asarray([0.05, 0.05])):
            message = pybop.classify_using_hessian(results)
            assert message == "The optimiser has located a minimum."
        elif np.all(x == np.asarray([0.1, 0.05])):
            message = pybop.classify_using_hessian(results)
            assert message == (
                "The optimiser has not converged to a stationary point."
                " The result is near the upper bound of R0 [Ohm]."
            )
        elif np.all(x == np.asarray([0.05, 0.01])):
            message = pybop.classify_using_hessian(results)
            assert message == (
                "The optimiser has not converged to a stationary point."
                " The result is near the lower bound of R1 [Ohm]."
            )
        else:
            raise Exception(f"Please add a check for these values: {x}")

        if np.all(x == np.asarray([0.05, 0.05])):
            cost = pybop.GaussianLogLikelihoodKnownSigma(problem, sigma0=0.002)
            optim = pybop.Optimisation(cost=cost)
            results = pybop.OptimisationResult(x=x, optim=optim)

            message = pybop.classify_using_hessian(results)
            assert message == "The optimiser has located a maximum."

        # message = pybop.classify_using_hessian(results)
        # assert message == "The optimiser has located a saddle point."

    def test_insensitive_classify_using_hessian(self, parameter_set):
        param_R0_a = pybop.Parameter(
            "R0_a [Ohm]",
            bounds=[0, 0.002],
            true_value=0.001,
        )
        param_R0_b = pybop.Parameter(
            "R0_b [Ohm]",
            bounds=[-1e-4, 1e-4],
            true_value=0,
        )
        parameter_set.update(
            {"R0_a [Ohm]": 0.001, "R0_b [Ohm]": 0},
            check_already_exists=False,
        )
        parameter_set.update(
            {"R0 [Ohm]": Parameter("R0_a [Ohm]") + Parameter("R0_b [Ohm]")},
        )
        model = pybop.empirical.Thevenin(parameter_set=parameter_set)

        experiment = pybop.Experiment(
            ["Discharge at 0.5C for 2 minutes (4 seconds period)"]
        )
        solution = model.predict(experiment=experiment)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data,
            }
        )

        for parameters in [
            pybop.Parameters(param_R0_b, param_R0_a),
            pybop.Parameters(param_R0_a, param_R0_b),
        ]:
            problem = pybop.FittingProblem(model, parameters, dataset)
            cost = pybop.SumofPower(problem, p=1)
            x = cost.parameters.true_value()
            optim = pybop.Optimisation(cost=cost)
            results = pybop.OptimisationResult(x=x, optim=optim)

            message = pybop.classify_using_hessian(results)
            assert message == (
                "The cost variation is too small to classify with certainty."
                " The cost is insensitive to a change of 1e-42 in R0_b [Ohm]."
            )

        message = pybop.classify_using_hessian(results, dx=[0.0001, 0.0001])
        assert message == (
            "The optimiser has located a minimum."
            " There may be a correlation between these parameters."
        )

        message = pybop.classify_using_hessian(results, cost_tolerance=1e-2)
        assert message == (
            "The cost variation is smaller than the cost tolerance: 0.01."
        )

        message = pybop.classify_using_hessian(results, dx=[1, 1])
        assert message == (
            "Classification cannot proceed due to infinite cost value(s)."
            " The result is near the upper bound of R0_a [Ohm]."
        )
