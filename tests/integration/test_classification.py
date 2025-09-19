import json

import numpy as np
import pybamm
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
        self.ground_truth = request.param
        return pybop.Parameters(
            pybop.Parameter(
                "R0 [Ohm]",
                prior=pybop.Gaussian(0.05, 0.01),
                bounds=[0.02, 0.08],
            ),
            pybop.Parameter(
                "R1 [Ohm]",
                prior=pybop.Gaussian(0.05, 0.01),
                bounds=[0.02, 0.08],
            ),
        )

    @pytest.fixture
    def model(self):
        return pybamm.equivalent_circuit.Thevenin()

    @pytest.fixture
    def parameter_values(self, model, parameters):
        with open("examples/parameters/initial_ecm_parameters.json") as file:
            parameter_values = pybamm.ParameterValues(json.load(file))
        parameter_values.update(
            {
                "Open-circuit voltage [V]": model.default_parameter_values[
                    "Open-circuit voltage [V]"
                ]
            },
            check_already_exists=False,
        )
        parameter_values.update({"C1 [F]": 1000})
        parameter_values.update(parameters.to_dict(self.ground_truth))
        return parameter_values

    @pytest.fixture
    def dataset(self, model, parameter_values):
        experiment = pybamm.Experiment(
            [
                "Discharge at 0.5C for 2 minutes (4 seconds period)",
                "Charge at 0.5C for 2 minutes (4 seconds period)",
            ]
        )
        solution = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        ).solve()
        return pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data,
            }
        )

    @pytest.fixture
    def problem(self, model, parameter_values, parameters, dataset):
        simulator = pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            input_parameter_names=parameters.names,
            protocol=dataset,
        )
        return pybop.FittingProblem(simulator, parameters, dataset)

    def test_classify_using_hessian(self, problem):
        cost = pybop.RootMeanSquaredError(problem)
        x = self.ground_truth
        bounds = cost.parameters.get_bounds()
        x0 = np.clip(x, bounds["lower"], bounds["upper"])
        optim = pybop.XNES(cost=cost)
        logger = pybop.Logger(minimising=cost.minimising)
        logger.iteration = 1
        logger.extend_log(x_search=[x0], x_model=[x0], cost=[cost(x0)])
        results = pybop.OptimisationResult(optim=optim, logger=logger, time=1.0)

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
            optim = pybop.XNES(cost=cost)
            logger = pybop.Logger(minimising=cost.minimising)
            logger.iteration = 1
            logger.extend_log(x_search=[x], x_model=[x], cost=[cost(x)])
            results = pybop.OptimisationResult(optim=optim, logger=logger, time=1.0)

            message = pybop.classify_using_hessian(results)
            assert message == "The optimiser has located a maximum."

        # message = pybop.classify_using_hessian(results)
        # assert message == "The optimiser has located a saddle point."

    def test_insensitive_classify_using_hessian(self, model, parameter_values):
        param_R0_a = pybop.Parameter("R0_a [Ohm]", bounds=[0, 0.002])
        param_R0_b = pybop.Parameter("R0_b [Ohm]", bounds=[-0.001, 0.001])
        parameter_values.update(
            {"R0_a [Ohm]": 0.001, "R0_b [Ohm]": 0},
            check_already_exists=False,
        )
        parameter_values.update(
            {"R0 [Ohm]": Parameter("R0_a [Ohm]") + Parameter("R0_b [Ohm]")},
        )

        experiment = pybamm.Experiment(
            ["Discharge at 0.5C for 2 minutes (4 seconds period)"]
        )
        solution = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        ).solve()
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data,
            }
        )

        parameters = pybop.Parameters(param_R0_a, param_R0_b)
        simulator = pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            input_parameter_names=parameters.names,
            protocol=dataset,
        )
        problem = pybop.FittingProblem(simulator, parameters, dataset)
        cost = pybop.SumOfPower(problem, p=1)
        x = [0.001, 0]
        optim = pybop.XNES(cost=cost)
        logger = pybop.Logger(minimising=cost.minimising)
        logger.iteration = 1
        logger.extend_log(x_search=[x], x_model=[x], cost=[cost(x)])
        results = pybop.OptimisationResult(optim=optim, logger=logger, time=1.0)

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
