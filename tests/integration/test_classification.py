import numpy as np
import pytest

import pybop
from examples.standalone.cost import StandaloneCost
from pybop._classification import classify_using_Hessian


class TestClassification:
    """
    A class to test the classification of different optimisation results.
    """

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
    def model(self, parameters):
        parameter_set = pybop.ParameterSet(
            json_path="examples/scripts/parameters/initial_ecm_parameters.json"
        )
        parameter_set.import_parameters()
        parameter_set.params.update({"C1 [F]": 1000})
        parameter_set.params.update(parameters.as_dict(parameters.true_value()))
        return pybop.empirical.Thevenin(parameter_set=parameter_set)

    @pytest.fixture
    def dataset(self, model):
        experiment = pybop.Experiment(
            [
                (
                    "Discharge at 0.5C for 2 minutes (4 seconds period)",
                    "Charge at 0.5C for 2 minutes (4 seconds period)",
                ),
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

    @pytest.mark.integration
    def test_classify_using_Hessian(self, problem):
        cost = pybop.RootMeanSquaredError(problem)
        optim = pybop.XNES(cost=cost)

        x = np.asarray(cost.parameters.true_value())
        bounds = cost.parameters.get_bounds()
        optim_x = np.clip(x, bounds["lower"], bounds["upper"])

        if np.all(x == np.asarray([0.05, 0.05])):
            message = classify_using_Hessian(optim, optim_x)
            assert message == "The optimiser has located a minimum."
        elif np.all(x == np.asarray([0.1, 0.05])):
            message = classify_using_Hessian(optim, optim_x)
            assert message == (
                "The optimiser has not converged to a stationary point."
                + " The result is near the upper bound of R0 [Ohm]."
            )
        elif np.all(x == np.asarray([0.05, 0.01])):
            message = classify_using_Hessian(optim, optim_x)
            assert message == (
                "The optimiser has not converged to a stationary point."
                + " The result is near the lower bound of R1 [Ohm]."
            )
        else:
            raise Exception("Please add a check for these values.")

        if np.all(x == np.asarray([0.05, 0.05])):
            cost = pybop.GaussianLogLikelihoodKnownSigma(problem, sigma0=0.002)
            optim = pybop.XNES(cost=cost)

            message = classify_using_Hessian(optim, x)
            assert message == "The optimiser has located a maximum."

    @pytest.mark.integration
    def test_classify_using_Hessian_invalid(self):
        cost = StandaloneCost()
        optim = pybop.XNES(cost=cost)
        optim.run()

        with pytest.raises(
            ValueError,
            match="The function classify_using_Hessian currently only works"
            + " in the case of 2 parameters.",
        ):
            classify_using_Hessian(optim)
