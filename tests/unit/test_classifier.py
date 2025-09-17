import numpy as np
import pybamm
import pytest

import pybop


class TestClassifier:
    """
    A class to test the classification of different optimisation results.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def problem(self):
        model = pybop.empirical.Thevenin()
        experiment = pybamm.Experiment(
            [
                "Discharge at 0.5C for 2 minutes (4 seconds period)",
                "Charge at 0.5C for 2 minutes (4 seconds period)",
            ]
        )
        solution = model.predict(experiment=experiment)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data,
            }
        )
        parameters = pybop.Parameters(
            pybop.Parameter(
                "R0 [Ohm]",
                prior=pybop.Uniform(0.001, 0.1),
                bounds=[1e-4, 0.1],
            ),
        )
        return pybop.FittingProblem(model, parameters, dataset)

    def test_classify_using_hessian_invalid(self, problem):
        cost = pybop.SumSquaredError(problem)
        optim = pybop.XNES(cost=cost)
        logger = pybop.Logger(minimising=True)
        logger.iteration = 1
        logger.extend_log(
            x_search=[np.asarray([1e-3])],
            x_model=[np.asarray([1e-3])],
            cost=[cost(np.asarray([1e-3]))],
        )
        result = pybop.OptimisationResult(optim=optim, logger=logger, time=1.0)

        with pytest.raises(
            ValueError,
            match="The function classify_using_hessian currently only works"
            " in the case of 2 parameters, and dx must have the same length as x.",
        ):
            pybop.classify_using_hessian(result)
