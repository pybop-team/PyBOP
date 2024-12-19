import numpy as np
import pytest

import pybop


class TestClassifier:
    """
    A class to test the classification of different optimisation results.
    """

    @pytest.fixture
    def problem(self):
        model = pybop.empirical.Thevenin()
        experiment = pybop.Experiment(
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

    @pytest.mark.unit
    def test_classify_using_Hessian_invalid(self, problem):
        cost = pybop.SumSquaredError(problem)
        optim = pybop.Optimisation(cost=cost)
        x = np.asarray([0.001])
        results = pybop.OptimisationResult(x=x, optim=optim)

        with pytest.raises(
            ValueError,
            match="The function classify_using_Hessian currently only works"
            " in the case of 2 parameters, and dx must have the same length as x.",
        ):
            pybop.classify_using_Hessian(results)
