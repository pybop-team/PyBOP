import numpy as np
import pybamm
import pytest
from scipy import stats

import pybop


class TestClassifier:
    """
    A class to test the classification of different optimisation results.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def problem(self):
        model = pybamm.equivalent_circuit.Thevenin()
        experiment = pybamm.Experiment(
            [
                "Discharge at 0.5C for 2 minutes (4 seconds period)",
                "Charge at 0.5C for 2 minutes (4 seconds period)",
            ]
        )
        solution = pybamm.Simulation(model, experiment=experiment).solve()
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data,
            }
        )

        parameter_values = model.default_parameter_values
        parameter_values.update(
            {
                "R0 [Ohm]": pybop.ParameterDistribution(
                    distribution=stats.Uniform(a=0.001, b=0.1)
                )
            }
        )
        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )
        cost = pybop.SumSquaredError(dataset)
        return pybop.Problem(simulator, cost)

    def test_classify_using_hessian_invalid(self, problem):
        optim = pybop.XNES(problem)
        logger = pybop.Logger(minimising=True)
        logger.iteration = 1
        logger.extend_log(
            x_search=[np.asarray([1e-3])],
            x_model=[np.asarray([1e-3])],
            cost=[problem(np.asarray([1e-3]))],
        )
        result = pybop.OptimisationResult(optim=optim, logger=logger, time=1.0)

        with pytest.raises(
            ValueError,
            match="The function classify_using_hessian currently only works"
            " in the case of 2 parameters, and dx must have the same length as x.",
        ):
            pybop.classify_using_hessian(result)
