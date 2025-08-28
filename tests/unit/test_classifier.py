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
        builder = pybop.Pybamm()
        builder.set_simulation(model)
        builder.set_dataset(dataset)
        builder.add_parameter(
            pybop.Parameter(
                "R0 [Ohm]",
                prior=pybop.Uniform(1e-4, 1e-3),
                bounds=[1e-4, 1e-3],
            )
        )
        builder.add_cost(
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]")
        )
        return builder.build()

    def test_classify_using_hessian_invalid(self, problem):
        options = pybop.GradientDescent.default_options()
        options.max_iterations = 1
        options.sigma = 1e-2
        results = pybop.GradientDescent(problem, options=options).run()

        with pytest.raises(
            ValueError,
            match="The function classify_using_hessian currently only works"
            " in the case of 2 parameters, and dx must have the same length as x.",
        ):
            pybop.classify_using_hessian(results, dx=[0.01])
