import json

import numpy as np
import pybamm
import pybamm.models
import pytest

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
        return [
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
        ]

    @pytest.fixture
    def parameter_values(self, model):
        params = model.default_parameter_values
        with open("examples/parameters/initial_ecm_parameters.json") as f:
            new_params = json.load(f)
            for key, value in new_params.items():
                if key not in params:
                    continue
                params.update({key: value})
        params.update({"C1 [F]": 1000})
        return params

    @pytest.fixture
    def model(self):
        return pybamm.equivalent_circuit.Thevenin()

    @pytest.fixture
    def dataset(self, model, parameter_values, parameters):
        parameters = pybop.Parameters(parameters)
        parameter_values.update(parameters.to_dict(parameters.true_values()))
        experiment = pybop.Experiment(
            [
                "Discharge at 0.5C for 2 minutes (4 seconds period)",
                "Charge at 0.5C for 2 minutes (4 seconds period)",
            ]
        )

        solution = pybamm.Simulation(
            model, experiment=experiment, parameter_values=parameter_values
        ).solve()
        return pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data,
            }
        )

    @pytest.fixture
    def problem(self, model, parameters, parameter_values, dataset):
        builder = pybop.Pybamm()
        builder.set_simulation(model, parameter_values=parameter_values)
        builder.set_dataset(dataset)
        for p in parameters:
            builder.add_parameter(p)
        builder.add_cost(
            pybop.costs.pybamm.RootMeanSquaredError("Voltage [V]", "Voltage [V]")
        )
        return builder.build()

    def test_classify_using_hessian(self, problem):
        x = problem.params.true_value()
        bounds = problem.params.get_bounds()
        x0 = np.clip(x, bounds["lower"], bounds["upper"])
        problem.set_params(x0)
        final_cost = problem.run()
        bounds = problem.params.get_bounds()
        results = pybop.OptimisationResult(
            problem=problem,
            x=x0,
            final_cost=final_cost,
            n_iterations=1,
            n_evaluations=1,
            time=0.1,
        )

        if np.all(x == np.asarray([0.05, 0.05])):
            message = pybop.classify_using_hessian(problem, results)
            assert message == "The optimiser has located a minimum."
        elif np.all(x == np.asarray([0.1, 0.05])):
            message = pybop.classify_using_hessian(problem, results)
            assert message == (
                "The optimiser has not converged to a stationary point."
                " The result is near the upper bound of R0 [Ohm]."
            )
        elif np.all(x == np.asarray([0.05, 0.01])):
            message = pybop.classify_using_hessian(problem, results)
            assert message == (
                "The optimiser has not converged to a stationary point."
                " The result is near the lower bound of R1 [Ohm]."
            )
        else:
            raise Exception(f"Please add a check for these values: {x}")

    def test_insensitive_classify_using_hessian(self, parameter_values):
        param_R0_a = pybop.Parameter(
            "R0_a [Ohm]",
            bounds=[0, 0.002],
            initial_value=0.001,
            true_value=0.001,
        )
        param_R0_b = pybop.Parameter(
            "R0_b [Ohm]",
            bounds=[-1e-4, 1e-4],
            initial_value=0,
            true_value=0,
        )
        parameter_values.update(
            {"R0_a [Ohm]": 0.001, "R0_b [Ohm]": 0},
            check_already_exists=False,
        )
        parameter_values.update(
            {
                "R0 [Ohm]": pybamm.Parameter("R0_a [Ohm]")
                + pybamm.Parameter("R0_b [Ohm]")
            },
        )
        model = pybamm.equivalent_circuit.Thevenin()

        experiment = pybop.Experiment(
            ["Discharge at 0.5C for 2 minutes (4 seconds period)"]
        )
        solution = pybamm.Simulation(
            model,
            experiment=experiment,
            parameter_values=parameter_values,
        ).solve()
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data,
            }
        )

        for parameters in [
            [param_R0_b, param_R0_a],
            [param_R0_a, param_R0_b],
        ]:
            builder = pybop.Pybamm()
            builder.set_simulation(model, parameter_values=parameter_values)
            builder.set_dataset(dataset)
            for p in parameters:
                builder.add_parameter(p)
            builder.add_cost(
                pybop.costs.pybamm.SumOfPower("Voltage [V]", "Voltage [V]")
            )
            problem = builder.build()

            x = problem.params.true_value()
            problem.set_params(x)
            final_cost = problem.run()
            results = pybop.OptimisationResult(
                problem=problem,
                x=x,
                final_cost=final_cost,
                n_iterations=1,
                n_evaluations=1,
                time=0.1,
            )

            message = pybop.classify_using_hessian(problem, results)
            assert message == (
                "The cost variation is too small to classify with certainty."
                " The cost is insensitive to a change of 1e-42 in R0_b [Ohm]."
            )

            message = pybop.classify_using_hessian(
                problem, results, dx=[0.0001, 0.0001]
            )
            assert message == (
                "The optimiser has located a minimum."
                " There may be a correlation between these parameters."
            )

            message = pybop.classify_using_hessian(
                problem, results, cost_tolerance=1e-2
            )
            assert message == (
                "The cost variation is smaller than the cost tolerance: 0.01."
            )

            message = pybop.classify_using_hessian(problem, results, dx=[1, 1])
            assert message == (
                "Classification cannot proceed due to infinite cost value(s)."
                " The result is near the upper bound of R0_a [Ohm]."
            )
