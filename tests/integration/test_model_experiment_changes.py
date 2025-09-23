import numpy as np
import pybamm
import pytest

import pybop


class TestModelAndExperimentChanges:
    """
    A class to test that different inputs return different outputs.
    """

    pytestmark = pytest.mark.integration

    @pytest.fixture(
        params=[
            [
                pybop.Parameters(
                    pybop.Parameter(  # geometric parameter
                        "Negative particle radius [m]",
                        prior=pybop.Gaussian(6e-06, 0.1e-6),
                        bounds=[1e-6, 9e-6],
                        initial_value=5.86e-6,
                    ),
                ),
                [5.86e-6],
            ],
            [
                pybop.Parameters(
                    pybop.Parameter(  # non-geometric parameter
                        "Positive particle diffusivity [m2.s-1]",
                        prior=pybop.Gaussian(3.43e-15, 1e-15),
                        bounds=[1e-15, 5e-15],
                        initial_value=4e-15,
                    ),
                ),
                [4e-15],
            ],
        ]
    )
    def parameters_and_value(self, request):
        return request.param

    @pytest.fixture
    def solver(self):
        return pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6)

    @pytest.mark.integration
    def test_changing_experiment(self, parameters_and_value, solver):
        parameters, value = parameters_and_value
        # Change the experiment and check that the results are different.

        parameter_values = pybamm.ParameterValues("Chen2020")
        inputs = parameters.to_dict(value)
        parameter_values.update(inputs)
        initial_state = {"Initial SoC": 0.5}
        model = pybamm.lithium_ion.SPM()

        t_eval = np.arange(0, 3600, 2)  # Default 1C discharge to cut-off voltage
        simulator_1 = pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            input_parameter_names=parameters.names,
            initial_state=initial_state,
            protocol=t_eval,
            solver=solver,
        )
        solution_1 = simulator_1.solve(inputs)
        cost_1 = self.final_cost(simulator_1, parameters, solution_1)

        experiment = pybamm.Experiment(["Charge at 1C until 4.1 V (2 seconds period)"])
        simulator_2 = pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            input_parameter_names=parameters.names,
            initial_state=initial_state,
            protocol=experiment,
        )
        solution_2 = simulator_2.solve(inputs)
        cost_2 = self.final_cost(simulator_2, parameters, solution_2)

        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(
                solution_1["Voltage [V]"].data,
                solution_2["Voltage [V]"].data,
            )

        # The datasets are not corrupted so the costs should be zero
        np.testing.assert_allclose(cost_1, 0, atol=1e-5)
        np.testing.assert_allclose(cost_2, 0, atol=1e-5)

    def test_changing_model(self, parameters_and_value, solver):
        # Change the model and check that the results are different.

        parameter_values = pybamm.ParameterValues("Chen2020")
        parameters, value = parameters_and_value
        inputs = parameters.to_dict(value)
        parameter_values.update(inputs)
        initial_state = {"Initial SoC": 0.5}
        experiment = pybamm.Experiment(["Charge at 1C until 4.1 V (30 seconds period)"])

        model = pybamm.lithium_ion.SPM()
        simulator_1 = pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            input_parameter_names=parameters.names,
            protocol=experiment,
            initial_state=initial_state,
            solver=solver,
        )
        solution_1 = simulator_1.solve(inputs)
        cost_1 = self.final_cost(simulator_1, parameters, solution_1)

        model = pybamm.lithium_ion.SPMe()
        simulator_2 = pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            input_parameter_names=parameters.names,
            protocol=experiment,
            initial_state=initial_state,
            solver=solver,
        )
        solution_2 = simulator_2.solve(inputs)
        cost_2 = self.final_cost(simulator_2, parameters, solution_2)

        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(
                solution_1["Voltage [V]"].data,
                solution_2["Voltage [V]"].data,
            )

        # The datasets are not corrupted so the costs should be zero
        np.testing.assert_allclose(cost_1, 0, atol=1e-5)
        np.testing.assert_allclose(cost_2, 0, atol=1e-5)

    def final_cost(self, simulator, parameters, solution):
        # Compute the cost corresponding to a particular solution
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data,
            }
        )
        cost = pybop.RootMeanSquaredError(dataset)
        problem = pybop.FittingProblem(simulator, parameters, cost)
        optim = pybop.NelderMead(problem)
        results = optim.run()
        return results.best_cost

    def test_multi_fitting_problem(self, solver):
        parameter_values = pybamm.ParameterValues("Chen2020")
        ground_truth = parameter_values[
            "Negative electrode active material volume fraction"
        ]
        parameters = pybop.Parameters(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.68, 0.05),
            )
        )
        parameter_values.update(parameters.to_dict(ground_truth))

        model_1 = pybamm.lithium_ion.SPM()
        experiment_1 = pybamm.Experiment(
            ["Discharge at 0.5C for 5 minutes (10 seconds period)"]
        )
        dataset_1 = self.get_data(model_1, parameter_values, experiment_1, solver)
        simulator_1 = pybop.pybamm.Simulator(
            model_1,
            parameter_values=parameter_values,
            input_parameter_names=parameters.names,
            protocol=dataset_1,
            solver=solver,
        )

        model_2 = pybamm.lithium_ion.SPMe()
        experiment_2 = pybamm.Experiment(
            ["Discharge at 1C for 3 minutes (10 seconds period)"]
        )
        dataset_2 = self.get_data(model_2, parameter_values, experiment_2, solver)
        simulator_2 = pybop.pybamm.Simulator(
            model_2,
            parameter_values=parameter_values,
            input_parameter_names=parameters.names,
            protocol=dataset_2,
            solver=solver,
        )

        # Define a problem for each dataset and combine them into one
        cost_1 = pybop.RootMeanSquaredError(dataset_1)
        cost_2 = pybop.RootMeanSquaredError(dataset_2)
        problem_1 = pybop.FittingProblem(simulator_1, parameters, cost_1)
        problem_2 = pybop.FittingProblem(simulator_2, parameters, cost_2)
        problem = pybop.MetaProblem(problem_1, problem_2)

        # Test with a gradient and non-gradient-based optimiser
        for optimiser in [pybop.SNES, pybop.IRPropMin]:
            options = pybop.PintsOptions(
                sigma=0.05, max_iterations=100, max_unchanged_iterations=30
            )
            optim = optimiser(problem, options=options)
            results = optim.run()
            np.testing.assert_allclose(results.x, ground_truth, atol=2e-5)
            np.testing.assert_allclose(results.best_cost, 0, atol=3e-5)

    def get_data(self, model, parameter_values, experiment, solver):
        solution = pybamm.Simulation(
            model,
            parameter_values=parameter_values,
            experiment=experiment,
            solver=solver,
        ).solve()
        return pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data,
            }
        )
