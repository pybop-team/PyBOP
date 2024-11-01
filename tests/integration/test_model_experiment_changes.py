import numpy as np
import pybamm
import pytest

import pybop


class TestModelAndExperimentChanges:
    """
    A class to test that different inputs return different outputs.
    """

    @pytest.fixture(
        params=[
            pybop.Parameters(
                pybop.Parameter(  # geometric parameter
                    "Negative particle radius [m]",
                    prior=pybop.Gaussian(6e-06, 0.1e-6),
                    bounds=[1e-6, 9e-6],
                    true_value=5.86e-6,
                    initial_value=5.86e-6,
                ),
            ),
            pybop.Parameters(
                pybop.Parameter(  # non-geometric parameter
                    "Positive particle diffusivity [m2.s-1]",
                    prior=pybop.Gaussian(3.43e-15, 1e-15),
                    bounds=[1e-15, 5e-15],
                    true_value=4e-15,
                    initial_value=4e-15,
                ),
            ),
        ]
    )
    def parameters(self, request):
        return request.param

    @pytest.fixture
    def solver(self):
        return pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6)

    @pytest.mark.integration
    def test_changing_experiment(self, parameters, solver):
        # Change the experiment and check that the results are different.

        parameter_set = pybop.ParameterSet.pybamm("Chen2020")
        parameter_set.update(parameters.as_dict("true"))
        initial_state = {"Initial SoC": 0.5}
        model = pybop.lithium_ion.SPM(parameter_set=parameter_set, solver=solver)

        t_eval = np.arange(0, 3600, 2)  # Default 1C discharge to cut-off voltage
        solution_1 = model.predict(initial_state=initial_state, t_eval=t_eval)
        cost_1 = self.final_cost(solution_1, model, parameters)

        experiment = pybop.Experiment(["Charge at 1C until 4.1 V (2 seconds period)"])
        solution_2 = model.predict(
            initial_state=initial_state,
            experiment=experiment,
            inputs=parameters.as_dict("true"),
        )
        cost_2 = self.final_cost(solution_2, model, parameters)

        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(
                solution_1["Voltage [V]"].data,
                solution_2["Voltage [V]"].data,
            )

        # The datasets are not corrupted so the costs should be zero
        np.testing.assert_allclose(cost_1, 0, atol=1e-5)
        np.testing.assert_allclose(cost_2, 0, atol=1e-5)

    @pytest.mark.integration
    def test_changing_model(self, parameters, solver):
        # Change the model and check that the results are different.

        parameter_set = pybop.ParameterSet.pybamm("Chen2020")
        parameter_set.update(parameters.as_dict("true"))
        initial_state = {"Initial SoC": 0.5}
        experiment = pybop.Experiment(["Charge at 1C until 4.1 V (2 seconds period)"])

        model = pybop.lithium_ion.SPM(parameter_set=parameter_set, solver=solver)
        solution_1 = model.predict(initial_state=initial_state, experiment=experiment)
        cost_1 = self.final_cost(solution_1, model, parameters)

        model = pybop.lithium_ion.SPMe(parameter_set=parameter_set, solver=solver)
        solution_2 = model.predict(initial_state=initial_state, experiment=experiment)
        cost_2 = self.final_cost(solution_2, model, parameters)

        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(
                solution_1["Voltage [V]"].data,
                solution_2["Voltage [V]"].data,
            )

        # The datasets are not corrupted so the costs should be zero
        np.testing.assert_allclose(cost_1, 0, atol=1e-5)
        np.testing.assert_allclose(cost_2, 0, atol=1e-5)

    def final_cost(self, solution, model, parameters):
        # Compute the cost corresponding to a particular solution
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data,
            }
        )
        signal = ["Voltage [V]"]
        problem = pybop.FittingProblem(model, parameters, dataset, signal=signal)
        cost = pybop.RootMeanSquaredError(problem)
        optim = pybop.PSO(cost)
        results = optim.run()
        return results.final_cost

    @pytest.mark.integration
    def test_multi_fitting_problem(self, solver):
        parameter_set = pybop.ParameterSet.pybamm("Chen2020")
        parameters = pybop.Parameter(
            "Negative electrode active material volume fraction",
            prior=pybop.Gaussian(0.68, 0.05),
            true_value=parameter_set[
                "Negative electrode active material volume fraction"
            ],
        )

        model_1 = pybop.lithium_ion.SPM(parameter_set=parameter_set, solver=solver)
        experiment_1 = pybop.Experiment(
            ["Discharge at 1C until 3 V (4 seconds period)"]
        )
        solution_1 = model_1.predict(experiment=experiment_1)
        dataset_1 = pybop.Dataset(
            {
                "Time [s]": solution_1["Time [s]"].data,
                "Current function [A]": solution_1["Current [A]"].data,
                "Voltage [V]": solution_1["Voltage [V]"].data,
            }
        )

        model_2 = pybop.lithium_ion.SPMe(
            parameter_set=parameter_set.copy(), solver=solver
        )
        experiment_2 = pybop.Experiment(
            ["Discharge at 3C until 3 V (4 seconds period)"]
        )
        solution_2 = model_2.predict(experiment=experiment_2)
        dataset_2 = pybop.Dataset(
            {
                "Time [s]": solution_2["Time [s]"].data,
                "Current function [A]": solution_2["Current [A]"].data,
                "Voltage [V]": solution_2["Voltage [V]"].data,
            }
        )

        # Define a problem for each dataset and combine them into one
        problem_1 = pybop.FittingProblem(model_1, parameters, dataset_1)
        problem_2 = pybop.FittingProblem(model_2, parameters, dataset_2)
        problem = pybop.MultiFittingProblem(problem_1, problem_2)
        cost = pybop.RootMeanSquaredError(problem)

        # Test with a gradient and non-gradient-based optimiser
        for optimiser in [pybop.SNES, pybop.IRPropMin]:
            optim = optimiser(cost)
            results = optim.run()
            np.testing.assert_allclose(results.x, parameters.true_value, atol=2e-5)
            np.testing.assert_allclose(results.final_cost, 0, atol=2e-5)
