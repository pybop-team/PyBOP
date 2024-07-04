import numpy as np
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
                    "Positive electrode diffusivity [m2.s-1]",
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

    @pytest.mark.integration
    def test_changing_experiment(self, parameters):
        # Change the experiment and check that the results are different.

        parameter_set = pybop.ParameterSet.pybamm("Chen2020")
        init_soc = 0.5
        model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

        t_eval = np.arange(0, 3600, 2)  # Default 1C discharge to cut-off voltage
        solution_1 = model.predict(init_soc=init_soc, t_eval=t_eval)
        cost_1 = self.final_cost(solution_1, model, parameters, init_soc)

        experiment = pybop.Experiment(["Charge at 1C until 4.1 V (2 seconds period)"])
        solution_2 = model.predict(
            init_soc=init_soc,
            experiment=experiment,
            inputs=parameters.as_dict("true"),
        )
        cost_2 = self.final_cost(solution_2, model, parameters, init_soc)

        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(
                solution_1["Voltage [V]"].data,
                solution_2["Voltage [V]"].data,
            )

        # The datasets are not corrupted so the costs should be zero
        np.testing.assert_allclose(cost_1, 0, atol=1e-5)
        np.testing.assert_allclose(cost_2, 0, atol=1e-5)

    @pytest.mark.integration
    def test_changing_model(self, parameters):
        # Change the model and check that the results are different.

        parameter_set = pybop.ParameterSet.pybamm("Chen2020")
        init_soc = 0.5
        experiment = pybop.Experiment(["Charge at 1C until 4.1 V (2 seconds period)"])

        model = pybop.lithium_ion.SPM(parameter_set=parameter_set)
        solution_1 = model.predict(init_soc=init_soc, experiment=experiment)
        cost_1 = self.final_cost(solution_1, model, parameters, init_soc)

        model = pybop.lithium_ion.SPMe(parameter_set=parameter_set)
        solution_2 = model.predict(init_soc=init_soc, experiment=experiment)
        cost_2 = self.final_cost(solution_2, model, parameters, init_soc)

        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(
                solution_1["Voltage [V]"].data,
                solution_2["Voltage [V]"].data,
            )

        # The datasets are not corrupted so the costs should be zero
        np.testing.assert_allclose(cost_1, 0, atol=1e-5)
        np.testing.assert_allclose(cost_2, 0, atol=1e-5)

    def final_cost(self, solution, model, parameters, init_soc):
        # Compute the cost corresponding to a particular solution
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data,
            }
        )
        signal = ["Voltage [V]"]
        problem = pybop.FittingProblem(
            model, parameters, dataset, signal=signal, init_soc=init_soc
        )
        cost = pybop.RootMeanSquaredError(problem)
        optim = pybop.PSO(cost)
        x, final_cost = optim.run()
        return final_cost
