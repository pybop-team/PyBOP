import json

import numpy as np
import pybamm
import pytest

import pybop


class TestTheveninParameterisation:
    """
    A class to test a subset of optimisers on a simple model.
    """

    pytestmark = pytest.mark.integration

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ground_truth = np.clip(
            np.asarray([0.05, 0.05]) + np.random.normal(loc=0.0, scale=0.01, size=2),
            a_min=0.0,
            a_max=0.1,
        )

    @pytest.fixture
    def model(self):
        return pybamm.equivalent_circuit.Thevenin()

    @pytest.fixture
    def parameter_values(self, model):
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
        parameter_values.update(
            {
                "C1 [F]": 1000,
                "R0 [Ohm]": self.ground_truth[0],
                "R1 [Ohm]": self.ground_truth[1],
            }
        )
        return parameter_values

    @pytest.fixture
    def parameters(self):
        return pybop.Parameters(
            pybop.Parameter(
                "R0 [Ohm]",
                prior=pybop.Gaussian(0.05, 0.01),
                bounds=[1e-6, 0.1],
                transformation=pybop.LogTransformation(),
            ),
            pybop.Parameter(
                "R1 [Ohm]",
                prior=pybop.Gaussian(0.05, 0.01),
                bounds=[1e-6, 0.1],
                transformation=pybop.LogTransformation(),
            ),
        )

    @pytest.fixture
    def dataset(self, model, parameter_values):
        return self.get_data(model, parameter_values)

    @pytest.mark.parametrize(
        "cost_class",
        [pybop.RootMeanSquaredError, pybop.SumSquaredError],
    )
    @pytest.mark.parametrize(
        "optimiser, method",
        [
            (pybop.SciPyMinimize, "SLSQP"),
            (pybop.SciPyMinimize, "trust-constr"),
            (pybop.SciPyMinimize, "L-BFGS-B"),
            (pybop.SciPyMinimize, "COBYLA"),
            (pybop.GradientDescent, ""),
            (pybop.PSO, ""),
        ],
    )
    def test_optimisers_on_thevenin_model(
        self,
        model,
        parameter_values,
        parameters,
        dataset,
        cost_class,
        optimiser,
        method,
    ):
        simulator = pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            input_parameter_names=parameters.names,
            protocol=dataset,
        )
        # Define the cost to optimise
        cost = cost_class(dataset)
        problem = pybop.Problem(simulator, parameters, cost)

        x0 = problem.parameters.get_initial_values()
        if optimiser is pybop.SciPyMinimize:
            options = pybop.SciPyMinimizeOptions(maxiter=150)
        else:
            options = pybop.PintsOptions(max_iterations=150)
        if optimiser in [pybop.GradientDescent]:
            options.sigma0 = 2.5e-2
        elif method == "L-BFGS-B":
            options.sigma0 = 2.5e-2
            options.method = method
            options.jac = True
        else:
            options.sigma0 = 0.02
            options.method = method
        optim = optimiser(problem, options=options)

        if isinstance(optimiser, pybop.BasePintsOptimiser):
            optim.set_max_unchanged_iterations(iterations=35, absolute_tolerance=1e-5)

        initial_cost = optim.problem(optim.problem.parameters.get_initial_values())
        results = optim.run()

        # Assertions
        if not np.allclose(x0, self.ground_truth, atol=1e-5):
            if results.minimising:
                assert initial_cost > results.best_cost
            else:
                assert initial_cost < results.best_cost
        else:
            raise ValueError("Initial value is the same as the ground truth value.")
        np.testing.assert_allclose(results.x, self.ground_truth, atol=1.5e-2)

        if isinstance(optimiser, pybop.SciPyMinimize):
            assert results.scipy_result.success is True

    def get_data(self, model, parameter_values):
        experiment = pybamm.Experiment(
            [
                "Discharge at 0.5C for 6 minutes (12 seconds period)",
                "Rest for 20 seconds (4 seconds period)",
                "Charge at 0.5C for 6 minutes (12 seconds period)",
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
