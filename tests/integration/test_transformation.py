import itertools
import json

import numpy as np
import pybamm
import pytest

import pybop


def transformation_id(val):
    """Create a readable name for each transformation."""
    if isinstance(val, pybop.IdentityTransformation):
        return "Identity"
    elif isinstance(val, pybop.UnitHyperCube):
        return "UnitHyperCube"
    elif isinstance(val, pybop.LogTransformation):
        return "Log"
    else:
        return str(val)


class TestTransformation:
    """
    A class for integration tests of the transformation methods.
    """

    pytestmark = pytest.mark.integration

    @pytest.fixture(autouse=True)
    def setup(self):
        self.sigma0 = 2e-3
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
    def parameters(self, transformation_r0, transformation_r1):
        return {
            "R0 [Ohm]": pybop.Parameter(
                "R0 [Ohm]",
                prior=pybop.Gaussian(0.05, 0.02),
                bounds=[1e-4, 0.1],
                transformation=transformation_r0,
            ),
            "R1 [Ohm]": pybop.Parameter(
                "R1 [Ohm]",
                prior=pybop.Gaussian(0.05, 0.02),
                bounds=[1e-4, 0.1],
                transformation=transformation_r1,
            ),
        }

    def noisy(self, data, sigma):
        return data + np.random.normal(0, sigma, len(data))

    @pytest.fixture(
        params=[
            pybop.GaussianLogLikelihood,
            pybop.RootMeanSquaredError,
            pybop.SumSquaredError,
            pybop.LogPosterior,
        ]
    )
    def cost_class(self, request):
        return request.param

    @pytest.fixture
    def problem(self, model, parameter_values, parameters, cost_class):
        parameter_values.set_initial_state(0.6)
        dataset = self.get_data(model, parameter_values)

        parameter_values.update(parameters)
        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )

        # Construct the cost
        if cost_class is pybop.LogPosterior:
            likelihood = pybop.GaussianLogLikelihood(dataset, sigma0=self.sigma0)
            cost = cost_class(likelihood)
        else:
            cost = cost_class(dataset)
        return pybop.Problem(simulator, cost)

    @pytest.mark.parametrize(
        "optimiser",
        [pybop.IRPropMin, pybop.CMAES, pybop.SciPyDifferentialEvolution],
        ids=["IRPropMin", "CMAES", "SciPyDifferentialEvolution"],
    )
    @pytest.mark.parametrize(
        "transformation_r0, transformation_r1",
        list(
            itertools.product(
                [
                    pybop.IdentityTransformation(),
                    pybop.UnitHyperCube(0.001, 0.1),
                    pybop.LogTransformation(),
                ],
                repeat=2,
            )
        ),
        ids=lambda val: f"{transformation_id(val)}",
    )
    def test_thevenin_transformation(self, optimiser, problem):
        x0 = problem.parameters.get_initial_values()
        if optimiser is pybop.SciPyDifferentialEvolution:
            options = pybop.SciPyDifferentialEvolutionOptions(
                maxiter=50,
                popsize=3,
            )
        else:
            options = pybop.PintsOptions(
                max_iterations=150,
                max_unchanged_iterations=45,
            )
        optim = optimiser(problem, options=options)

        initial_cost = optim.problem.evaluate(x0)
        results = optim.run()

        # Add sigma0 to ground truth for GaussianLogLikelihood
        if isinstance(problem.cost, pybop.GaussianLogLikelihood | pybop.LogPosterior):
            self.ground_truth = np.concatenate(
                (self.ground_truth, np.asarray([self.sigma0]))
            )

        # Assertions
        if np.allclose(x0, self.ground_truth, atol=1e-5):
            raise AssertionError("Initial guess is too close to ground truth")

        assert (
            (initial_cost > results.best_cost)
            if results.minimising
            else (initial_cost < results.best_cost)
        )
        np.testing.assert_allclose(results.x, self.ground_truth, atol=1.5e-2)

    def get_data(self, model, parameter_values):
        experiment = pybamm.Experiment(
            [
                "Rest for 10 seconds (2 second period)",
                "Discharge at 0.5C for 6 minutes (12 second period)",
            ]
        )
        solution = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        ).solve()
        return pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": self.noisy(solution["Voltage [V]"].data, self.sigma0),
            }
        )
