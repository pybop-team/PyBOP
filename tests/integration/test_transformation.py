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
        self.ground_truth = np.clip(
            np.asarray([0.05, 0.05]) + np.random.normal(loc=0.0, scale=0.01, size=2),
            a_min=0.0,
            a_max=0.1,
        )

    @pytest.fixture
    def parameter_values(self, model):
        params = model.default_parameter_values
        with open("examples/parameters/initial_ecm_parameters.json") as f:
            new_params = json.load(f)
            for key, value in new_params.items():
                if key not in params:
                    continue
                params.update({key: value})
        params.update(
            {
                "C1 [F]": 1000,
                "R0 [Ohm]": self.ground_truth[0],
                "R1 [Ohm]": self.ground_truth[1],
            }
        )
        return params

    @pytest.fixture
    def parameters(self, transformation_r0, transformation_r1):
        return [
            pybop.Parameter(
                "R0 [Ohm]",
                prior=pybop.Gaussian(0.05, 0.02),
                bounds=[1e-4, 0.1],
                transformation=transformation_r0,
            ),
            pybop.Parameter(
                "R1 [Ohm]",
                prior=pybop.Gaussian(0.05, 0.02),
                bounds=[1e-4, 0.1],
                transformation=transformation_r1,
            ),
        ]

    @pytest.fixture
    def model(self):
        return pybamm.equivalent_circuit.Thevenin()

    @pytest.fixture
    def dataset(self, model, parameter_values):
        t_eval = np.linspace(0, 10, 100)
        solution = pybamm.Simulation(model, parameter_values=parameter_values).solve(
            t_eval=t_eval
        )
        return pybop.Dataset(
            {
                "Time [s]": solution.t,
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

    @pytest.mark.parametrize(
        "optimiser",
        [pybop.IRPropMin, pybop.CMAES],
        ids=["IRPropMin", "CMAES"],
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
        x0 = problem.params.get_initial_values()
        problem.set_params(x0)
        initial_cost = problem.run()

        optim = optimiser(problem)
        results = optim.run()

        # Assertions
        if np.allclose(x0, self.ground_truth, atol=1e-5):
            raise AssertionError("Initial guess is too close to ground truth")

        assert initial_cost > results.final_cost
        np.testing.assert_allclose(results.x, self.ground_truth, atol=1.5e-2)
