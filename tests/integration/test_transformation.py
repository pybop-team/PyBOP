import itertools

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
            np.asarray([1e-3, 2e-4]) + np.random.normal(loc=0.0, scale=2e-5, size=2),
            a_min=1e-5,
            a_max=2e-3,
        )

    @pytest.fixture
    def parameter_values(self, model):
        parameter_values = model.default_parameter_values
        parameter_values.update(
            {
                "R0 [Ohm]": self.ground_truth[0],
                "R1 [Ohm]": self.ground_truth[1],
            }
        )
        return parameter_values

    @pytest.fixture
    def parameters(self, transformation_r0, transformation_r1):
        return [
            pybop.Parameter(
                "R0 [Ohm]",
                prior=pybop.Gaussian(1e-3, 5e-4),
                transformation=transformation_r0,
                initial_value=pybop.Uniform(1e-4, 1.5e-3).rvs()[0],
                bounds=[1e-5, 3e-3],
            ),
            pybop.Parameter(
                "R1 [Ohm]",
                prior=pybop.Gaussian(2e-4, 5e-5),
                transformation=transformation_r1,
                initial_value=pybop.Uniform(1e-5, 4e-4).rvs()[0],
                bounds=[1e-5, 1e-3],
            ),
        ]

    @pytest.fixture
    def model(self):
        return pybamm.equivalent_circuit.Thevenin()

    @pytest.fixture
    def dataset(self, model, parameter_values):
        t_eval = np.linspace(0, 100, 20)
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

        options = pybop.PintsOptions()
        options.sigma = 2e-2
        options.max_iterations = 50
        options.maximum_iterations = 20
        optim = optimiser(problem)
        results = optim.run()

        # Assertions
        if np.allclose(x0, self.ground_truth, atol=1e-5):
            raise AssertionError("Initial guess is too close to ground truth")

        assert results.initial_cost > results.best_cost
        np.testing.assert_allclose(results.x, self.ground_truth, atol=1.5e-2)
