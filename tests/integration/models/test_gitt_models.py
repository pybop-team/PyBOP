import numpy as np
import pybamm
import pytest

import pybop

# Test configuration constants
TIME_POINTS = 20
TIME_MAX = 100
FREQ_POINTS = 30
TEST_PARAM_VALUES = np.asarray([0.6, 40])
RELATIVE_TOLERANCE = 1e-5
ABSOLUTE_TOLERANCE = 1e-5

# Parameter configurations
DIFFUSION_PARAMS = [
    ("Theoretical electrode capacity [A.s]", 10),
    ("Particle diffusion time scale [s]", 2000),
]


class TestGITTModels:
    """
    A class to test the pybop GITT models.
    """

    pytestmark = pytest.mark.integration

    @pytest.fixture(
        params=[
            pybop.lithium_ion.WeppnerHuggins(),
            pybop.lithium_ion.SPDiffusion(),
        ],
        scope="module",
    )
    def model_config(self, request):
        """Shared model configuration to avoid repeated initialisation."""
        model = request.param
        return {
            "model": model,
            "parameter_values": model.default_parameter_values,
            "solver": model.default_solver,
        }

    @pytest.fixture(scope="module")
    def dataset(self, model_config):
        """Generate dataset"""
        config = model_config
        t_eval = np.linspace(0, TIME_MAX, TIME_POINTS)
        solution = pybamm.Simulation(
            config["model"],
            parameter_values=config["parameter_values"],
            solver=config["solver"],
        ).solve(t_eval=t_eval)

        return pybop.Dataset(
            {
                "Time [s]": t_eval,
                "Current function [A]": solution["Current [A]"](t_eval),
                "Voltage [V]": solution["Voltage [V]"](t_eval),
            }
        )

    @pytest.fixture(scope="module")
    def parameters(self):
        """Create parameter objects for reuse."""
        return {
            param_name: pybop.Parameter(initial_value=val)
            for param_name, val in DIFFUSION_PARAMS
        }

    def assert_parameter_sensitivity(
        self, problem, initial_inputs, example_inputs, tolerance=RELATIVE_TOLERANCE
    ):
        """Reusable assertion for parameter sensitivity."""
        value1 = problem.evaluate(initial_inputs).values
        value2 = problem.evaluate(example_inputs).values

        relative_change = abs((value1 - value2) / value1)
        assert relative_change > tolerance, (
            f"Parameter change insufficient: {relative_change}"
        )

    def test_build(self, dataset, model_config, parameters):
        """Test model with voltage-based cost functions."""
        parameter_values = model_config["parameter_values"]
        parameter_values.update(parameters)
        simulator = pybop.pybamm.Simulator(
            model_config["model"],
            parameter_values=parameter_values,
            solver=model_config["solver"],
            protocol=dataset,
        )
        cost_1 = pybop.SumSquaredError(dataset)
        cost_2 = pybop.MeanAbsoluteError(dataset)
        cost = pybop.WeightedCost(cost_1, cost_2)
        problem = pybop.Problem(simulator, cost)

        # Test parameter sensitivity
        initial_params = problem.parameters.get_initial_values()
        initial_inputs = problem.parameters.to_dict(initial_params)
        example_inputs = problem.parameters.to_dict(TEST_PARAM_VALUES)
        self.assert_parameter_sensitivity(problem, initial_inputs, example_inputs)
