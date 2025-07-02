from typing import Any

import numpy as np
import pybamm
import pytest

import pybop

# Test configuration constants
TIME_POINTS = 20
TIME_MAX = 100
FREQ_POINTS = 30
TEST_PARAM_VALUES = np.asarray([1, 1])
RELATIVE_TOLERANCE = 5e-5
ABSOLUTE_TOLERANCE = 5e-5

# Parameter configurations
EXPONENTIAL_DECAY_PARAMS = [("k", 2), ("y0", 2)]


class TestDecay:
    """
    A class to test the Exponential Decay class.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture(scope="session")
    def base_model_config(self):
        """Shared model configuration to avoid repeated initialisation."""
        model = pybop.ExponentialDecayModel()
        return {
            "model": model,
            "parameter_values": model.default_parameter_values,
            "solver": model.default_solver,
        }

    @pytest.fixture
    def dataset(self, base_model_config):
        """Generate dataset"""
        config = base_model_config
        sim = pybamm.Simulation(
            config["model"],
            parameter_values=config["parameter_values"],
            solver=config["solver"],
        )

        t_eval = np.linspace(0, TIME_MAX, TIME_POINTS)
        sol = sim.solve(t_eval=t_eval)

        # Filter discontinuities
        unique_indices = np.unique(sol.t, return_index=True)[1]

        return pybop.Dataset(
            {
                "Time [s]": sol.t[unique_indices],
                "y_0": sol["y_0"].data[unique_indices],
            }
        )

    @pytest.fixture
    def test_parameters(self):
        """Create parameter objects for reuse."""
        return [
            pybop.Parameter(param_name, initial_value=val)
            for param_name, val in EXPONENTIAL_DECAY_PARAMS
        ]

    def create_pybamm_builder(self, dataset, model_config: dict[str, Any], parameters):
        """Factory function for creating Pybamm builders."""
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model_config["model"], parameter_values=model_config["parameter_values"]
        )

        for parameter in parameters:
            builder.add_parameter(parameter)

        return builder

    def assert_parameter_sensitivity(
        self, problem, initial_params, test_params, tolerance=RELATIVE_TOLERANCE
    ):
        """Reusable assertion for parameter sensitivity."""
        problem.set_params(initial_params)
        value1 = problem.run()

        problem.set_params(test_params)
        value2 = problem.run()

        relative_change = abs((value1 - value2) / value1)
        assert relative_change > tolerance, (
            f"Parameter change insufficient: {relative_change}"
        )

        return value1, value2

    def test_decay_builder(self, dataset, base_model_config, test_parameters):
        """Test decay model with voltage-based cost functions."""
        builder = self.create_pybamm_builder(
            dataset, base_model_config, test_parameters
        )
        builder.add_cost(pybop.costs.pybamm.SumSquaredError("y_0", "y_0"))
        builder.add_cost(pybop.costs.pybamm.MeanAbsoluteError("y_0", "y_0"))

        problem = builder.build()

        # Test parameter sensitivity
        initial_params = problem.params.get_initial_values()

        value1, value2 = self.assert_parameter_sensitivity(
            problem, initial_params, TEST_PARAM_VALUES
        )

        # Test sensitivity computation consistency
        problem.set_params(initial_params)
        value1_sens, grad1 = problem.run_with_sensitivities()

        problem.set_params(TEST_PARAM_VALUES)
        value2_sens, grad2 = problem.run_with_sensitivities()

        # Validate gradient shape and value consistency
        assert grad1.shape == (len(test_parameters),), (
            f"Gradient shape mismatch: {grad1.shape}"
        )
        assert grad2.shape == (len(test_parameters),), (
            f"Gradient shape mismatch: {grad2.shape}"
        )

        np.testing.assert_allclose(
            value1_sens, value1, atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE
        )
        np.testing.assert_allclose(
            value2_sens, value2, atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE
        )
