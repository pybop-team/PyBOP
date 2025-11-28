import numpy as np
import pybamm
import pytest

import pybop

# Test configuration constants
TIME_POINTS = 20
TIME_MAX = 20
FREQ_POINTS = 30
TEST_PARAM_VALUES = np.asarray([1, 2])
RELATIVE_TOLERANCE = 5e-5
ABSOLUTE_TOLERANCE = 5e-5

# Parameter configurations
EXPONENTIAL_DECAY_PARAMS = [("k", 2), ("y0", 2)]


class TestDecayModel:
    """
    A class to test the exponential decay model.
    """

    pytestmark = pytest.mark.integration

    @pytest.fixture(scope="module")
    def model_config(self):
        """Shared model configuration to avoid repeated initialisation."""
        model = pybop.ExponentialDecayModel()
        return {
            "model": model,
            "parameter_values": model.default_parameter_values,
            "solver": pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6),
        }

    @pytest.fixture(scope="module")
    def dataset(self, model_config):
        """Generate dataset"""
        sol = pybamm.Simulation(
            model_config["model"],
            parameter_values=model_config["parameter_values"],
            solver=model_config["solver"],
        ).solve(t_eval=np.linspace(0, TIME_MAX, TIME_POINTS))

        return pybop.Dataset({"Time [s]": sol.t, "y_0": sol["y_0"].data})

    @pytest.fixture(scope="module")
    def parameters(self):
        """Create parameter objects for reuse."""
        return {
            param_name: pybop.ParameterInfo(initial_value=val)
            for param_name, val in EXPONENTIAL_DECAY_PARAMS
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

        return value1, value2

    def test_decay_model(self, dataset, model_config, parameters):
        """Test decay model with voltage-based cost functions."""
        parameter_values = model_config["parameter_values"]
        parameter_values.update(parameters)
        simulator = pybop.pybamm.Simulator(
            model_config["model"],
            parameter_values=parameter_values,
            solver=model_config["solver"],
            protocol=dataset,
        )
        cost_1 = pybop.SumSquaredError(dataset, target=["y_0"])
        cost_2 = pybop.MeanAbsoluteError(dataset, target=["y_0"])
        cost = pybop.WeightedCost(cost_1, cost_2)
        problem = pybop.Problem(simulator, cost)

        # Test parameter sensitivity
        initial_params = problem.parameters.get_initial_values()
        initial_inputs = problem.parameters.to_dict(initial_params)
        example_inputs = problem.parameters.to_dict(TEST_PARAM_VALUES)
        value1, value2 = self.assert_parameter_sensitivity(
            problem, initial_inputs, example_inputs
        )

        # Test sensitivity computation consistency
        value1_sens, grad1 = problem.evaluate(
            initial_inputs, calculate_sensitivities=True
        ).get_values()
        value2_sens, grad2 = problem.evaluate(
            example_inputs, calculate_sensitivities=True
        ).get_values()

        # Validate gradient shape and value consistency
        assert grad1.shape == (
            1,
            len(problem.parameters),
        ), f"Gradient shape mismatch: {grad1.shape}"
        assert grad2.shape == (
            1,
            len(problem.parameters),
        ), f"Gradient shape mismatch: {grad2.shape}"

        np.testing.assert_allclose(
            value1_sens, value1, atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE
        )
        np.testing.assert_allclose(
            value2_sens, value2, atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE
        )
