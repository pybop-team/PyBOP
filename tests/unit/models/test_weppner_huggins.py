from typing import Any

import numpy as np
import pybamm
import pytest

import pybop

# Test configuration constants
TIME_POINTS = 20
TIME_MAX = 100
FREQ_POINTS = 30
TEST_PARAM_VALUES = np.asarray([0.6, 4e-14])
RELATIVE_TOLERANCE = 1e-5
ABSOLUTE_TOLERANCE = 1e-5

# Parameter configurations
WEPPNER_HUGGINS_PARAMS = [
    ("Positive electrode active material volume fraction", 0.518),
    ("Positive particle diffusivity [m2.s-1]", 1e-14),
]


class TestWeppnerHuggins:
    """
    A class to test the WeppnerHuggins class.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture(scope="session")
    def base_model_config(self):
        """Shared model configuration to avoid repeated initialisation."""
        model = pybop.lithium_ion.WeppnerHuggins()
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
                "Current function [A]": sol["Current [A]"].data[unique_indices],
                "Voltage [V]": sol["Voltage [V]"].data[unique_indices],
            }
        )

    @pytest.fixture
    def test_parameters(self):
        """Create parameter objects for reuse."""
        return [
            pybop.Parameter(param_name, initial_value=val)
            for param_name, val in WEPPNER_HUGGINS_PARAMS
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

    def test_weppner_huggin_build(self, dataset, base_model_config, test_parameters):
        """Test model with voltage-based cost functions."""
        builder = self.create_pybamm_builder(
            dataset, base_model_config, test_parameters
        )
        builder.add_cost(
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]")
        )
        builder.add_cost(
            pybop.costs.pybamm.MeanAbsoluteError("Voltage [V]", "Voltage [V]")
        )

        problem = builder.build()

        # Test parameter sensitivity
        initial_params = problem.params.get_initial_values()
        self.assert_parameter_sensitivity(problem, initial_params, TEST_PARAM_VALUES)
