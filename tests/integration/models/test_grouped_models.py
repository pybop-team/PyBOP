import numpy as np
import pybamm
import pytest

import pybop

# Test configuration constants
TIME_POINTS = 20
TIME_MAX = 100
FREQ_POINTS = 30
TEST_PARAM_VALUES = np.asarray([550, 550])
RELATIVE_TOLERANCE = 1e-5
ABSOLUTE_TOLERANCE = 1e-5

# Parameter configurations
CHARGE_TRANSFER_PARAMS = [
    ("Positive electrode charge transfer time scale [s]", 500),
    ("Negative electrode charge transfer time scale [s]", 500),
]


class TestGroupedModels:
    """
    A class to test the pybop grouped-parameter battery models.
    """

    pytestmark = pytest.mark.integration

    @pytest.fixture(
        params=[
            pybop.lithium_ion.GroupedSPM(),
            pybop.lithium_ion.GroupedSPM(options={"surface form": "differential"}),
            pybop.lithium_ion.GroupedSPMe(),
            pybop.lithium_ion.GroupedSPMe(options={"surface form": "differential"}),
        ],
        scope="module",
    )
    def model_config(self, request):
        """Shared model configuration to avoid repeated initialisation."""
        model = request.param
        parameter_values = model.default_parameter_values

        return {
            "model": model,
            "parameter_values": parameter_values,
            "solver": model.default_solver,
        }

    @pytest.fixture(scope="module")
    def dataset(self, model_config):
        """Generate dataset with optimised simulation."""
        sim = pybamm.Simulation(
            model_config["model"],
            parameter_values=model_config["parameter_values"],
            solver=model_config["solver"],
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

    @pytest.fixture(scope="module")
    def eis_dataset(self):
        """EIS dataset with pre-allocated arrays."""
        frequencies = np.logspace(-4.5, 5, FREQ_POINTS)
        zeros = np.zeros(FREQ_POINTS)

        return pybop.Dataset(
            {
                "Frequency [Hz]": frequencies,
                "Current function [A]": zeros,
                "Impedance": zeros,
            }
        )

    @pytest.fixture(scope="module")
    def parameters(self):
        """Create parameter objects for reuse."""
        return [
            pybop.Parameter(param_name, initial_value=val)
            for param_name, val in CHARGE_TRANSFER_PARAMS
        ]

    def create_pybamm_builder(self, dataset, model_config, parameters):
        """Factory function for creating Pybamm builders."""
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model_config["model"], parameter_values=model_config["parameter_values"]
        )

        for parameter in parameters:
            builder.add_parameter(parameter)

        return builder

    def create_eis_builder(self, eis_dataset, model_config, parameters):
        """Factory function for creating EIS builders."""
        model = model_config["model"]
        builder = pybop.builders.PybammEIS()
        builder.set_dataset(eis_dataset)
        builder.set_simulation(model, parameter_values=model_config["parameter_values"])

        for parameter in parameters:
            builder.add_parameter(parameter)

        return builder

    def assert_parameter_sensitivity(
        self, problem, initial_params, test_params, tolerance=RELATIVE_TOLERANCE
    ):
        """Reusable assertion for parameter sensitivity."""
        value1 = problem.run(initial_params)
        value2 = problem.run(test_params)

        relative_change = abs((value1 - value2) / value1)
        assert relative_change > tolerance, (
            f"Parameter change insufficient: {relative_change}"
        )

        return value1, value2

    def test_voltage_fitting(self, dataset, model_config, parameters):
        """Test grouped SPMe model with voltage-based cost functions."""
        # Build problem with multiple cost functions
        builder = self.create_pybamm_builder(dataset, model_config, parameters)
        builder.add_cost(pybop.costs.pybamm.SumSquaredError("Voltage [V]"))
        builder.add_cost(pybop.costs.pybamm.MeanAbsoluteError("Voltage [V]"))

        fitting_problem = builder.build()

        # Test parameter sensitivity
        initial_params = fitting_problem.params.get_initial_values()

        value1, value2 = self.assert_parameter_sensitivity(
            fitting_problem, initial_params, TEST_PARAM_VALUES
        )

        # Test sensitivity computation consistency
        value1_sens, grad1 = fitting_problem.run_with_sensitivities(initial_params)
        value2_sens, grad2 = fitting_problem.run_with_sensitivities(TEST_PARAM_VALUES)

        # Validate gradient shape and value consistency
        assert grad1.shape == (len(parameters),), (
            f"Gradient shape mismatch: {grad1.shape}"
        )
        assert grad2.shape == (len(parameters),), (
            f"Gradient shape mismatch: {grad2.shape}"
        )

        np.testing.assert_allclose(
            value1_sens, value1, atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE
        )
        np.testing.assert_allclose(
            value2_sens, value2, atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE
        )

    def test_eis_fitting(self, eis_dataset, model_config, parameters):
        builder = self.create_eis_builder(eis_dataset, model_config, parameters)

        builder.add_cost(pybop.MeanSquaredError(weighting="equal"))

        eis_problem = builder.build()
        assert eis_problem is not None, "EIS problem build failed"

        # Test parameter sensitivity with different values
        initial_params = eis_problem.params.get_initial_values()

        self.assert_parameter_sensitivity(
            eis_problem, TEST_PARAM_VALUES, initial_params
        )
