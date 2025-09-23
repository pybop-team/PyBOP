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
        t_eval = np.linspace(0, TIME_MAX, TIME_POINTS)
        solution = pybamm.Simulation(
            model_config["model"],
            parameter_values=model_config["parameter_values"],
            solver=model_config["solver"],
        ).solve(t_eval=t_eval)

        return pybop.Dataset(
            {
                "Time [s]": t_eval,
                "Current function [A]": solution["Current [A]"](t_eval),
                "Voltage [V]": solution["Voltage [V]"](t_eval),
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
            },
            domain="Frequency [Hz]",
        )

    @pytest.fixture(scope="module")
    def parameters(self):
        """Create parameter objects for reuse."""
        return pybop.Parameters(
            *[
                pybop.Parameter(param_name, initial_value=val)
                for param_name, val in CHARGE_TRANSFER_PARAMS
            ]
        )

    def assert_parameter_sensitivity(
        self, problem, initial_inputs, example_inputs, tolerance=RELATIVE_TOLERANCE
    ):
        """Reusable assertion for parameter sensitivity."""
        value1 = problem(initial_inputs)
        value2 = problem(example_inputs)

        relative_change = abs((value1 - value2) / value1)
        assert relative_change > tolerance, (
            f"Parameter change insufficient: {relative_change}"
        )

        return value1, value2

    def test_voltage_fitting(self, dataset, model_config, parameters):
        """Test grouped SPMe model with voltage-based cost functions."""
        # Build problem with multiple cost functions
        simulator = pybop.pybamm.Simulator(
            model_config["model"],
            parameter_values=model_config["parameter_values"],
            input_parameter_names=parameters.names,
            protocol=dataset,
        )
        cost_1 = pybop.SumSquaredError(dataset)
        cost_2 = pybop.MeanAbsoluteError(dataset)
        cost = pybop.WeightedCost(cost_1, cost_2)
        problem = pybop.FittingProblem(simulator, parameters, cost)

        # Test parameter sensitivity
        initial_params = parameters.get_initial_values()
        initial_inputs = parameters.to_dict(initial_params)
        example_inputs = parameters.to_dict(TEST_PARAM_VALUES)
        value1, value2 = self.assert_parameter_sensitivity(
            problem, initial_inputs, example_inputs
        )

        # Test sensitivity computation consistency
        value1_sens, grad1 = problem.single_call(initial_inputs, calculate_grad=True)
        value2_sens, grad2 = problem.single_call(example_inputs, calculate_grad=True)

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
        simulator = pybop.pybamm.EISSimulator(
            model_config["model"],
            parameter_values=model_config["parameter_values"],
            input_parameter_names=parameters.names,
            f_eval=eis_dataset["Frequency [Hz]"],
        )
        cost = pybop.MeanSquaredError(
            eis_dataset, target="Impedance", weighting="equal"
        )
        eis_problem = pybop.FittingProblem(simulator, parameters, cost)

        # Test parameter sensitivity
        initial_params = parameters.get_initial_values()
        initial_inputs = parameters.to_dict(initial_params)
        example_inputs = parameters.to_dict(TEST_PARAM_VALUES)
        self.assert_parameter_sensitivity(eis_problem, initial_inputs, example_inputs)
