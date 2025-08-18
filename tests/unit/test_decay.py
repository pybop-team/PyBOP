import numpy as np
import pybamm
import pytest

import pybop


class TestDecay:
    """
    A class to test the exponential decay model.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture(scope="session")
    def model_config(self):
        """Shared model configuration to avoid repeated initialisation."""
        model = pybop.ExponentialDecayModel()
        parameter_values = model.default_parameter_values
        return {
            "model": model,
            "parameter_values": parameter_values,
            "solver": pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6),
        }

    def test_model_creation(self, model_config):
        model_instance = model_config["model"]
        assert model_instance is not None

        parameter_values = model_instance.default_parameter_values
        assert isinstance(parameter_values, pybamm.ParameterValues)

        variable_list = model_instance.default_quick_plot_variables
        assert isinstance(variable_list, list)

    def test_solution_and_sensitivities(self, model_config):
        # Define inputs (necessary in order to calculate sensitivities)
        parameter_values = model_config["parameter_values"]
        param = [parameter_values["k"], parameter_values["y0"]]
        inputs = {"k": param[0], "y0": param[1]}
        parameter_values.update({"k": "[input]", "y0": "[input]"})

        # Simulate
        sim = pybamm.Simulation(
            model_config["model"],
            parameter_values=parameter_values,
            solver=model_config["solver"],
        )
        t_eval = np.linspace(0, 20, 21)
        sol = sim.solve(
            t_eval=t_eval, t_interp=t_eval, inputs=inputs, calculate_sensitivities=True
        )

        fig = sol.plot()
        assert isinstance(fig, pybamm.QuickPlot)

        # Validate against analytic solution
        analytic_y = param[1] * np.exp(-param[0] * t_eval)
        analytic_sens = {
            "k": -t_eval * analytic_y,
            "y0": np.exp(-param[0] * t_eval),
        }

        np.testing.assert_allclose(sol["Time [s]"].data, t_eval)
        np.testing.assert_allclose(sol["y_0"].data, analytic_y, rtol=1e-4)
        np.testing.assert_allclose(
            sol["y_0"].sensitivities["k"], analytic_sens["k"], rtol=1e-4
        )
        np.testing.assert_allclose(
            sol["y_0"].sensitivities["y0"], analytic_sens["y0"], rtol=1e-4
        )
