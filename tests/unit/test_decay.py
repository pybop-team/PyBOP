import numpy as np
import pybamm
import pytest

import pybop


class TestDecay:
    """
    A class to test the exponential decay model.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture(scope="module")
    def model_config(self):
        """Shared model configuration to avoid repeated initialisation."""
        model = pybop.ExponentialDecayModel()
        parameter_values = model.default_parameter_values
        return {
            "model": model,
            "parameter_values": parameter_values,
            "solver": pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6),
        }

    def test_initialisation(self):
        with pytest.raises(
            ValueError,
            match=r"The number of states \(n_states\) must be at least 1.",
        ):
            pybop.ExponentialDecayModel(n_states=0)

    def test_solution_and_sensitivities(self, model_config):
        # Define inputs (necessary in order to calculate sensitivities)
        parameter_values = model_config["parameter_values"]
        param = [parameter_values["k"], parameter_values["y0"]]
        parameters = {
            "k": pybop.Parameter("k", initial_value=param[0]),
            "y0": pybop.Parameter("y0", initial_value=param[1]),
        }
        inputs = {"k": param[0], "y0": param[1]}
        parameter_values.update(parameters)

        # Simulate
        t_eval = np.linspace(0, 20, 21)
        sim = pybop.pybamm.Simulator(
            model_config["model"],
            parameter_values=parameter_values,
            protocol=t_eval,
            solver=model_config["solver"],
        )
        solution = sim.solve(inputs, calculate_sensitivities=True)

        fig = solution.plot()
        assert isinstance(fig, pybamm.QuickPlot)

        # Validate against analytic solution
        analytic_y = param[1] * np.exp(-param[0] * t_eval)
        analytic_sens = {
            "k": -t_eval * analytic_y,
            "y0": np.exp(-param[0] * t_eval),
        }

        np.testing.assert_allclose(solution["Time [s]"].data, t_eval)
        np.testing.assert_allclose(solution["y_0"].data, analytic_y, rtol=1e-4)
        np.testing.assert_allclose(
            solution["y_0"].sensitivities["k"], analytic_sens["k"], rtol=1e-4
        )
        np.testing.assert_allclose(
            solution["y_0"].sensitivities["y0"], analytic_sens["y0"], rtol=1e-4
        )
