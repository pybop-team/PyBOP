import pybamm
import pytest

import pybop
from pybop.simulators.base_simulator import BaseSimulator


class TestSimulator:
    """
    A class to test the BaseSimulator class.
    """

    pytestmark = pytest.mark.unit

    def test_parameter_errors_constructor(self):
        params = {
            "Negative particle radius [m]": pybop.Gaussian(
                2e-05, 0.1e-5, truncated_at=[1e-6, 5e-5]
            ),
            "Positive particle radius [m]": pybop.Gaussian(
                0.5e-05, 0.1e-5, truncated_at=[1e-6, 5e-5]
            ),
        }

        with pytest.raises(
            TypeError,
            match="All elements in the list must be pybop.Parameter objects.",
        ):
            BaseSimulator(params)

        params = [
            pybop.Parameter(pybop.Gaussian(2e-05, 0.1e-5, truncated_at=[1e-6, 5e-5])),
            pybop.Parameter(pybop.Gaussian(2e-05, 0.1e-5, truncated_at=[1e-6, 5e-5])),
        ]

        with pytest.raises(
            TypeError,
            match="The input parameters must be a a dictionary of Parameter objects or a pybop.Parameters object.",
        ):
            BaseSimulator(params)


class TestPybammSimulator:
    """
    A class to test the pybamm.Simulator class.
    """

    pytestmark = pytest.mark.unit

    def test_set_output_variables(self):
        model = pybamm.lithium_ion.SPM()
        parameter_values = model.default_parameter_values
        experiment = pybamm.Experiment(
            ["Discharge at 1C for 5 minutes (10 second period)"]
        )

        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=experiment
        )

        with pytest.raises(
            ValueError, match="Not a variable is not a variable in the model."
        ):
            simulator.set_output_variables(["Not a variable"])
