from copy import copy

import numpy as np
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
            "Negative particle radius [m]": pybop.Parameter(
                pybop.Gaussian(2e-05, 1e-6, truncated_at=[1e-6, 5e-5])
            ),
            "Positive particle radius [m]": pybop.Parameter(
                pybop.Gaussian(0.5e-05, 1e-6, truncated_at=[1e-6, 5e-5])
            ),
        }

        simulator = BaseSimulator(copy(params))
        for key in params.keys():
            assert simulator.parameters[key] is params[key]
        # Check that pybop.Parameter objects are removed from the dictionary
        simulator = BaseSimulator(params)
        for key in params.keys():
            assert params[key] == "[input]"

        params = {
            "Negative particle radius [m]": (
                pybop.Gaussian(2e-05, 0.1e-5, truncated_at=[1e-6, 5e-5])
            ),
            "Positive particle radius [m]": (
                pybop.Gaussian(0.5e-05, 0.1e-5, truncated_at=[1e-6, 5e-5])
            ),
        }

        simulator = BaseSimulator(copy(params))
        assert len(simulator.parameters) == 0

        params = [
            pybop.Parameter(pybop.Gaussian(2e-05, 1e-6, truncated_at=[1e-6, 5e-5])),
            pybop.Parameter(pybop.Gaussian(2e-05, 1e-6, truncated_at=[1e-6, 5e-5])),
        ]

        with pytest.raises(
            TypeError,
            match="Parameters must be a dictionary of pybop.Parameter objects or a pybop.Parameters object.",
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


class TestPybammEISSimulator:
    """
    A class to tet the pybamm.EISSimulator class.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def model(self):
        return pybamm.lithium_ion.SPM()

    @pytest.fixture
    def parameter_values(self):
        parameter_values = pybamm.ParameterValues("Chen2020")
        parameter_values.set_initial_state(0.9)
        return parameter_values

    def test_input_parameters_are_not_swapped(self, model, parameter_values):
        """
        Passing values as inputs must give the same spectrum as fixing them in the
        parameter values, whatever the order of the inputs dictionary. The compiled
        Jacobian takes a single stacked vector, so an order which disagrees with the
        one used to compile it silently evaluates the model at the wrong point.
        """
        # Deliberately not in alphabetical order
        inputs = {
            "Positive particle diffusivity [m2.s-1]": 4e-15,
            "Negative particle diffusivity [m2.s-1]": 3.0e-14,
        }

        fixed_param = parameter_values.copy()
        fixed_param.update(inputs)
        expected_data = self.get_data(model, fixed_param)

        fitting_param = parameter_values.copy()
        fitting_param.update(
            {
                name: pybop.Parameter(pybop.Uniform(value / 2, value * 2))
                for name, value in inputs.items()
            }
        )
        fitting_data = self.get_data(model, fitting_param, inputs=inputs)

        np.testing.assert_allclose(
            fitting_data["Impedance"].data, expected_data["Impedance"], rtol=1e-10
        )

    def get_data(self, model, parameter_values, inputs=None):
        frequencies = np.logspace(-1, 3, 4)
        solution = pybop.pybamm.EISSimulator(
            model, parameter_values=parameter_values, f_eval=frequencies
        ).solve(inputs=inputs)
        return pybop.Dataset({"Impedance": solution["Impedance"].data})
