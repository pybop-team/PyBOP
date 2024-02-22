import pybop
import numpy as np
import pytest


class TestParameterSets:
    """
    A class to test parameter sets.
    """

    @pytest.mark.unit
    def test_parameter_set(self):
        # Tests parameter set creation and validation
        with pytest.raises(ValueError):
            pybop.ParameterSet.pybamm("sChen2010s")

        parameter_test = pybop.ParameterSet.pybamm("Chen2020")
        np.testing.assert_allclose(
            parameter_test["Negative electrode active material volume fraction"], 0.75
        )

    @pytest.mark.unit
    def test_ecm_parameter_sets(self):
        # Test importing a json file
        json_params = pybop.ParameterSet(
            json_path="examples/scripts/parameters/initial_ecm_parameters.json"
        )
        json_params.import_parameters()

        params = pybop.ParameterSet(
            params_dict={
                "chemistry": "ecm",
                "Initial SoC": 0.5,
                "Initial temperature [K]": 25 + 273.15,
                "Cell capacity [A.h]": 5,
                "Nominal cell capacity [A.h]": 5,
                "Ambient temperature [K]": 25 + 273.15,
                "Current function [A]": 5,
                "Upper voltage cut-off [V]": 4.2,
                "Lower voltage cut-off [V]": 3.0,
                "Cell thermal mass [J/K]": 1000,
                "Cell-jig heat transfer coefficient [W/K]": 10,
                "Jig thermal mass [J/K]": 500,
                "Jig-air heat transfer coefficient [W/K]": 10,
                "Open-circuit voltage [V]": pybop.empirical.Thevenin().default_parameter_values[
                    "Open-circuit voltage [V]"
                ],
                "R0 [Ohm]": 0.001,
                "Element-1 initial overpotential [V]": 0,
                "Element-2 initial overpotential [V]": 0,
                "R1 [Ohm]": 0.0002,
                "R2 [Ohm]": 0.0003,
                "C1 [F]": 10000,
                "C2 [F]": 5000,
                "Entropic change [V/K]": 0.0004,
            }
        )
        params.import_parameters()

        assert json_params.params == params.params

        # Test exporting a json file
        parameters = [
            pybop.Parameter(
                "R0 [Ohm]",
                prior=pybop.Gaussian(0.0002, 0.0001),
                bounds=[1e-4, 1e-2],
                initial_value=0.001,
            ),
            pybop.Parameter(
                "R1 [Ohm]",
                prior=pybop.Gaussian(0.0001, 0.0001),
                bounds=[1e-5, 1e-2],
                initial_value=0.0002,
            ),
        ]
        params.export_parameters(
            "examples/scripts/parameters/fit_ecm_parameters.json", fit_params=parameters
        )
