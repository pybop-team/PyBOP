import numpy as np
import pytest

import pybop


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

        # Test getting and setting parameters
        parameter_test["Negative electrode active material volume fraction"] = 0.8
        assert (
            parameter_test["Negative electrode active material volume fraction"] == 0.8
        )

    @pytest.mark.unit
    def test_ecm_parameter_sets(self):
        # Test importing a json file
        json_params = pybop.ParameterSet()
        with pytest.raises(
            ValueError,
            match="Parameter set already constructed, or path to json file not provided.",
        ):
            json_params.import_parameters()

        json_params = pybop.ParameterSet(
            json_path="examples/scripts/parameters/initial_ecm_parameters.json"
        )
        json_params.import_parameters()

        with pytest.raises(
            ValueError,
            match="Parameter set already constructed, or path to json file not provided.",
        ):
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
                "Open-circuit voltage [V]": "default",
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

        assert json_params.params == params.params
        assert params() == params.params

        # Test exporting a json file
        parameters = pybop.Parameters(
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
        )
        params.export_parameters(
            "examples/scripts/parameters/fit_ecm_parameters.json", fit_params=parameters
        )

        # Test error when there no parameters to export
        empty_params = pybop.ParameterSet()
        with pytest.raises(ValueError):
            empty_params.export_parameters(
                "examples/scripts/parameters/fit_ecm_parameters.json"
            )

    @pytest.mark.unit
    def test_bpx_parameter_sets(self):
        # Test importing a BPX json file
        bpx_parameters = pybop.ParameterSet()
        with pytest.raises(
            ValueError,
            match="Parameter set already constructed, or path to bpx file not provided.",
        ):
            bpx_parameters.import_from_bpx()

        bpx_parameters = pybop.ParameterSet(
            json_path="examples/scripts/parameters/example_BPX.json"
        )
        bpx_parameters.import_from_bpx()

        with pytest.raises(
            ValueError,
            match="Parameter set already constructed, or path to bpx file not provided.",
        ):
            bpx_parameters.import_from_bpx()
