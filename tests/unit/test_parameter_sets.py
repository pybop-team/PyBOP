import pybamm
import pytest

import pybop


class TestParameterSets:
    """
    A class to test parameter sets.
    """

    pytestmark = pytest.mark.unit

    def test_set_formation_concentrations(self):
        parameter_values = pybamm.ParameterValues("Chen2020")
        pybop.pybamm.set_formation_concentrations(parameter_values)

        parameter_values.set_initial_state(0.5)
        cn_half = parameter_values[
            "Initial concentration in negative electrode [mol.m-3]"
        ]
        cp_half = parameter_values[
            "Initial concentration in positive electrode [mol.m-3]"
        ]

        parameter_values.set_initial_state(1.0)
        assert (
            parameter_values["Initial concentration in negative electrode [mol.m-3]"]
            > cn_half
        )
        assert (
            parameter_values["Initial concentration in positive electrode [mol.m-3]"]
            < cp_half
        )

        parameter_values.set_initial_state(0.0)
        assert (
            parameter_values["Initial concentration in negative electrode [mol.m-3]"]
            < cn_half
        )
        assert (
            parameter_values["Initial concentration in positive electrode [mol.m-3]"]
            > cp_half
        )
