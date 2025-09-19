import pybamm
import pytest

from pybop.pybamm.parameter_utils import set_formation_concentrations


class TestParameterSets:
    """
    A class to test parameter sets.
    """

    pytestmark = pytest.mark.unit

    def test_set_formation_concentrations(self):
        parameter_values = pybamm.ParameterValues("Chen2020")
        set_formation_concentrations(parameter_values)

        assert (
            parameter_values["Initial concentration in negative electrode [mol.m-3]"]
            == 0
        )
        assert (
            parameter_values["Initial concentration in positive electrode [mol.m-3]"]
            > 0
        )
