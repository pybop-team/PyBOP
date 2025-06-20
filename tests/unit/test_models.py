import pybamm
import pytest

import pybop


class TestModels:
    """
    A class to test pybop created models.
    """

    pytestmark = pytest.mark.unit

    def test_weppner_huggins(self):
        model = pybop.lithium_ion.WeppnerHuggins()
        assert model is not None

    def test_grouped_spme_create_grouped_parameters(self):
        parameter_values = pybamm.ParameterValues("Chen2020")
        parameter_values["Electrolyte diffusivity [m2.s-1]"] = 1.769e-10
        parameter_values["Electrolyte conductivity [S.m-1]"] = 0.9487

        grouped_values = pybop.lithium_ion.GroupedSPMe.create_grouped_parameters(
            parameter_values
        )
        assert isinstance(grouped_values, pybamm.ParameterValues)

        model = pybop.lithium_ion.GroupedSPMe()
        variable_list = model.default_quick_plot_variables
        assert isinstance(variable_list, list)
