import numpy as np
import pybamm
import pytest

import pybop
from pybop.models.lithium_ion.grouped_spme import convert_physical_to_grouped_parameters


class TestModels:
    """
    A class to test the models.
    """

    pytestmark = pytest.mark.unit

    @pytest.mark.parametrize(
        "model",
        [
            pybop.lithium_ion.WeppnerHuggins(),
            pybop.ExponentialDecayModel(),
            pybop.lithium_ion.GroupedSPMe(),
            pybop.lithium_ion.GroupedSPMe(options={"surface form": "differential"}),
        ],
    )
    def test_set_initial_state(self, model):
        model = pybamm.lithium_ion.SPM()
        t_eval = np.linspace(0, 10, 100)
        model.build(initial_state={"Initial SoC": 0.7})
        values_1 = model.predict(t_eval=t_eval)

        model = pybop.lithium_ion.SPM()
        model.build(initial_state={"Initial SoC": 0.4})
        model.set_initial_state({"Initial SoC": 0.7})
        values_2 = model.predict(t_eval=t_eval)

        np.testing.assert_allclose(
            values_1["Voltage [V]"].data, values_2["Voltage [V]"].data, atol=1e-8
        )

        init_ocp_p = model.parameter_set["Positive electrode OCP [V]"](0.7)
        init_ocp_n = model.parameter_set["Negative electrode OCP [V]"](0.7)
        model.set_initial_state(
            {"Initial open-circuit voltage [V]": init_ocp_p - init_ocp_n}
        )
        values_3 = model.predict(t_eval=t_eval)

        np.testing.assert_allclose(
            values_1["Voltage [V]"].data, values_3["Voltage [V]"].data, atol=0.05
        )

        with pytest.raises(ValueError, match="Expecting only one initial state."):
            model.set_initial_state(
                {"Initial open-circuit voltage [V]": 3.7, "Initial SoC": 0.7}
            )
        with pytest.raises(ValueError, match="Unrecognised initial state"):
            model.set_initial_state({"Initial voltage [V]": 3.7})

    def test_grouped_SPMe(self):
        parameter_set = pybop.ParameterSet.pybamm("Chen2020")
        parameter_set["Electrolyte diffusivity [m2.s-1]"] = 1.769e-10
        parameter_set["Electrolyte conductivity [S.m-1]"] = 0.9487

        grouped_parameter_set = convert_physical_to_grouped_parameters(parameter_set)
        model = pybop.lithium_ion.GroupedSPMe(parameter_set=grouped_parameter_set)

        with pytest.raises(
            ValueError, match="GroupedSPMe can currently only accept an initial SoC."
        ):
            model.set_initial_state({"Initial open-circuit voltage [V]": 3.7})

        model.set_initial_state({"Initial SoC": 1.0})
        res = model.predict(t_eval=np.linspace(0, 10, 100))
        assert len(res["Voltage [V]"].data) == 100

        variable_list = model.pybamm_model.default_quick_plot_variables
        assert isinstance(variable_list, list)
