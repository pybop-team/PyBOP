import numpy as np
import pybamm
import pytest

import pybop


class TestModels:
    """
    A class to test pybop created models.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture(
        params=[
            (pybop.lithium_ion.WeppnerHuggins, {}),
            (pybop.lithium_ion.SPDiffusion, {}),
            (pybop.lithium_ion.GroupedSPM, {}),
            (pybop.lithium_ion.GroupedSPM, {"surface form": "differential"}),
            (pybop.lithium_ion.GroupedSPMe, {}),
            (pybop.lithium_ion.GroupedSPMe, {"surface form": "differential"}),
        ],
    )
    def model_and_options(self, request):
        return request.param

    def test_model_creation(self, model_and_options):
        model, options = model_and_options
        model_instance = model(options=options)
        assert model_instance is not None

        parameter_values = model_instance.default_parameter_values
        assert isinstance(parameter_values, pybamm.ParameterValues)

        variable_list = model_instance.default_quick_plot_variables
        assert isinstance(variable_list, list)

    def test_model_simulation(self, model_and_options):
        model, options = model_and_options
        sim = pybamm.Simulation(model(options=options))
        t_eval = np.linspace(0, 10, 11)
        sol = sim.solve(t_eval=t_eval, t_interp=t_eval)
        np.testing.assert_allclose(sol["Time [s]"].data, t_eval)

        fig = sol.plot()
        assert isinstance(fig, pybamm.QuickPlot)
