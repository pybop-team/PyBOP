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
            pybop.ExponentialDecayModel(),
            pybop.lithium_ion.WeppnerHuggins(),
            pybop.lithium_ion.SPDiffusion(),
            pybop.lithium_ion.GroupedSPM(),
            pybop.lithium_ion.GroupedSPM(options={"surface form": "differential"}),
            pybop.lithium_ion.GroupedSPMe(),
            pybop.lithium_ion.GroupedSPMe(options={"surface form": "differential"}),
        ],
        scope="module",
    )
    def model(self, request):
        return request.param

    def test_model_instance(self, model):
        parameter_values = model.default_parameter_values
        assert isinstance(parameter_values, pybamm.ParameterValues)

        variable_list = model.default_quick_plot_variables
        assert isinstance(variable_list, list)

        if not model.built:
            model.build_model()
            assert model is not None

    def test_model_simulation(self, model):
        t_eval = np.linspace(0, 10, 11)
        solution = pybamm.Simulation(model).solve(t_eval=t_eval, t_interp=t_eval)
        np.testing.assert_allclose(solution["Time [s]"].data, t_eval)

        fig = solution.plot()
        assert isinstance(fig, pybamm.QuickPlot)
