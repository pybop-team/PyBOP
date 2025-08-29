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
            (pybop.ExponentialDecayModel, {}),
            (pybop.lithium_ion.WeppnerHuggins, {}),
            (pybop.lithium_ion.SPDiffusion, {}),
            (pybop.lithium_ion.GroupedSPM, {}),
            (
                pybop.lithium_ion.GroupedSPM,
                {"options": {"surface form": "differential"}},
            ),
            (pybop.lithium_ion.GroupedSPMe, {}),
            (
                pybop.lithium_ion.GroupedSPMe,
                {"options": {"surface form": "differential"}},
            ),
        ],
        scope="module",
    )
    def model_and_kwargs(self, request):
        return request.param

    def test_model_creation(self, model_and_kwargs):
        model, kwargs = model_and_kwargs
        model_instance = model(**kwargs)
        assert model_instance is not None

        parameter_values = model_instance.default_parameter_values
        assert isinstance(parameter_values, pybamm.ParameterValues)

        variable_list = model_instance.default_quick_plot_variables
        assert isinstance(variable_list, list)

        if not model_instance.built:
            model_instance.build_model()
            assert model_instance is not None

    def test_model_simulation(self, model_and_kwargs):
        model, kwargs = model_and_kwargs
        sim = pybamm.Simulation(model(**kwargs))
        t_eval = np.linspace(0, 10, 11)
        sol = sim.solve(t_eval=t_eval, t_interp=t_eval)
        np.testing.assert_allclose(sol["Time [s]"].data, t_eval)

        fig = sol.plot()
        assert isinstance(fig, pybamm.QuickPlot)
