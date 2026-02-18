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
            pybop.lithium_ion.CellTemperature(),
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

    def test_set_initial_state(self, model):
        if isinstance(model, pybop.ExponentialDecayModel):
            pass  # Only testing the battery models for now

        elif isinstance(model, pybop.lithium_ion.WeppnerHuggins):
            param = model.default_parameter_values
            with pytest.raises(
                ValueError,
                match="The Weppner & Huggins model does not have an initial state.",
            ):
                param.set_initial_state(0.5)

        else:
            if isinstance(model, pybop.lithium_ion.SPDiffusion):
                initial_state = "Initial stoichiometry"
            else:
                initial_state = "Initial SoC"

            param = model.default_parameter_values
            param.set_initial_state(0.5)
            assert param[initial_state] == 0.5

            param.set_initial_state("2.8 V")
            assert 0 <= param[initial_state] <= 1

            with pytest.raises(
                ValueError,
                match="Initial value must be a float or a string ending in 'V'.",
            ):
                param.set_initial_state([1])

            with pytest.raises(ValueError, match="should be between 0 and 1."):
                param.set_initial_state(-1)

            if not isinstance(model, pybop.lithium_ion.SPDiffusion):
                with pytest.raises(
                    ValueError, match=r"V is outside the voltage limits"
                ):
                    param.set_initial_state("-1 V")


class TestUtils:
    """
    A class to test PyBOP models utility functions.
    """

    pytestmark = pytest.mark.unit

    def test_inverse_ocv(self):
        def ocv_function(x):
            return x**3

        inverse_ocv = pybop.models.lithium_ion.utils.InverseOCV(ocv_function)

        root = inverse_ocv(0.125)
        assert np.isclose(root, 0.5)

    def test_interpolant(self):
        x = np.linspace(-2, 2, 100)
        y = x**2
        interpolant = pybop.models.lithium_ion.utils.Interpolant(x, y)

        # Test numeric evaluation
        np.testing.assert_almost_equal(interpolant(0.5), 0.25, decimal=3)
        np.testing.assert_almost_equal(interpolant(-1.5), 2.25, decimal=3)

        # Test symbolic evaluation
        x_sym = pybamm.Scalar(0.5)
        interp_sym = interpolant(x_sym)
        assert isinstance(interp_sym, pybamm.Interpolant)
