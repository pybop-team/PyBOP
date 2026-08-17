import numpy as np
import pybamm
import pytest
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure

import pybop

GROUPED_MODEL = pybop.lithium_ion.GroupedSPM | pybop.lithium_ion.GroupedSPMe


class TestModels:
    """
    A class to test pybop created models.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture(
        params=[
            pybop.ExponentialDecayModel(),
            pybop.lithium_ion.CellTemperature(),
            pybop.lithium_ion.GroupedSPM(),
            pybop.lithium_ion.GroupedSPM(options={"surface form": "differential"}),
            pybop.lithium_ion.GroupedSPMe(),
            pybop.lithium_ion.GroupedSPMe(options={"surface form": "differential"}),
            pybop.li_half_cell.WeppnerHuggins(),
            pybop.li_half_cell.SPDiffusion(),
        ],
        ids=lambda val: f"{type(val).__name__}",
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

        if isinstance(model, GROUPED_MODEL):
            for split in [False, True]:
                fig, ax = solution.plot_voltage_components(split_by_electrode=split)
                assert isinstance(fig, Figure)
                assert isinstance(ax, Axes)

        if not isinstance(model, pybop.ExponentialDecayModel):
            experiment = pybamm.Experiment(["Discharge at 1C for 1 minute"])
            solution = pybamm.Simulation(model, experiment=experiment).solve(
                calc_esoh=False
            )
            assert solution["Time [s]"].data[-1] == 60

    def test_set_initial_state(self, model):
        if isinstance(model, pybop.ExponentialDecayModel):
            pass  # Only testing the battery models for now

        elif isinstance(model, pybop.li_half_cell.WeppnerHuggins):
            param = model.default_parameter_values
            with pytest.raises(
                ValueError,
                match="The Weppner & Huggins model does not have an initial state.",
            ):
                param.set_initial_state(0.5)

        else:
            if isinstance(model, pybop.li_half_cell.SPDiffusion):
                initial_state = "Initial stoichiometry"
            else:
                initial_state = "Initial SoC"

            param = model.default_parameter_values
            param.set_initial_state(0.5)
            assert param[initial_state] == 0.5

            param.set_initial_state("3.8 V")
            assert 0 <= param[initial_state] <= 1

            with pytest.raises(
                ValueError,
                match="Initial value must be a float or a string ending in 'V'.",
            ):
                param.set_initial_state([1])

            with pytest.raises(ValueError, match="should be between 0 and 1."):
                param.set_initial_state(-1)

            if not isinstance(model, pybop.li_half_cell.SPDiffusion):
                with pytest.raises(
                    ValueError, match=r"V is outside the voltage limits"
                ):
                    param.set_initial_state("-1 V")

    def test_alternative_functions(self, model):
        if isinstance(model, GROUPED_MODEL):
            parameter_values = model.default_parameter_values

            # Use alternative Butler-Volmer kinetics
            parameter_values["Negative electrode charge transfer coefficient"] = 0.6
            parameter_values["Negative electrode dimensionless exchange rate"] = (
                model.get_asymmetric_butler_volmer("negative")
            )
            parameter_values["Positive electrode charge transfer coefficient"] = 0.6
            parameter_values["Positive electrode charge transfer ideality factor"] = 0.6
            parameter_values["Positive electrode dimensionless exchange rate"] = (
                model.get_multiphase_butler_volmer("positive")
            )

            t_eval = np.linspace(0, 10, 11)
            solution = pybamm.Simulation(
                model, parameter_values=parameter_values
            ).solve(t_eval=t_eval, t_interp=t_eval)
            np.testing.assert_allclose(solution["Time [s]"].data, t_eval)


class TestModelUtils:
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
