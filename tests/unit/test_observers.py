import pybop
import numpy as np
import pybamm
import pytest

from tests.unit.exponential_decay import ExponentialDecay


class TestObserver:
    """
    A class to test the unscented kalman filter.
    """

    @pytest.fixture(params=[1, 2])
    def model(self, request):
        model = ExponentialDecay(
            parameters=pybamm.ParameterValues({"k": "[input]", "y0": "[input]"}),
            nstate=request.param,
        )
        model.build()
        return model

    @pytest.mark.unit
    def test_observer(self, model):
        inputs = {"k": 0.1, "y0": 1.0}
        signal = ["2y"]
        n = model.nstate
        observer = pybop.Observer(model, inputs, signal)
        t_eval = np.linspace(0, 1, 100)
        expected = inputs["y0"] * np.exp(-inputs["k"] * t_eval)
        for y, t in zip(expected, t_eval):
            observer.observe(t)
            np.testing.assert_array_almost_equal(
                observer.get_current_state().as_ndarray(),
                np.array([[y]] * n),
                decimal=4,
            )
            np.testing.assert_array_almost_equal(
                observer.get_current_measure(),
                np.array([[2 * y]]),
                decimal=4,
            )
