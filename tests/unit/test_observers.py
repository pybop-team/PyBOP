import pybop
import numpy as np
import pybamm
import pytest

from tests.unit.exponential_decay import ExponentialDecay


class TestObserver:
    """
    A class to test the unscented kalman filter.
    """

    @pytest.fixture
    def model(self):
        model = ExponentialDecay(
            parameters=pybamm.ParameterValues({"k": "[input]", "y0": "[input]"})
        )
        model.build()
        return model

    @pytest.fixture
    def dataset(self, model: pybop.BaseModel):
        inputs = {"k": 0.1, "y0": 1.0}
        observer = pybop.Observer(model, inputs, "2y")
        measurements = []
        t_eval = np.linspace(0, 1, 10)
        for t in t_eval:
            observer.observe(t)
            measurements.append(observer.get_current_measure())
        measurements = np.hstack(measurements)
        return {"Time [s]": t_eval, "y": measurements}

    @pytest.mark.unit
    def test_observer(self, model):
        inputs = {"k": 0.1, "y0": 1.0}
        signal = "2y"
        observer = pybop.Observer(model, inputs, signal)
        t_eval = np.linspace(0, 1, 100)
        expected = inputs["y0"] * np.exp(-inputs["k"] * t_eval)
        for y, t in zip(expected, t_eval):
            observer.observe(t)
            np.testing.assert_array_almost_equal(
                observer.get_current_state().as_ndarray(),
                np.array([[y]]),
                decimal=4,
            )
            np.testing.assert_array_almost_equal(
                observer.get_current_measure(),
                np.array([2 * y]),
                decimal=4,
            )

    @pytest.mark.unit
    def test_unscented_kalman_filter(self, model, dataset):
        inputs = {"k": 0.1, "y0": 1.0}
        signal = "2y"
        sigma0 = np.diag([0.1, 0.1])
        process = np.diag([0.1, 0.1])
        measure = np.diag([0.1])
        observer = pybop.UnscentedKalmanFilterObserver(
            model, inputs, signal, sigma0, process, measure
        )
        t_eval = dataset["Time [s]"]
        measurements = dataset["y"]
        expected = inputs["y0"] * np.exp(-inputs["k"] * t_eval)
        for y, t, ym in zip(expected, t_eval, measurements):
            observer.observe(t, ym)
            np.testing.assert_array_almost_equal(
                observer.get_current_state().as_ndarray(),
                np.array([[y]]),
                decimal=4,
            )
            np.testing.assert_array_almost_equal(
                observer.get_current_measure(),
                np.array([2 * y]),
                decimal=4,
            )
