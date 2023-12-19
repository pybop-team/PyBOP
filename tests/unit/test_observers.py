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
    def parameters(self):
        return [
            pybop.Parameter(
                "k",
                prior=pybop.Gaussian(0.1, 0.02),
                bounds=[0.0, 1.0],
            ),
            pybop.Parameter(
                "y0",
                prior=pybop.Gaussian(1.0, 0.2),
                bounds=[0.0, 1.0],
            ),
        ]

    @pytest.fixture
    def dataset(self, model, experiment):
        model.parameter_set.update(
            {
                "k": 0.1,
                "y0": 1.0,
            }
        )
        solution = model.predict()
        return [
            pybop.Dataset("Time [s]", solution["Time [s]"].data),
            pybop.Dataset("y", solution["y"].data),
        ]

    @pytest.mark.unit
    def test_observer(self, model):
        inputs = {"k": 0.1, "y0": 1.0}
        signal = "2y"
        observer = pybop.Observer(model, inputs, signal)
        t_eval = np.linspace(0, 1, 100)
        expected = inputs["y0"] * np.exp(-inputs["k"] * t_eval)
        for y, t in zip(expected, t_eval):
            if t != 0:
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
