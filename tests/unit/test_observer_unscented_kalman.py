import pybop
import numpy as np
import pybamm
import pytest
from pybop.observers.unscented_kalman import UkfFilter

from tests.unit.exponential_decay import ExponentialDecay


class TestUKF:
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

    @pytest.fixture
    def dataset(self, model: pybop.BaseModel):
        inputs = {"k": 0.1, "y0": 1.0}
        observer = pybop.Observer(model, inputs, ["2y"])
        measurements = []
        t_eval = np.linspace(0, 20, 10)
        for t in t_eval:
            observer.observe(t)
            measurements.append(observer.get_current_measure())
        measurements = np.hstack(measurements)
        return {"Time [s]": t_eval, "y": measurements}

    @pytest.fixture
    def observer(self, model: pybop.BaseModel):
        inputs = {"k": 0.1, "y0": 1.0}
        signal = ["2y"]
        n = model.nstate
        sigma0 = np.diag([1e-4] * n)
        process = np.diag([1e-4] * n)
        measure = np.diag([1e-4])
        observer = pybop.UnscentedKalmanFilterObserver(
            model, inputs, signal, sigma0, process, measure
        )
        return observer

    @pytest.mark.unit
    def test_cholupdate(self):
        # Create a random positive definite matrix, V
        np.random.seed(1)
        X = np.random.normal(size=(100, 10))
        V = np.dot(X.transpose(), X)

        # Calculate the upper Cholesky factor, R
        R = np.linalg.cholesky(V).transpose()

        # Create a random update vector, u
        u = np.random.normal(size=R.shape[0])

        # Calculate the updated positive definite matrix, V1, and its Cholesky factor, R1
        V1 = V + np.outer(u, u)
        R1 = np.linalg.cholesky(V1).transpose()

        # The following is equivalent to the above
        R1_ = R.copy()
        UkfFilter.cholupdate(R1_, u.copy(), 1.0)
        np.testing.assert_array_almost_equal(R1, R1_)

    @pytest.mark.unit
    def test_unscented_kalman_filter(self, dataset, observer):
        t_eval = dataset["Time [s]"]
        measurements = dataset["y"]
        inputs = observer._state.inputs
        n = observer._model.nstate
        expected = inputs["y0"] * np.exp(-inputs["k"] * t_eval)
        for i in range(len(t_eval)):
            y = np.array([[expected[i]]] * n)
            t = t_eval[i]
            ym = measurements[:, i]
            observer.observe(t, ym)
            np.testing.assert_array_almost_equal(
                observer.get_current_state().as_ndarray(),
                y,
                decimal=4,
            )
            np.testing.assert_array_almost_equal(
                observer.get_current_measure(),
                np.array([2 * y[0]]),
                decimal=4,
            )

    @pytest.mark.unit
    def test_observe_no_measurement(self, observer):
        with pytest.raises(ValueError):
            observer.observe(0, None)

    @pytest.mark.unit
    def test_observe_decreasing_time(self, observer):
        observer.observe(0, np.array([2]))
        observer.observe(0.1, np.array([2]))
        with pytest.raises(ValueError):
            observer.observe(0, np.array([2]))

    @pytest.mark.unit
    def test_wrong_input_shapes(self, model):
        inputs = {"k": 0.1, "y0": 1.0}
        signal = "2y"
        n = model.nstate

        sigma0 = np.diag([1e-4] * (n + 1))
        process = np.diag([1e-4] * n)
        measure = np.diag([1e-4])
        with pytest.raises(ValueError):
            pybop.UnscentedKalmanFilterObserver(
                model, inputs, signal, sigma0, process, measure
            )

        sigma0 = np.diag([1e-4] * n)
        process = np.diag([1e-4] * (n + 1))
        measure = np.diag([1e-4])
        with pytest.raises(ValueError):
            pybop.UnscentedKalmanFilterObserver(
                model, inputs, signal, sigma0, process, measure
            )

        sigma0 = np.diag([1e-4] * n)
        process = np.diag([1e-4] * n)
        measure = np.diag([1e-4] * 2)
        with pytest.raises(ValueError):
            pybop.UnscentedKalmanFilterObserver(
                model, inputs, signal, sigma0, process, measure
            )
