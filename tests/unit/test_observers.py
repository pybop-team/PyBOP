import pybop
import numpy as np
import pybamm
import pytest
from examples.standalone.model import ExponentialDecay


class TestObserver:
    """
    A class to test the observer class.
    """

    @pytest.fixture(params=[1, 2])
    def model(self, request):
        model = ExponentialDecay(
            parameter_set=pybamm.ParameterValues({"k": "[input]", "y0": "[input]"}),
            n_states=request.param,
        )
        model.build()
        return model

    @pytest.fixture
    def parameters(self):
        return [
            pybop.Parameter(
                "k",
                prior=pybop.Gaussian(0.1, 0.05),
                bounds=[0, 1],
            ),
            pybop.Parameter(
                "y0",
                prior=pybop.Gaussian(1, 0.05),
                bounds=[0, 3],
            ),
        ]

    @pytest.fixture
    def x0(self):
        return np.array([0.1, 1.0])

    @pytest.mark.unit
    def test_observer(self, model, parameters, x0):
        n = model.n_states
        observer = pybop.Observer(parameters, model, signal=["2y"], x0=x0)
        t_eval = np.linspace(0, 1, 100)
        expected = x0[1] * np.exp(-x0[0] * t_eval)
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

        # Test with invalid inputs
        with pytest.raises(ValueError):
            observer.observe(-1)
        with pytest.raises(ValueError):
            observer.log_likelihood(
                t_eval, np.array([1]), inputs=observer._state.inputs
            )

        # Test covariance
        covariance = observer.get_current_covariance()
        assert np.shape(covariance) == (n, n)

    @pytest.mark.unit
    def test_unbuilt_model(self, parameters):
        model = ExponentialDecay()
        with pytest.raises(ValueError):
            pybop.Observer(parameters, model)
