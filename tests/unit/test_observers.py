import numpy as np
import pybamm
import pytest

import pybop
from examples.standalone.model import ExponentialDecay


class TestObserver:
    """
    A class to test the observer class.
    """

    @pytest.fixture
    def parameters(self):
        return pybop.Parameters(
            pybop.Parameter(
                "k",
                prior=pybop.Gaussian(0.1, 0.05),
                bounds=[0, 1],
                initial_value=0.1,
            ),
            pybop.Parameter(
                "y0",
                prior=pybop.Gaussian(1, 0.05),
                bounds=[0, 3],
                initial_value=1.0,
            ),
        )

    @pytest.fixture(params=[1, 2])
    def model(self, parameters, request):
        model = ExponentialDecay(
            parameter_set=pybamm.ParameterValues({"k": "[input]", "y0": "[input]"}),
            n_states=request.param,
        )
        model.build(parameters=parameters)
        return model

    @pytest.mark.unit
    def test_observer(self, model, parameters):
        n = model.n_states
        observer = pybop.Observer(parameters, model, signal=["2y"])
        t_eval = np.linspace(0, 1, 100)
        expected = parameters["y0"].initial_value * np.exp(
            -parameters["k"].initial_value * t_eval
        )
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
                {"2y": t_eval}, np.array([1]), inputs=observer._state.inputs
            )

        # Test covariance
        covariance = observer.get_current_covariance()
        assert np.shape(covariance) == (n, n)

        # Test evaluate with different inputs
        observer._time_data = t_eval
        observer.evaluate(parameters.as_dict())
        observer.evaluate(parameters.current_value())

        # Test evaluate with dataset
        observer._dataset = pybop.Dataset(
            {
                "Time [s]": t_eval,
                "Output": expected,
            }
        )
        observer._target = {"2y": expected}
        observer.evaluate(parameters.as_dict())

    @pytest.mark.unit
    def test_unbuilt_model(self, parameters):
        model = ExponentialDecay()
        with pytest.raises(ValueError):
            pybop.Observer(parameters, model)
