import numpy as np
import pybamm
import pytest

import pybop


class TestObserver:
    """
    A class to test the observer class.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def parameters(self):
        return pybop.Parameters(
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
        )

    @pytest.fixture(params=[1, 2])
    def model(self, parameters, request):
        parameter_set = pybamm.ParameterValues({"k": 0.1, "y0": 1.0})
        model = pybop.ExponentialDecayModel(
            parameter_set=parameter_set, n_states=request.param
        )
        model.build(parameters=parameters)
        return model

    def test_observer(self, model, parameters):
        n = model.n_states
        observer = pybop.Observer(parameters, model, signal=["y_0"])
        t_eval = np.linspace(0, 20, 10)
        expected = parameters["y0"].initial_value * np.exp(
            -parameters["k"].initial_value * t_eval
        )
        for y, t in zip(expected, t_eval, strict=False):
            observer.observe(t)
            np.testing.assert_array_almost_equal(
                observer.get_current_state().as_ndarray(),
                np.array([[y]] * n),
                decimal=4,
            )
            np.testing.assert_array_almost_equal(
                observer.get_current_measure(),
                np.array([[y]]),
                decimal=4,
            )

        # Test with invalid inputs
        with pytest.raises(ValueError, match="Time must be increasing."):
            observer.observe(-1)
        with pytest.raises(
            ValueError, match="values and times must have the same length."
        ):
            observer.log_likelihood(
                {"y_0": t_eval}, np.array([1]), inputs=observer._state.inputs
            )

        # Test covariance
        covariance = observer.get_current_covariance()
        assert np.shape(covariance) == (n, n)

        # Test evaluate with different inputs
        observer.domain_data = t_eval
        observer.evaluate(parameters.as_dict())
        observer.evaluate(parameters.current_value())

        # Test evaluate with dataset
        observer._dataset = pybop.Dataset(
            {
                "Time [s]": t_eval,
                "Output": expected,
            }
        )
        observer._target = {"y_0": expected}
        observer.evaluate(parameters.as_dict())

    def test_unbuilt_model(self, parameters):
        model = pybop.ExponentialDecayModel()
        with pytest.raises(
            ValueError, match="Only built models can be used in Observers"
        ):
            pybop.Observer(parameters, model)

    def test_observer_inputs(self):
        initial_state = {"Initial open-circuit voltage [V]": 4.0}
        t_eval = np.linspace(0, 1, 100)
        model = pybop.ExponentialDecayModel(n_states=1)
        model.build()
        observer = pybop.Observer(
            pybop.Parameters(),
            model,
            signal=["y_0", "y_0"],
            initial_state=initial_state,
        )
        assert observer.initial_state == initial_state

        with pytest.raises(
            ValueError,
            match="Observer.log_likelihood is currently restricted to single output models.",
        ):
            observer.log_likelihood(
                {"y_0": t_eval}, t_eval, inputs=observer._state.inputs
            )
