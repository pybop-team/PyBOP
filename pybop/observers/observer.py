from typing import Dict, Optional
import numpy as np
from pybop.models.base_model import BaseModel, Inputs, TimeSeriesState


class Observer(object):
    """
    An observer of a time series state. Observers:
     1. keep track of the distribution of a current time series model state
     2. predict forward in time the distribution of the state
     3. update the distribution of the state with new observations

    Parameters
    ----------
    model : BaseModel
      The model to observe.
    inputs: Dict[str, float]
      The inputs to the model.
    sigmal: str
      The signal to observe.
    """

    Covariance = np.ndarray

    def __init__(self, model: BaseModel, inputs: Dict[str, float], signal: str) -> None:
        if model._built_model is None:
            raise ValueError("Only built models can bse used in Observers")
        model.signal = signal
        self._state = model.reinit(inputs)
        self._model = model
        self._signal = signal

    def reset(self) -> None:
        self._state = self._model.reinit()

    def observe(self, time: float, value: Optional[np.ndarray] = None) -> float:
        """
        Predict the time series model until t = `time` and optionally observe the measurement `value`.

        Returns the log likelihood of the model given the value and inputs. If no value is given, the log likelihood is 0.

        The base observer does not perform any value observation and always returns 0.

        Parameters
        ----------
        time : float
          The time of the new observation.

        value : np.ndarray (optional)
          The new observation.
        """
        if time < self._state.t:
            raise ValueError("Time must be increasing.")
        if time != self._state.t:
            self._state = self._model.step(self._state, time)
        return 0.0

    def log_likelihood(
        self, values: np.ndarray, times: np.ndarray, inputs: Inputs
    ) -> float:
        """
        Returns the log likelihood of the model given the values and inputs.

        Parameters
        ----------
        values : np.ndarray
          The values of the model.
        inputs : Inputs
          The inputs to the model.
        """
        if len(values) != len(times):
            raise ValueError("values and times must have the same length.")
        log_likelihood = 0.0
        self.reset()
        for t, v in zip(times, values):
            log_likelihood += self.observe(t, v)
        return log_likelihood

    def get_current_state(self) -> TimeSeriesState:
        """
        Returns the current state of the model.
        """
        return self._state

    def get_current_measure(self) -> np.ndarray:
        """
        Returns the current measurement.
        """
        return self.get_measure(self._state)

    def get_measure(self, x: TimeSeriesState) -> np.ndarray:
        m = x.sol[self._signal].data[-1]
        return np.array([[m]])

    def get_current_time(self) -> float:
        """
        Returns the current time.
        """
        return self._state.t
