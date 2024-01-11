from typing import List, Optional
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
    signal: List[str]
      The signal to observe.
    """

    Covariance = np.ndarray

    def __init__(self, model: BaseModel, inputs: Inputs, signal: List[str]) -> None:
        if model._built_model is None:
            raise ValueError("Only built models can be used in Observers")
        if not isinstance(inputs, dict):
            raise ValueError("Inputs must be of type Dict[str, float]")
        if not isinstance(signal, list):
            raise ValueError("Signal must be of type List[str]")

        if model.signal is None:
            model.signal = signal
        self._state = model.reinit(inputs)
        self._model = model
        self._signal = signal

    def reset(self, inputs: Inputs) -> None:
        self._state = self._model.reinit(inputs)

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
        self.reset(inputs)
        for t, v in zip(times, values):
            try:
                log_likelihood += self.observe(t, v)
            except ValueError:
                return np.float64(-np.inf)
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
        measures = [x.sol[s].data[-1] for s in self._signal]
        return np.array([[m] for m in measures])

    def get_current_time(self) -> float:
        """
        Returns the current time.
        """
        return self._state.t
