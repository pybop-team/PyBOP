from typing import Dict, Optional
import numpy as np
from pybop.models.base_model import BaseModel, TimeSeriesState


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

    def observe(self, time: float, value: Optional[np.ndarray] = None) -> None:
        """
        Predict the time series model until t = `time` and optionally observe the measurement `value`.

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
