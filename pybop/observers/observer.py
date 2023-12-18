from typing import Callable
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
    measure : Callable[[TimeSeriesState], np.ndarray]
      The measurement function.
    sigma0 : np.ndarray
      The covariance matrix of the initial state.
    """

    Covariance = np.ndarray

    def __init__(self, model: BaseModel) -> None:
        self._state = model.reinit()
        self._model = model

    def reset(self) -> None:
        self._state = self._model.reinit()

    def observe(self, value: np.ndarray, time: float) -> None:
        """
        Predict the time series model until t = `time` and observe the measurement `value`.

        Parameters
        ----------
        value : np.ndarray
          The new observation.

        time : float
          The time of the new observation.
        """
        if time < self._state.t:
            raise ValueError("Time must be increasing.")

        self._state = self._model.predict(self._state, time)

    def get_current_state(self) -> TimeSeriesState:
        """
        Returns the current state of the model.
        """
        return self._state
