from typing import List, Optional

import numpy as np

from pybop._problem import BaseProblem
from pybop.models.base_model import BaseModel, Inputs, TimeSeriesState
from pybop.parameters.parameter import Parameter


class Observer(BaseProblem):
    """
    An observer of a time series state. Observers:
     1. keep track of the distribution of a current time series model state
     2. predict forward in time the distribution of the state
     3. update the distribution of the state with new observations

    Parameters
    ----------
    parameters : list
        List of parameters for the problem.
    model : BaseModel
      The model to observe.
    check_model : bool, optional
        Flag to indicate if the model should be checked (default: True).
    signal: List[str]
      The signal to observe.
    init_soc : float, optional
        Initial state of charge (default: None).
    x0 : np.ndarray, optional
        Initial parameter values (default: None).
    """

    # define a subtype for covariance matrices for use by derived classes
    Covariance = np.ndarray

    def __init__(
        self,
        parameters: List[Parameter],
        model: BaseModel,
        check_model=True,
        signal=["Voltage [V]"],
        init_soc=None,
        x0=None,
    ) -> None:
        super().__init__(parameters, model, check_model, signal, init_soc, x0)
        if model._built_model is None:
            raise ValueError("Only built models can be used in Observers")
        if model.signal is None:
            model.signal = self.signal

        inputs = dict()
        for param in self.parameters:
            inputs[param.name] = param.value

        self._state = model.reinit(inputs)
        self._model = model
        self._signal = self.signal

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
        times : np.ndarray
            The times at which to observe the model.
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
            except Exception:
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

    def get_current_covariance(self) -> Covariance:
        """
        Returns the current covariance of the model.
        """
        n = len(self._state)
        return np.zeros((n, n))

    def get_measure(self, x: TimeSeriesState) -> np.ndarray:
        measures = [x.sol[s].data[-1] for s in self._signal]
        return np.array([[m] for m in measures])

    def get_current_time(self) -> float:
        """
        Returns the current time.
        """
        return self._state.t

    def evaluate(self, x):
        """
        Evaluate the model with the given parameters and return the signal.

        Parameters
        ----------
        x : np.ndarray
            Parameter values to evaluate the model at.

        Returns
        -------
        y : np.ndarray
            The model output y(t) simulated with inputs x.
        """
        inputs = dict()
        if isinstance(x[0], Parameter):
            for param in x:
                inputs[param.name] = param.value
        else:  # x is an array of parameter values
            for i, param in enumerate(self.parameters):
                inputs[param.name] = x[i]
        self.reset(inputs)

        output = []
        if hasattr(self, "_dataset"):
            ym = self._target
            for i, t in enumerate(self._time_data):
                self.observe(t, ym[i])
                ys = self.get_current_measure()
                output.append(ys)
        else:
            for t in self._time_data:
                self.observe(t)
                ys = self.get_current_measure()
                output.append(ys)

        return np.vstack(output)
