from typing import Optional

import numpy as np

from pybop import BaseProblem
from pybop.models.base_model import BaseModel, Inputs, TimeSeriesState
from pybop.parameters.parameter import Parameters


class Observer(BaseProblem):
    """
    An observer of a time series state. Observers:
     1. keep track of the distribution of a current time series model state
     2. predict forward in time the distribution of the state
     3. update the distribution of the state with new observations

    Parameters
    ----------
    parameters : pybop.Parameter or pybop.Parameters
        An object or list of the parameters for the problem.
    model : BaseModel
      The model to observe.
    check_model : bool, optional
        Flag to indicate if the model should be checked (default: True).
    signal: List[str]
      The signal to observe.
    additional_variables : List[str], optional
        Additional variables to observe and store in the solution (default: []).
    init_soc : float, optional
        Initial state of charge (default: None).
    """

    # define a subtype for covariance matrices for use by derived classes
    Covariance = np.ndarray

    def __init__(
        self,
        parameters: Parameters,
        model: BaseModel,
        check_model=True,
        signal=None,
        additional_variables=None,
        init_soc=None,
    ) -> None:
        if additional_variables is None:
            additional_variables = []
        if signal is None:
            signal = ["Voltage [V]"]
        super().__init__(
            parameters, model, check_model, signal, additional_variables, init_soc
        )
        if model._built_model is None:
            raise ValueError("Only built models can be used in Observers")
        if model.signal is None:
            model.signal = self.signal

        inputs = self.parameters.as_dict("initial")
        self._state = model.reinit(inputs)
        self._model = model
        self._signal = self.signal
        self._n_outputs = len(self._signal)

    def reset(self, inputs: Inputs) -> None:
        inputs = self.parameters.verify(inputs)

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

    def log_likelihood(self, values: dict, times: np.ndarray, inputs: Inputs) -> float:
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
        inputs = self.parameters.verify(inputs)

        if self._n_outputs == 1:
            signal = self._signal[0]
            if len(values[signal]) != len(times):
                raise ValueError("values and times must have the same length.")
            log_likelihood = 0.0
            self.reset(inputs)
            for t, v in zip(times, values[signal]):
                try:
                    log_likelihood += self.observe(t, v)
                except Exception:
                    return np.float64(-np.inf)
            return log_likelihood
        else:
            raise ValueError(
                "Observer.log_likelihood is currently restricted to single output models."
            )

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
        return np.asarray([[m] for m in measures])

    def get_current_time(self) -> float:
        """
        Returns the current time.
        """
        return self._state.t

    def evaluate(self, inputs: Inputs):
        """
        Evaluate the model with the given parameters and return the signal.

        Parameters
        ----------
        inputs : Inputs
            Parameters for evaluation of the model.

        Returns
        -------
        y : np.ndarray
            The model output y(t) simulated with given inputs.
        """
        self.reset(inputs)

        output = {}
        ys = []
        if hasattr(self, "_dataset"):
            for signal in self._signal:
                ym = self._target[signal]
                for i, t in enumerate(self._time_data):
                    self.observe(t, ym[i])
                    ys.append(self.get_current_measure())
                output[signal] = np.vstack(ys)
        else:
            for signal in self._signal:
                for t in self._time_data:
                    self.observe(t)
                    ys.append(self.get_current_measure())
                output[signal] = np.vstack(ys)

        return output
