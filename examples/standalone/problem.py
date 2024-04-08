import numpy as np
from pybop import BaseProblem


class StandaloneProblem(BaseProblem):
    """
    Defines an example standalone problem without a Model.
    """

    def __init__(
        self,
        parameters,
        dataset,
        model=None,
        check_model=True,
        signal=None,
        init_soc=None,
        x0=None,
    ):
        super().__init__(parameters, model, check_model, signal, init_soc, x0)
        self._dataset = dataset.data

        # Check that the dataset contains time and current
        for name in ["Time [s]"] + self.signal:
            if name not in self._dataset:
                raise ValueError(f"expected {name} in list of dataset")

        self._time_data = self._dataset["Time [s]"]
        self.n_time_data = len(self._time_data)
        if np.any(self._time_data < 0):
            raise ValueError("Times can not be negative.")
        if np.any(self._time_data[:-1] >= self._time_data[1:]):
            raise ValueError("Times must be increasing.")

        for signal in self.signal:
            if len(self._dataset[signal]) != self.n_time_data:
                raise ValueError(
                    f"Time data and {signal} data must be the same length."
                )
        target = [self._dataset[signal] for signal in self.signal]
        self._target = np.vstack(target).T

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

        return x[0] * self._time_data + x[1]

    def evaluateS1(self, x):
        """
        Evaluate the model with the given parameters and return the signal and its derivatives.

        Parameters
        ----------
        x : np.ndarray
            Parameter values to evaluate the model at.

        Returns
        -------
        tuple
            A tuple containing the simulation result y(t) and the sensitivities dy/dx(t) evaluated
            with given inputs x.
        """

        y = x[0] * self._time_data + x[1]

        dy = np.dstack([self._time_data, np.zeros(self._time_data.shape)])

        return (np.asarray(y), np.asarray(dy))
