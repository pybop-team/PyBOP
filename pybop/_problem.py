import numpy as np


class SingleOutputProblem:
    """
    Defines a PyBOP single output problem, follows the PINTS interface.
    """

    def __init__(self, model, parameters, signal, dataset):
        self._model = model
        self.parameters = {o.name: o for o in parameters}
        self.signal = signal
        self._dataset = dataset

        if self._model._built_model is None:
            self._model.build(fit_parameters=self.parameters)

        for i, item in enumerate(self._dataset):
            if item.name == "Time [s]":
                self._time_data_available = True
                self._time_data = self._dataset[i].data

            if item.name == signal:
                self._ground_truth = self._dataset[i].data

        if self._time_data_available is False:
            raise ValueError("Dataset must contain time data")

        if np.any(self._time_data < 0):
            raise ValueError("Times can not be negative.")
        if np.any(self._time_data[:-1] >= self._time_data[1:]):
            raise ValueError("Times must be increasing.")

        if len(self._ground_truth) != len(self._time_data):
            raise ValueError("Time data and signal data must be the same length.")

    def evaluate(self, parameters):
        """
        Evaluate the model with the given parameters and return the signal.
        """

        y = np.asarray(
            self._model.simulate(inputs=parameters, t_eval=self.model.time_data)[
                self.signal
            ].data
        )

        return y

    def evaluateS1(self, parameters):
        """
        Evaluate the model with the given parameters and return the signal and
        its derivatives.
        """

        y, dy_dp = self._model.simulateS1(
            inputs=parameters, t_eval=self.model.time_data, calculate_sensitivities=True
        )[self.signal]

        return (np.asarray(y), np.asarray(dy_dp))
