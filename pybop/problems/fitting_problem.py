import numpy as np
from pybop import BaseProblem


class FittingProblem(BaseProblem):
    """
    Problem class for fitting (parameter estimation) problems.

    Extends `BaseProblem` with specifics for fitting a model to a dataset.

    Parameters
    ----------
    model : object
        The model to fit.
    parameters : list
        List of parameters for the problem.
    dataset : Dataset
        Dataset object containing the data to fit the model to.
    signal : str, optional
        The signal to fit (default: "Voltage [V]").
    """

    def __init__(
        self,
        model,
        parameters,
        dataset,
        check_model=True,
        signal=["Voltage [V]"],
        init_soc=None,
        x0=None,
    ):
        super().__init__(parameters, model, check_model, signal, init_soc, x0)
        self._dataset = dataset.data
        self.x = self.x0

        # Check that the dataset contains time and current
        for name in ["Time [s]", "Current function [A]"] + self.signal:
            if name not in self._dataset:
                raise ValueError(f"Expected {name} in list of dataset")

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

        # Add useful parameters to model
        if model is not None:
            self._model.signal = self.signal
            self._model.n_outputs = self.n_outputs
            self._model.n_time_data = self.n_time_data

            # Build the model from scratch
            if self._model._built_model is not None:
                self._model._model_with_set_params = None
                self._model._built_model = None
                self._model._built_initial_soc = None
                self._model._mesh = None
                self._model._disc = None
            self._model.build(
                dataset=self._dataset,
                parameters=self.parameters,
                check_model=self.check_model,
                init_soc=self.init_soc,
            )

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
        if (x != self.x).any() and self._model.matched_parameters:
            for i, param in enumerate(self.parameters):
                param.update(value=x[i])

            self._model.rebuild(parameters=self.parameters)
            self.x = x

        y = np.asarray(self._model.simulate(inputs=x, t_eval=self._time_data))

        return y

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
        if self._model.matched_parameters:
            raise RuntimeError(
                "Gradient not available when using geometric parameters."
            )

        y, dy = self._model.simulateS1(
            inputs=x,
            t_eval=self._time_data,
        )

        return (np.asarray(y), np.asarray(dy))
