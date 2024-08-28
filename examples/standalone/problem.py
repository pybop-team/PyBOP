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
        additional_variables=None,
        initial_state=None,
    ):
        super().__init__(parameters, model, check_model, signal, additional_variables)
        self._dataset = dataset.data

        # Check that the dataset contains time and current
        for name in ["Time [s]", *self.signal]:
            if name not in self._dataset:
                raise ValueError(f"expected {name} in list of dataset")

        self._domain_data = self._dataset[self.domain]
        self.n_data = len(self._domain_data)
        if np.any(self._domain_data < 0):
            raise ValueError("Times can not be negative.")
        if np.any(self._domain_data[:-1] >= self._domain_data[1:]):
            raise ValueError("Times must be increasing.")

        for signal in self.signal:
            if len(self._dataset[signal]) != self.n_data:
                raise ValueError(
                    f"Time data and {signal} data must be the same length."
                )
        self._target = {signal: self._dataset[signal] for signal in self.signal}

    def evaluate(self, inputs, **kwargs):
        """
        Evaluate the model with the given parameters and return the signal.

        Parameters
        ----------
        inputs : Dict
            Parameters for evaluation of the model.

        Returns
        -------
        y : np.ndarray
            The model output y(t) simulated with given inputs.
        """

        return {
            signal: inputs["Gradient"] * self._domain_data + inputs["Intercept"]
            for signal in self.signal
        }

    def evaluateS1(self, inputs):
        """
        Evaluate the model with the given parameters and return the signal and its derivatives.

        Parameters
        ----------
        inputs : Dict
            Parameters for evaluation of the model.

        Returns
        -------
        tuple
            A tuple containing the simulation result y(t) and the sensitivities dy/dx(t) evaluated
            with given inputs x.
        """

        y = self.evaluate(inputs)

        dy = np.zeros((self.n_data, self.n_outputs, self.n_parameters))
        dy[:, 0, 0] = self._domain_data

        return (y, dy)
