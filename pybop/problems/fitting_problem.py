import numpy as np

from pybop import BaseProblem
from pybop.parameters.parameter import Inputs


class FittingProblem(BaseProblem):
    """
    Problem class for fitting (parameter estimation) problems.

    Extends `BaseProblem` with specifics for fitting a model to a dataset.

    Parameters
    ----------
    model : object
        The model to fit.
    parameters : pybop.Parameter or pybop.Parameters
        An object or list of the parameters for the problem.
    dataset : Dataset
        Dataset object containing the data to fit the model to.
    signal : str, optional
        The variable used for fitting (default: "Voltage [V]").
    additional_variables : List[str], optional
        Additional variables to observe and store in the solution (default additions are: ["Time [s]"]).
    init_soc : float, optional
        Initial state of charge (default: None).
    """

    def __init__(
        self,
        model,
        parameters,
        dataset,
        check_model=True,
        signal=None,
        additional_variables=None,
        init_soc=None,
    ):
        # Add time and remove duplicates
        if additional_variables is None:
            additional_variables = []
        if signal is None:
            signal = ["Voltage [V]"]
        additional_variables.extend(["Time [s]"])
        additional_variables = list(set(additional_variables))

        super().__init__(
            parameters, model, check_model, signal, additional_variables, init_soc
        )
        self._dataset = dataset.data
        self.parameters.initial_value()
        self._n_parameters = len(self.parameters)

        # Check that the dataset contains time and current
        dataset.check([*self.signal, "Current function [A]"])

        # Unpack time and target data
        self._time_data = self._dataset["Time [s]"]
        self.n_time_data = len(self._time_data)
        self.set_target(dataset)

        # Add useful parameters to model
        if model is not None:
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
        inputs = self.parameters.verify(inputs)

        sol = self._model.simulate(inputs=inputs, t_eval=self._time_data)

        if sol == [np.inf]:
            return {signal: [np.inf] for signal in self.signal}

        else:
            y = {
                signal: sol[signal].data
                for signal in (self.signal + self.additional_variables)
            }

            return y

    def evaluateS1(self, inputs: Inputs):
        """
        Evaluate the model with the given parameters and return the signal and its derivatives.

        Parameters
        ----------
        inputs : Inputs
            Parameters for evaluation of the model.

        Returns
        -------
        tuple
            A tuple containing the simulation result y(t) and the sensitivities dy/dx(t) evaluated
            with given inputs.
        """
        inputs = self.parameters.verify(inputs)

        sol = self._model.simulateS1(
            inputs=inputs,
            t_eval=self._time_data,
        )

        if sol == [np.inf]:
            return {signal: [np.inf] for signal in self.signal}, [np.inf]

        else:
            y = {signal: sol[signal].data for signal in self.signal}

            # Extract the sensitivities and stack them along a new axis for each signal
            dy = np.empty(
                (
                    sol[self.signal[0]].data.shape[0],
                    self.n_outputs,
                    self._n_parameters,
                )
            )

            for i, signal in enumerate(self.signal):
                dy[:, i, :] = np.stack(
                    [
                        sol[signal].sensitivities[key].toarray()[:, 0]
                        for key in self.parameters.keys()
                    ],
                    axis=-1,
                )

            return (y, np.asarray(dy))
