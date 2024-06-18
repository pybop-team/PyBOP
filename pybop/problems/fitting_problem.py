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
        signal=["Voltage [V]"],
        additional_variables=[],
        init_soc=None,
    ):
        # Add time and remove duplicates
        additional_variables.extend(["Time [s]"])
        additional_variables = list(set(additional_variables))

        super().__init__(
            parameters, model, check_model, signal, additional_variables, init_soc
        )
        self._dataset = dataset.data
        self.x = self.x0

        # Check that the dataset contains time and current
        dataset.check(self.signal + ["Current function [A]"])

        # Unpack time and target data
        self._time_data = self._dataset["Time [s]"]
        self.n_time_data = len(self._time_data)
        self.set_target(dataset)

        # Add useful parameters to model
        if model is not None:
            self._model.signal = self.signal
            self._model.additional_variables = self.additional_variables
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
        if np.any(x != self.x) and self._model.rebuild_parameters:
            self.parameters.update(values=x)
            self._model.rebuild(parameters=self.parameters)
            self.x = x

        y = self._model.simulate(inputs=x, t_eval=self._time_data)

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
        if self._model.rebuild_parameters:
            raise RuntimeError(
                "Gradient not available when using geometric parameters."
            )

        y, dy = self._model.simulateS1(
            inputs=x,
            t_eval=self._time_data,
        )

        return (y, np.asarray(dy))
