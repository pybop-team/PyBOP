from typing import Optional

import numpy as np

from pybop import BaseModel, BaseProblem
from pybop._dataset import Dataset
from pybop.parameters.parameter import Inputs, Parameters


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
    check_model : bool, optional
        Flag to indicate if the model should be checked (default: True).
    signal : str, optional
        The variable used for fitting (default: "Voltage [V]").
    additional_variables : list[str], optional
        Additional variables to observe and store in the solution (default: []).
    init_soc : float, optional
        Initial state of charge (default: None).

    Additional Attributes
    ---------------------
    _dataset : dictionary
        The dictionary from a Dataset object containing the signal keys and values to fit the model to.
    _time_data : np.ndarray
        The time points in the dataset.
    n_time_data : int
        The number of time points.
    _target : np.ndarray
        The target values of the signals.
    """

    def __init__(
        self,
        model: BaseModel,
        parameters: Parameters,
        dataset: Dataset,
        check_model: bool = True,
        signal: Optional[list[str]] = None,
        additional_variables: Optional[list[str]] = None,
        init_soc: Optional[float] = None,
    ):
        if signal is None:
            signal = ["Voltage [V]"]

        super().__init__(
            parameters, model, check_model, signal, additional_variables, init_soc
        )
        self._dataset = dataset.data
        self.parameters.initial_value()

        # Check that the dataset contains necessary variables
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

    def _evaluate(self, inputs: Inputs):
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
        requires_rebuild = False
        for key, value in inputs.items():
            if key in self._model.rebuild_parameters:
                current_value = self.parameters[key].value
                if value != current_value:
                    self.parameters[key].update(value=value)
                    requires_rebuild = True

        if requires_rebuild:
            self._model.rebuild(parameters=self.parameters)

        return self._model._simulate(
            inputs=inputs, t_eval=self._time_data, output_variables=self.variables
        )

    def _evaluateS1(self, inputs: Inputs):
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
        if self._model.rebuild_parameters:
            raise RuntimeError(
                "Gradient not available when using geometric parameters."
            )

        return self._model._simulateS1(
            inputs=inputs, t_eval=self._time_data, output_variables=self.variables
        )


class MultiFittingProblem(BaseProblem):
    """
    Problem class for joining mulitple fitting problems into one combined fitting problem.

    Extends `BaseProblem` in a similar way to FittingProblem but for multiple parameter
    estimation problems, which must first be defined individually.

    Additional Attributes
    ---------------------
    problems : list[pybop.FittingProblem]
        A list of PyBOP fitting problems.
    """

    def __init__(self, *args):
        self.problems = []
        for problem in args:
            self.problems.append(problem)

        # Compile the set of parameters, ignoring duplicates
        combined_parameters = Parameters()
        for problem in self.problems:
            combined_parameters.join(problem.parameters)

        # Combine the target datasets
        combined_time_data = []
        combined_signal = []
        for problem in self.problems:
            for signal in problem.signal:
                combined_time_data.extend(problem._time_data)
                combined_signal.extend(problem._target[signal])
        combined_dataset = Dataset(
            {
                "Time [s]": np.asarray(combined_time_data),
                "Combined signal": np.asarray(combined_signal),
            }
        )

        super().__init__(
            parameters=combined_parameters,
            model=None,
            signal=["Combined signal"],
        )
        self._dataset = combined_dataset.data
        self.parameters.initial_value()

        # Unpack time and target data
        self._time_data = self._dataset["Time [s]"]
        self.n_time_data = len(self._time_data)
        self.set_target(combined_dataset)

    def _evaluate(self, inputs: Inputs):
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
        self.parameters.update(values=list(inputs.values()))

        combined_signal = []

        for problem in self.problems:
            problem_inputs = problem.parameters.as_dict()
            signal_values = problem._evaluate(problem_inputs)

            # Collect signals
            for signal in problem.signal:
                combined_signal.extend(signal_values[signal])

        return {"Combined signal": np.asarray(combined_signal)}

    def _evaluateS1(self, inputs: Inputs):
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
        self.parameters.update(values=list(inputs.values()))

        combined_signal = []
        all_derivatives = []

        for problem in self.problems:
            problem_inputs = problem.parameters.as_dict()
            signal_values, dyi = problem._evaluateS1(problem_inputs)

            # Collect signals and derivatives
            for signal in problem.signal:
                combined_signal.extend(signal_values[signal])
            all_derivatives.append(dyi)

        y = {"Combined signal": np.asarray(combined_signal)}
        dy = np.concatenate(all_derivatives) if all_derivatives else None

        return (y, dy)
