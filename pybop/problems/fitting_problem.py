import warnings
from typing import Optional

import numpy as np

from pybop import BaseModel, BaseProblem, Dataset
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
        Additional variables to observe and store in the solution (default additions are: ["Time [s]"]).
    initial_state : dict, optional
        A valid initial state, e.g. the initial open-circuit voltage (default: None).

    Additional Attributes
    ---------------------
    dataset : dictionary
        The dictionary from a Dataset object containing the signal keys and values to fit the model to.
    time_data : np.ndarray
        The time points in the dataset.
    n_time_data : int
        The number of time points.
    target : np.ndarray
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
        initial_state: Optional[dict] = None,
    ):
        # Add time and remove duplicates
        additional_variables = additional_variables or []
        additional_variables.extend(["Time [s]"])
        additional_variables = list(set(additional_variables))

        if initial_state is not None and "Initial SoC" in initial_state.keys():
            warnings.warn(
                "It is usually better to define an initial open-circuit voltage as the "
                "initial_state for a FittingProblem because this value can typically be "
                "obtained from the data, unlike the intrinsic initial state of charge. "
                "In the case where the fitting parameters do not change the OCV-SOC "
                "relationship, the initial state of charge may be passed to the model "
                'using, for example, `model.set_initial_state({"Initial SoC": 1.0})` '
                "before constructing the FittingProblem.",
                UserWarning,
                stacklevel=1,
            )

        super().__init__(
            parameters, model, check_model, signal, additional_variables, initial_state
        )
        self._dataset = dataset.data
        self._n_parameters = len(self.parameters)

        # Check that the dataset contains necessary variables
        dataset.check([*self.signal, "Current function [A]"])

        # Unpack time and target data
        self._time_data = self._dataset["Time [s]"]
        self.n_time_data = len(self._time_data)
        self.set_target(dataset)

        if model is not None:
            # Build the model from scratch
            if self._model.built_model is not None:
                self._model.clear()
            self._model.build(
                dataset=self._dataset,
                parameters=self.parameters,
                check_model=self.check_model,
                initial_state=self.initial_state,
            )

    def evaluate(
        self, inputs: Inputs, update_capacity=False
    ) -> dict[str, np.ndarray[np.float64]]:
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

        try:
            sol = self._model.simulate(
                inputs=inputs, t_eval=self._time_data, initial_state=self.initial_state
            )
        except Exception as e:
            if self.verbose:
                print(f"Simulation error: {e}")
            return {signal: [np.inf] for signal in self.signal}

        return {
            signal: sol[signal].data
            for signal in (self.signal + self.additional_variables)
        }

    def evaluateS1(self, inputs: Inputs):
        """
        Evaluate the model with the given parameters and return the signal and its derivatives.

        Parameters
        ----------
        inputs : Inputs
            Parameters for evaluation of the model.

        Returns
        -------
        tuple[dict, np.ndarray]
            A tuple containing the simulation result y(t) as a dictionary and the sensitivities
            dy/dx(t) evaluated with given inputs.
        """
        inputs = self.parameters.verify(inputs)
        self.parameters.update(values=list(inputs.values()))

        try:
            sol = self._model.simulateS1(
                inputs=inputs, t_eval=self._time_data, initial_state=self.initial_state
            )
        except Exception as e:
            print(f"Error: {e}")
            return {signal: [np.inf] for signal in self.signal}, [np.inf]

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


class MultiFittingProblem(BaseProblem):
    """
    Problem class for joining mulitple fitting problems into one combined fitting problem.

    Extends `BaseProblem` in a similar way to FittingProblem but for multiple parameter
    estimation problems, which must first be defined individually.

    Additional Attributes
    ---------------------
    problems : List[pybop.FittingProblem]
        A list of PyBOP fitting problems.
    """

    def __init__(self, *args):
        self.problems = []
        for problem in args:
            if problem._model is not None:
                # Take a copy of the model and build from scratch
                problem._model = problem._model.new_copy()
                problem._model.build(
                    dataset=problem._dataset,
                    parameters=problem.parameters,
                    check_model=problem.check_model,
                    init_soc=problem.init_soc,
                )
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

    def evaluate(self, inputs: Inputs, **kwargs):
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
        self.parameters.update(values=list(inputs.values()))

        combined_signal = []

        for problem in self.problems:
            problem_inputs = problem.parameters.as_dict()
            signal_values = problem.evaluate(problem_inputs, **kwargs)

            # Collect signals
            for signal in problem.signal:
                combined_signal.extend(signal_values[signal])

        return {"Combined signal": np.asarray(combined_signal)}

    def evaluateS1(self, inputs: Inputs):
        """
        Evaluate the model with the given parameters and return the signal and its derivatives.

        Parameters
        ----------
        inputs : Inputs
            Parameters for evaluation of the model.

        Returns
        -------
        tuple[dict, np.ndarray]
            A tuple containing the simulation result y(t) as a dictionary and the sensitivities
            dy/dx(t) evaluated with given inputs.
        """
        inputs = self.parameters.verify(inputs)
        self.parameters.update(values=list(inputs.values()))

        combined_signal = []
        all_derivatives = []

        for problem in self.problems:
            problem_inputs = problem.parameters.as_dict()
            signal_values, dyi = problem.evaluateS1(problem_inputs)

            # Collect signals and derivatives
            for signal in problem.signal:
                combined_signal.extend(signal_values[signal])
            all_derivatives.append(dyi)

        y = {"Combined signal": np.asarray(combined_signal)}
        dy = np.concatenate(all_derivatives) if all_derivatives else None

        return (y, dy)
